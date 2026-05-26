# -*- coding: utf-8 -*-
"""
SER implementation module.

This file contains the core implementation of the Segmentation Expert-Mixture Regularization (SER) algorithm, including utility functions, regression experts, target-space segmentation strategies, alpha-blending prediction, and the main pipeline.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, HuberRegressor, QuantileRegressor,PoissonRegressor, Ridge,Lasso

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# ---------- Utilities ----------
def _to_numpy(x):
    """Ensures the input is converted to a NumPy array while preserving mask order."""
    return x.values if hasattr(x, "values") else np.asarray(x)

def _add_const_df(X):
    """Adds a statsmodels constant while preserving the DataFrame format when possible."""
    if hasattr(X, "assign"):
        Xc = sm.add_constant(X, has_constant="add")
    else:
        Xc = sm.add_constant(X, has_constant="add")
    return Xc

def _safe_weights(resid, eps=1e-6):
    """Computes WLS weights as 1 / (resid^2 + eps) to avoid division by zero."""
    r = _to_numpy(resid)
    return 1.0 / (r**2 + eps)

def metrics_table(y_true, y_pred, label):
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return pd.DataFrame([{
        "Modèle": label,
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
    }])

def mad(x):
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

#----------Models--------
def fit_ols(Xg, yg, cov_type="HC3"):
    Xc = _add_const_df(Xg)
    mdl = sm.OLS(yg, Xc).fit(cov_type=cov_type)
    yhat = mdl.predict(Xc)
    return {"model": mdl, "y_pred": _to_numpy(yhat), "name": "OLS"}
def fit_wls(Xg, yg, resid_from_ols, cov_type="HC3"):
    Xc = _add_const_df(Xg)
    w = _safe_weights(resid_from_ols)
    mdl = sm.WLS(yg, Xc, weights=w).fit(cov_type=cov_type)
    yhat = mdl.predict(Xc)
    return {"model": mdl, "y_pred": _to_numpy(yhat), "name": "WLS"}


def fit_polynomial(Xg, yg, degree=2):
    poly = PolynomialFeatures(degree=degree)
    Xg_poly = poly.fit_transform(Xg)
    mdl = sm.OLS(yg, Xg_poly).fit()
    yhat = mdl.predict(Xg_poly)
    return {"model": mdl, "y_pred": _to_numpy(yhat), "name": f"Poly ({degree})"}

def fit_huber(Xg, yg, epsilon=1.35, alpha=0.0):
    huber = HuberRegressor(epsilon=epsilon, alpha=alpha)
    huber.fit(_to_numpy(Xg), _to_numpy(yg))
    yhat = huber.predict(_to_numpy(Xg))
    return {"model": huber, "y_pred": _to_numpy(yhat), "name": "Huber"}
def fit_poisson(Xg, yg):
    poisson = PoissonRegressor()
    poisson.fit(_to_numpy(Xg), _to_numpy(yg))
    yhat = poisson.predict(_to_numpy(Xg))
    return {"model": poisson, "y_pred": _to_numpy(yhat), "name": "Poisson"}
def ridge(Xg, yg, alpha=1.0):
    ridge = Ridge(alpha=alpha)
    ridge.fit(_to_numpy(Xg), _to_numpy(yg))
    yhat = ridge.predict(_to_numpy(Xg))
    return {"model": ridge, "y_pred": _to_numpy(yhat), "name": "Ridge"}
def lasso(Xg, yg, alpha=1.0):
    lasso = Lasso(alpha=alpha)
    lasso.fit(_to_numpy(Xg), _to_numpy(yg))
    yhat = lasso.predict(_to_numpy(Xg))
    return {"model": lasso, "y_pred": _to_numpy(yhat), "name": "Lasso"}

def fit_quantile(Xg, yg, tau=0.10, alpha=0.0):
    qr = QuantileRegressor(quantile=tau, alpha=alpha, solver="highs")
    qr.fit(_to_numpy(Xg), _to_numpy(yg))
    yhat = qr.predict(_to_numpy(Xg))
    return {"model": qr, "y_pred": _to_numpy(yhat), "name": f"Quantile(τ={tau})"}
def predict_with_model(mdl, X_row, mdl_name):
    """
    Universal prediction function for different model types.
    It automatically adapts the structure of X_row depending on the model:
      - StatsModels (OLS, GLM, WLS)
      - Scikit-learn (Huber, Poisson, Quantile)
      - Polynomial models (Poly(n))
    """

    # Case 1 — Polynomial model: "Poly (2)"
    if "Poly" in mdl_name:
        try:
            degree = int(mdl_name.split("(")[1].split(")")[0])
        except:
            degree = 2  # Default value
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_row)
        return mdl.predict(X_poly)[0]

    # Case 2 — Scikit-learn models (Huber, Poisson, Quantile)
    elif hasattr(mdl, "coef_"):
        return mdl.predict(X_row.to_numpy())[0]

    # Case 3 — StatsModels models (OLS, WLS, GLM)
    elif hasattr(mdl, "predict"):
        Xc = sm.add_constant(X_row, has_constant="add")
        return mdl.predict(Xc)[0]


    # Unknown model type
    else:
        raise ValueError(f"Type de modèle non reconnu : {type(mdl)}")


def fit_group_all_models(
    Xg,
    yg,
    resid_from_ols=None,
    quantile_tau=0.5,
    verbose=False
):
    """
    Trains several candidate models on one region (Lower / Center / Upper)
    and selects the best one according to R².

    Returns:
      - best_model: dict {"name", "model", "R2", "MAE", "RMSE"}
      - df_perf: DataFrame containing the performance of all candidate models
    """
    # Format standardization
    Xg = pd.DataFrame(Xg).copy()
    Xg.index = range(len(Xg))
    yg = pd.Series(yg).copy()
    yg.index = range(len(yg))

    candidates = []

    # 1) OLS
    m_ols = fit_ols(Xg, yg)
    candidates.append(m_ols)

    # 2) WLS if residuals are provided
    if resid_from_ols is not None:
        try:
            m_wls = fit_wls(Xg, yg, resid_from_ols)
            candidates.append(m_wls)
        except Exception:
            # Silently ignore WLS if it fails
            pass



    # 4) Polynomial model (degree 2)
    candidates.append(fit_polynomial(Xg, yg, degree=2))


    # 5) Huber
    candidates.append(fit_huber(Xg, yg))

    # 6) Poisson
    try:
        candidates.append(fit_poisson(Xg, yg))
    except Exception:
        pass

    # 7) Quantile
    candidates.append(fit_quantile(Xg, yg, tau=quantile_tau))
    # 8) Ridge
    candidates.append(ridge(Xg, yg))
    # 9) Lasso
    candidates.append(lasso(Xg, yg))

    # Evaluate all candidate models
    results = []
    y_true = _to_numpy(yg)
    for c in candidates:
        y_pred = c["y_pred"]
        results.append({
            "name":  c["name"],
            "R2":    r2_score(y_true, y_pred),
            "MAE":   mean_absolute_error(y_true, y_pred),
            "RMSE":  np.sqrt(mean_squared_error(y_true, y_pred)),
            "model": c["model"]
        })

    df_perf = pd.DataFrame(results).sort_values("R2", ascending=False)
    best_row = df_perf.iloc[0]

    best_model = {
        "name":  best_row["name"],
        "model": best_row["model"],
        "R2":    best_row["R2"],
        "MAE":   best_row["MAE"],
        "RMSE":  best_row["RMSE"]
    }


    #print(f"The best model is: {best_model['name']}")
    return best_model, df_perf


def safe_fit_group(Xg, yg, label, tau, resid_from_ols=None, min_n=10, verbose=False):
    """
    Safety wrapper:
      - ignores the region if it is too small
      - calls fit_group_all_models
    Returns (best_model, df_perf) or (None, None).
    """
    n = 0 if Xg is None else len(Xg)
    if Xg is None or n < min_n:
        if verbose:
            print(f"Region {label} ignored (n={n})")
        return None, None

    #if verbose:
        #print(f"\nTraining region {label} (n={n})")

    best_model, perf_df = fit_group_all_models(
        Xg,
        yg,
        resid_from_ols=resid_from_ols,
        quantile_tau=tau,
        verbose=verbose
    )
    return best_model, perf_df

def fit_huber_global(X_train, y_train, X_test):
    huber = HuberRegressor(epsilon=1.35, alpha=0.0)
    huber.fit(X_train.to_numpy(), y_train.to_numpy())
    y_pred = huber.predict(X_test.to_numpy())
    return huber, y_pred


#-------Segmentation -------
def tail_mean_check(res, mask_tail, mask_center, alpha=0.15):
    """Returns True if the tail has a mean error distinct from the center."""
    eps = 1e-12
    mu_tail   = res[mask_tail].mean() if mask_tail.any() else 0.0
    mu_center = res[mask_center].mean() if mask_center.any() else 0.0
    return abs(mu_tail - mu_center) > alpha * (res.std() + eps)

def split_by_target_thresholds(y, low_thr, up_thr):
    """Splits y into three groups according to lower and upper thresholds."""
    y = np.asarray(y)
    mask_lower  = y <  low_thr
    mask_center = (y >= low_thr) & (y <= up_thr)
    mask_upper  = y >  up_thr
    idx_lower  = np.where(mask_lower)[0]
    idx_center = np.where(mask_center)[0]
    idx_upper  = np.where(mask_upper)[0]
    groups = np.zeros(len(y), dtype=int)
    groups[idx_lower]  = 1
    groups[idx_upper]  = 2
    groups[idx_center] = 3
    return groups, idx_lower, idx_center, idx_upper

def segmentation_3conditions_target(y, k_tau=1.5, alpha_tail=0.15):
    """
    Segments the target variable y into three groups:
    - center: values close to the median according to MAD
    - lower / upper tails: significant extreme values
    """
    y = np.asarray(y)
    med = np.median(y)
    tau_c = k_tau * mad(y)

    low_thr = med - tau_c
    up_thr  = med + tau_c

    # Create the groups
    groups, idxL, idxC, idxU = split_by_target_thresholds(y, low_thr, up_thr)

    # Test mean difference between each tail and the center
    keepL = tail_mean_check(y, np.isin(np.arange(len(y)), idxL), np.isin(np.arange(len(y)), idxC), alpha=alpha_tail)
    keepU = tail_mean_check(y, np.isin(np.arange(len(y)), idxU), np.isin(np.arange(len(y)), idxC), alpha=alpha_tail)

    if not keepL:
        groups[idxL] = 0
        idxC = np.union1d(idxC, idxL)
        idxL = np.array([], dtype=int)

    if not keepU:
        groups[idxU] = 0
        idxC = np.union1d(idxC, idxU)
        idxU = np.array([], dtype=int)

    return {
        "low_thr": low_thr,
        "up_thr": up_thr,
        "groups": groups,
        "idxL": idxL,
        "idxC": idxC,
        "idxU": idxU
    }

def segmentation_target_quantiles(y, q_low=0.10, q_up=0.90):
    y = np.asarray(y)
    low_thr = np.quantile(y, q_low)
    up_thr  = np.quantile(y, q_up)
    groups, idxL, idxC, idxU = split_by_target_thresholds(y, low_thr, up_thr)
    return {"low_thr": low_thr, "up_thr": up_thr, "groups": groups, "idxL": idxL, "idxC": idxC, "idxU": idxU}

def methode_iterative_retrait_target(X, y, cote='gauche', pas=50, stop_delta=1e-3):
    """
    Version based on the target variable, not on residuals.
    Progressively removes extreme target values from the lower or upper side.
    Returns the optimal boundary (low_thr or up_thr) that maximizes R².
    """
    y_arr = np.asarray(y)
    order = np.argsort(y_arr)  # indices sorted by target value
    perf = []
    n = len(y_arr)
    best = None

    if cote == 'gauche':
        iterator = range(0, n // 3, pas)
    else:
        iterator = range(0, n // 3, pas)

    for i in iterator:
        if cote == 'gauche':
            kept = order[i:]  # Removes the i smallest target values
            bound = y_arr[order[i]] if i < len(order) else y_arr.min()
        else:
            kept = order[:len(order) - i]  # Removes the i largest target values
            j = len(order) - i - 1
            bound = y_arr[order[j]] if j >= 0 else y_arr.max()

        if len(kept) <= X.shape[1] + 1:
            break  # Too few observations to train a model

        Xs = X.iloc[kept] if hasattr(X, 'iloc') else X[kept]
        ys = y.iloc[kept] if hasattr(y, 'iloc') else y[kept]
        model = sm.OLS(ys, sm.add_constant(Xs)).fit()

        perf.append((i, model.rsquared, np.mean(np.abs(model.resid)), bound))

        # Stopping criterion: weak improvement
        if len(perf) > 1 and abs(perf[-1][1] - perf[-2][1]) < stop_delta:
            best = perf[-1]
            break

    # Final selection if no convergence based on delta
    if not best and perf:
        best = max(perf, key=lambda t: t[1])
    if not best:
        return None

    if cote == 'gauche':
        return {"low_thr": best[3], "impr_points": best[0], "r2": best[1], "mae": best[2]}
    else:
        return {"up_thr": best[3], "impr_points": best[0], "r2": best[1], "mae": best[2]}

def optimise_biais_variance_target(
    X,
    y,
    q_low_grid=np.linspace(0.05, 0.30, 12),
    q_up_grid=np.linspace(0.70, 0.95, 12)
):
    """
    Optimizes the segmentation boundaries of the target variable y
    using a bias-variance criterion in the central segment.
    Searches for low_thr / up_thr values that produce a stable,
    homogeneous center balanced with the tails.
    """
    y_arr = np.asarray(y)
    rows = []

    for ql in q_low_grid:
        for qu in q_up_grid:
            if ql >= qu:
                continue

            # 1. Quantile thresholds on y
            low_thr = np.quantile(y_arr, ql)
            up_thr  = np.quantile(y_arr, qu)

            # 2. Split into groups according to the target
            _, idxL, idxC, idxU = split_by_target_thresholds(y_arr, low_thr, up_thr)

            # 3. Require a minimum number of observations in each group
            if len(idxC) <= X.shape[1] + 1 or min(len(idxL), len(idxU)) < 20:
                continue

            # 4. Central subset
            Xc = X.iloc[idxC] if hasattr(X, 'iloc') else X[idxC]
            yc = y.iloc[idxC] if hasattr(y, 'iloc') else y[idxC]

            # 5. Linear model on the center
            mdl = sm.OLS(yc, sm.add_constant(Xc)).fit()
            res_c = mdl.resid

            # 6. Compute bias-variance metrics
            var_c = np.var(res_c)
            bias_c = np.mean(res_c)
            score_vb = var_c + bias_c ** 2

            # 7. Size-balance penalty
            bal = min(len(idxL), len(idxC), len(idxU)) / max(len(idxL), len(idxC), len(idxU))
            score_final = score_vb * (1 + 0.1 * np.log(1.0 / max(bal, 1e-6)))

            rows.append({
                "q_low": ql,
                "q_up": qu,
                "low_thr": low_thr,
                "up_thr": up_thr,
                "nL": len(idxL),
                "nC": len(idxC),
                "nU": len(idxU),
                "r2C": mdl.rsquared,
                "vb": score_vb,
                "bal": bal,
                "score": score_final
            })

    # 8. Select the best pair of boundaries
    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("score")
    best = df.iloc[0].to_dict()

    return best, df


def blended_predict_full_noleak(X_test, global_model, seg_thr, group_models, alpha):
    """
    Continuous prediction with alpha-blending across Lower↔Center↔Upper,
    deciding regions ONLY from the global model prediction
    (y_hat_global). -> No information leakage.
    """
    low_thr, up_thr = seg_thr["low_thr"], seg_thr["up_thr"]
    # 1) Global predictions on X_test
    Xc_test = sm.add_constant(X_test, has_constant="add")
    y_hat_global = np.asarray(global_model.predict(Xc_test))

    # Safely get models, defaulting to Center if a specific group model is None
    mdl_C_info = group_models["Center"]
    mdl_L_info = group_models["Lower"] if group_models["Lower"] else mdl_C_info
    mdl_U_info = group_models["Upper"] if group_models["Upper"] else mdl_C_info

    y_preds = []
    for i, yg in enumerate(y_hat_global):
        x_i = X_test.iloc[i:i+1]
        try:
            # 1) Pure Lower region
            if yg < low_thr - alpha:
                pred = predict_with_model(mdl_L_info["model"], x_i, mdl_L_info["name"])

            # 2) Lower–Center transition
            elif low_thr - alpha <= yg <= low_thr + alpha:
                w = (low_thr + alpha - yg) / (2 * alpha)
                yL = predict_with_model(mdl_L_info["model"], x_i, mdl_L_info["name"])
                yC = predict_with_model(mdl_C_info["model"], x_i, mdl_C_info["name"])
                pred = w * yL + (1 - w) * yC

            # 3) Pure Center region
            elif low_thr + alpha < yg < up_thr - alpha:
                pred = predict_with_model(mdl_C_info["model"], x_i, mdl_C_info["name"])

            # 4) Center–Upper transition
            elif up_thr - alpha <= yg <= up_thr + alpha:
                w = (up_thr + alpha - yg) / (2 * alpha)
                yC = predict_with_model(mdl_C_info["model"], x_i, mdl_C_info["name"])
                yU = predict_with_model(mdl_U_info["model"], x_i, mdl_U_info["name"])
                pred = w * yC + (1 - w) * yU

            # 5) Pure Upper region
            else:
                pred = predict_with_model(mdl_U_info["model"], x_i, mdl_U_info["name"])

        except Exception as e:
            print(f"Error on observation {i}: {e}")
            pred = np.nan

        y_preds.append(pred)

    return np.array(y_preds)

def run_pipeline(
    X_train, y_train,
    X_test,  y_test,
    methode,
    quantiles_B=(0.15, 0.85),
    verbose=True
):
    """
    Full pipeline:
      - train/test split
      - global OLS model
      - segmentation according to A/B/C/D
      - fit lower/center/upper experts
      - blended prediction using continuous alpha-blending
      - returns summary and metrics

    methode = "A", "B", "C", "D"
    alpha = width of the transition zones
    """



    if verbose:
        print(f"\n==============================")
        print(f" PIPELINE – méthode {methode}")
        print(f"==============================")
        print(f"Train: {len(X_train)}, Test: {len(X_test)}\n")

    # ==============================
    # 2. Global naive model (OLS)
    # ==============================
    Xc_train = sm.add_constant(X_train, has_constant="add")
    mdl_global = sm.OLS(y_train, Xc_train).fit(cov_type="HC3")

    y_hat_train_global = pd.Series(mdl_global.predict(Xc_train)).reset_index(drop=True)
    y_train_ = pd.Series(y_train).reset_index(drop=True)

    resid_global = (y_train_ - y_hat_train_global).reset_index(drop=True)


    # ==============================
    # 3. SEGMENTATION
    # ==============================
    if methode == "A":
        seg = segmentation_3conditions_target(y_train)
    elif methode == "B":
        
        seg = segmentation_target_quantiles(y_train)
    elif methode == "C":
        # iterative lower + upper segmentation
        left  = methode_iterative_retrait_target(X_train, y_train, cote="gauche")
        right = methode_iterative_retrait_target(X_train, y_train, cote="droite")

        low_thr = left["low_thr"] if left else np.quantile(y_train, 0.10)
        up_thr  = right["up_thr"] if right else np.quantile(y_train, 0.90)

        seg = {
            "low_thr": low_thr,
            "up_thr": up_thr,
            "groups": split_by_target_thresholds(y_train, low_thr, up_thr)[0],
            "idxL": split_by_target_thresholds(y_train, low_thr, up_thr)[1],
            "idxC": split_by_target_thresholds(y_train, low_thr, up_thr)[2],
            "idxU": split_by_target_thresholds(y_train, low_thr, up_thr)[3]
        }
    elif methode == "D":
        best, _ = optimise_biais_variance_target(X_train, y_train)
        seg = {
            "low_thr": best["low_thr"],
            "up_thr": best["up_thr"],
            "groups": split_by_target_thresholds(y_train, best["low_thr"], best["up_thr"])[0],
            "idxL": split_by_target_thresholds(y_train, best["low_thr"], best["up_thr"])[1],
            "idxC": split_by_target_thresholds(y_train, best["low_thr"], best["up_thr"])[2],
            "idxU": split_by_target_thresholds(y_train, best["low_thr"], best["up_thr"])[3]
        }
    else:
        raise ValueError("Méthode inconnue. Choix: 'A','B','C','D'.")
    alpha = 0.05 * (seg["up_thr"] - seg["low_thr"])  # transition zone = 5% of the global interval

    mdl_huber, y_pred_huber = fit_huber_global(X_train, y_train, X_test)

    \


    # ==============================
    # 4. GROUP EXTRACTION
    # ==============================
    idxL, idxC, idxU = seg["idxL"], seg["idxC"], seg["idxU"]

    X_L = X_train.iloc[idxL]
    y_L = y_train.iloc[idxL]

    X_C = X_train.iloc[idxC]
    y_C = y_train.iloc[idxC]

    X_U = X_train.iloc[idxU]
    y_U = y_train.iloc[idxU]

    if verbose:
        print("\n Group sizes:")
        print(f"Lower : {len(X_L)} / Center : {len(X_C)} / Upper : {len(X_U)}")


    # ==============================
    # 5. TRAIN EXPERTS
    # ==============================
   # GROUP residuals, not all training residuals
    resid_L = resid_global.iloc[idxL].reset_index(drop=True)
    resid_C = resid_global.iloc[idxC].reset_index(drop=True)
    resid_U = resid_global.iloc[idxU].reset_index(drop=True)

    best_L, perf_L = safe_fit_group(X_L, y_L, "Lower",  tau=0.10, resid_from_ols=resid_L, verbose=verbose)
    best_C, perf_C = safe_fit_group(X_C, y_C, "Center", tau=0.50, resid_from_ols=resid_C, verbose=verbose)
    best_U, perf_U = safe_fit_group(X_U, y_U, "Upper",  tau=0.90, resid_from_ols=resid_U, verbose=verbose)

    group_models = {
        "Lower":  best_L,
        "Center": best_C,
        "Upper":  best_U
    }

    # Collect group performance dataframes
    group_perf_dfs = {
        "Lower": perf_L,
        "Center": perf_C,
        "Upper": perf_U
    }

    # === IMPORTANT PATCH ===


    if group_models["Lower"] is None:
        group_models["Lower"] = group_models["Center"]

    if group_models["Upper"] is None:
        group_models["Upper"] = group_models["Center"]


    # ==============================
    # 6. PREDICTION
    # ==============================
    y_pred_naive = mdl_global.predict(sm.add_constant(X_test, has_constant="add"))
    y_pred_blend = blended_predict_full_noleak(X_test, mdl_global, seg, group_models, alpha)

    # ==============================
    # 7. METRICS
    # ==============================
    df_naive = metrics_table(y_test, y_pred_naive, "Naïf")
    df_blend = metrics_table(y_test, y_pred_blend, "SER")
    df_huber = metrics_table(y_test, y_pred_huber, "Huber")

    df_all = pd.concat([df_naive, df_blend,df_huber], ignore_index=True)

    if verbose:
        print("\n==============================")
        print(f" Final summary (test set) - méthode : {methode}")
        print("==============================")
        print(df_all)

    return {
        "global_model": mdl_global,
        "seg": seg,
        "group_models": group_models,
        "pred_naive": y_pred_naive,
        "pred_blend": y_pred_blend,
        "metrics": df_all,
        "group_perf_dfs": group_perf_dfs, # Added this line
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }


