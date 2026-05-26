"""
Segmentation Expert-Mixture Regularization (SER)

Implementation of the SER framework proposed in:

Rashiwa, S. E., Branco, P.
"Segmentation Expert-Mixture Regularization:
An Adaptive Learning Method for Imbalanced Regression Problems"

CAIAC 2026
University of Ottawa

Author: Sarah Elyane Rashiwa
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import (
    HuberRegressor,
    QuantileRegressor,
    PoissonRegressor,
    Ridge,
    Lasso
)

from sklearn.preprocessing import PolynomialFeatures



# ---------- Utils ----------
def _to_numpy(x):
    """Convert input to NumPy array while preserving order."""
    return x.values if hasattr(x, "values") else np.asarray(x)

def _add_const_df(X):
    """Add intercept column for statsmodels."""
    if hasattr(X, "assign"):
        X_center = sm.add_constant(X, has_constant="add")
    else:
        X_center = sm.add_constant(X, has_constant="add")
    return X_center

def _safe_weights(resid, eps=1e-6):
    """Safe WLS weights."""
    r = _to_numpy(resid)
    return 1.0 / (r**2 + eps)

def metrics_table(y_true, y_pred, label):
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return pd.DataFrame([{
        "Model": label,
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
    }])

def mad(x: np.ndarray) -> float:
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

# ==========================================================
# Model Fitting Functions
# ==========================================================
def fit_ols(X_group: pd.DataFrame, y_group: pd.Series, cov_type="HC3") -> dict:
    X_center = _add_const_df(X_group)
    model = sm.OLS(y_group, X_center).fit(cov_type=cov_type)
    yhat = model.predict(X_center)
    return {"model": model, "y_pred": _to_numpy(yhat), "name": "OLS"}
def fit_wls(X_group, y_group, resid_from_ols, cov_type="HC3"):
    X_center = _add_const_df(X_group)
    w = _safe_weights(resid_from_ols)
    model = sm.WLS(y_group, X_center, weights=w).fit(cov_type=cov_type)
    yhat = model.predict(X_center)
    return {"model": model, "y_pred": _to_numpy(yhat), "name": "WLS"}


def fit_polynomial(X_group, y_group, degree=2):
    poly = PolynomialFeatures(degree=degree)
    Xg_poly = poly.fit_transform(X_group)
    model = sm.OLS(y_group, Xg_poly).fit()
    yhat = model.predict(Xg_poly)
    return {"model": model, "y_pred": _to_numpy(yhat), "name": f"Poly ({degree})"}

def fit_huber(X_group, y_group, epsilon=1.35, alpha=0.0):
    huber = HuberRegressor(epsilon=epsilon, alpha=alpha)
    huber.fit(_to_numpy(X_group), _to_numpy(y_group))
    yhat = huber.predict(_to_numpy(X_group))
    return {"model": huber, "y_pred": _to_numpy(yhat), "name": "Huber"}
def fit_poisson(X_group, y_group):
    poisson = PoissonRegressor()
    poisson.fit(_to_numpy(X_group), _to_numpy(y_group))
    yhat = poisson.predict(_to_numpy(X_group))
    return {"model": poisson, "y_pred": _to_numpy(yhat), "name": "Poisson"}
def ridge(X_group, y_group, alpha=1.0):
    ridge = Ridge(alpha=alpha)
    ridge.fit(_to_numpy(X_group), _to_numpy(y_group))
    yhat = ridge.predict(_to_numpy(X_group))
    return {"model": ridge, "y_pred": _to_numpy(yhat), "name": "Ridge"}
def lasso(X_group, y_group, alpha=1.0):
    lasso = Lasso(alpha=alpha)
    lasso.fit(_to_numpy(X_group), _to_numpy(y_group))
    yhat = lasso.predict(_to_numpy(X_group))
    return {"model": lasso, "y_pred": _to_numpy(yhat), "name": "Lasso"}

def fit_quantile(X_group, y_group, tau=0.10, alpha=0.0):
    qr = QuantileRegressor(quantile=tau, alpha=alpha, solver="highs")
    qr.fit(_to_numpy(X_group), _to_numpy(y_group))
    yhat = qr.predict(_to_numpy(X_group))
    return {"model": qr, "y_pred": _to_numpy(yhat), "name": f"Quantile(τ={tau})"}
def predict_with_model(model, X_row, mdl_name):
    """
    Generic prediction function supporting:
    - StatsModels estimators
    - Scikit-learn estimators
    - Polynomial regression models
    """

    # Case 1 — Polynomial models

    if "Poly" in mdl_name:
        try:
            degree = int(mdl_name.split("(")[1].split(")")[0])
        except Exception:
            degree = 2  # Default fallback value
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_row)
        return model.predict(X_poly)[0]
    
   
    # Case 2 — Scikit-learn models

    elif hasattr(model, "coef_"):
        return model.predict(X_row.to_numpy())[0]

    # Case 3 — StatsModels models
    elif hasattr(model, "predict"):
        X_center = sm.add_constant(X_row, has_constant="add")
        return model.predict(X_center)[0]


    # Case unknown
    else:
        raise ValueError(f"Unknown model type : {type(model)}")


def fit_group_all_models(
    X_group,
    y_group,
    resid_from_ols=None,
    quantile_tau=0.5,
    verbose=False
):
    """
    Train multiple regression models on a target segment
    and select the best-performing model according to R².

    Returns
    -------
    best_model : dict
        Dictionary containing:
        - model name
        - fitted estimator
        - R²
        - MAE
        - RMSE

    df_perf : pandas.DataFrame
        Performance table for all candidate models.
    """
    # Sécurisation des formats
    X_group = pd.DataFrame(X_group).copy()
    X_group.index = range(len(X_group))
    y_group = pd.Series(y_group).copy()
    y_group.index = range(len(y_group))

    candidates = []

    # 1) OLS
    m_ols = fit_ols(X_group, y_group)
    candidates.append(m_ols)

    # 2) WLS (si résidus fournis)
    if resid_from_ols is not None:
        try:
            m_wls = fit_wls(X_group, y_group, resid_from_ols)
            candidates.append(m_wls)
        except Exception:
            # Ignore WLS failure silently
            pass



    # 4) Polynômial (degré 2)
    candidates.append(fit_polynomial(X_group, y_group, degree=2))


    # 5) Huber
    candidates.append(fit_huber(X_group, y_group))

    # 6) Poisson
    try:
        candidates.append(fit_poisson(X_group, y_group))
    except Exception:
        pass

    # 7) Quantile
    candidates.append(fit_quantile(X_group, y_group, tau=quantile_tau))
    # 8) Ridge
    candidates.append(ridge(X_group, y_group))
    #9)Lasso
    candidates.append(lasso(X_group, y_group))

    # Évaluation de tous les modèles
    results = []
    y_true = _to_numpy(y_group)
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


    
    return best_model, df_perf


def safe_fit_group(X_group, y_group, label, tau, resid_from_ols=None, min_n=10, verbose=False):
    """
    Safety wrapper for group fitting.

    Groups smaller than `min_n` are ignored.
    """
    n = 0 if X_group is None else len(X_group)
    if X_group is None or n < min_n:
        if verbose:
            print(f"Groupe {label} ignored (n={n})")
        return None, None



    best_model, perf_df = fit_group_all_models(
        X_group,
        y_group,
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



# ==========================================================
# Segmentation Methods
# ==========================================================
def tail_mean_check(res, mask_tail, mask_center, alpha=0.15):
    """Return True if the tail region has a significantly different mean compared to the center region. """
    eps = 1e-12
    mu_tail   = res[mask_tail].mean() if mask_tail.any() else 0.0
    mu_center = res[mask_center].mean() if mask_center.any() else 0.0
    return abs(mu_tail - mu_center) > alpha * (res.std() + eps)

def split_by_target_thresholds(y, low_thr, up_thr):
    """
    Split target values into lower, center, and upper regions.
    """
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
    Segment the target variable into three regions:
    - center region around the median (MAD-based)
    - lower tail
    - upper tail
    """
    y = np.asarray(y)
    med = np.median(y)
    tau_c = k_tau * mad(y)

    low_thr = med - tau_c
    up_thr  = med + tau_c

    # Création des groupes
    groups, idxL, idxC, idxU = split_by_target_thresholds(y, low_thr, up_thr)

    # Test de différence moyenne queue vs centre
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
    Iterative trimming segmentation based on the target variable.

    Progressively removes extreme target values from the lower
    or upper tail and searches for the optimal segmentation threshold.
    """
    y_arr = np.asarray(y)
    order = np.argsort(y_arr)  # indices triés par valeur de y
    perf = []
    n = len(y_arr)
    best = None

    if cote == 'gauche':
        iterator = range(0, n // 3, pas)
    else:
        iterator = range(0, n // 3, pas)

    for i in iterator:
        if cote == 'gauche':
            kept = order[i:]  # Retire les i plus petites valeurs de y
            bound = y_arr[order[i]] if i < len(order) else y_arr.min()
        else:
            kept = order[:len(order) - i]  # Retire les i plus grandes valeurs de y
            j = len(order) - i - 1
            bound = y_arr[order[j]] if j >= 0 else y_arr.max()

        if len(kept) <= X.shape[1] + 1:
            break  # Trop peu de données pour entraîner un modèle

        Xs = X.iloc[kept] if hasattr(X, 'iloc') else X[kept]
        ys = y.iloc[kept] if hasattr(y, 'iloc') else y[kept]
        model = sm.OLS(ys, sm.add_constant(Xs)).fit()

        perf.append((i, model.rsquared, np.mean(np.abs(model.resid)), bound))

        # Critère d'arrêt : amélioration faible
        if len(perf) > 1 and abs(perf[-1][1] - perf[-2][1]) < stop_delta:
            best = perf[-1]
            break

    # Sélection finale si aucune convergence par delta
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
    Optimise les bornes de segmentation de la variable cible y
    selon un critère biais–variance dans le segment central.
    Cherche les seuils low_thr / up_thr qui donnent un centre
    stable, homogène, et équilibré avec les queues.
    """
    y_arr = np.asarray(y)
    rows = []

    for ql in q_low_grid:
        for qu in q_up_grid:
            if ql >= qu:
                continue

            # 1. Seuils en quantiles sur y
            low_thr = np.quantile(y_arr, ql)
            up_thr  = np.quantile(y_arr, qu)

            # 2. Découpage en groupes selon la cible
            _, idxL, idxC, idxU = split_by_target_thresholds(y_arr, low_thr, up_thr)

            # 3. On exige un minimum d'observations dans chaque groupe
            if len(idxC) <= X.shape[1] + 1 or min(len(idxL), len(idxU)) < 20:
                continue

            # 4. Sous-ensemble central
            X_center = X.iloc[idxC] if hasattr(X, 'iloc') else X[idxC]
            y_center = y.iloc[idxC] if hasattr(y, 'iloc') else y[idxC]

            # 5. Modèle linéaire sur le centre
            model = sm.OLS(y_center, sm.add_constant(X_center)).fit()
            res_c = model.resid

            # 6. Calcul des métriques biais–variance
            var_c = np.var(res_c)
            bias_c = np.mean(res_c)
            score_vb = var_c + bias_c ** 2

            # 7. Pénalité d'équilibre de tailles
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
                "r2C": model.rsquared,
                "vb": score_vb,
                "bal": bal,
                "score": score_final
            })

    # 8. Sélection du meilleur couple de bornes
    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("score")
    best = df.iloc[0].to_dict()

    return best, df

# ==========================================================
# Blending and Inference
# ==========================================================


def blended_predict_full_noleak(X_test, global_model, seg_thr, group_models, alpha):
    """
    Prédiction continue avec α-blending sur Lower↔Center↔Upper,
    en décidant des zones UNIQUEMENT à partir de la prédiction
    du modèle global (ŷ_global). -> Pas de fuite d'information.
    """
    low_thr, up_thr = seg_thr["low_thr"], seg_thr["up_thr"]
    # 1) Prédictions globales sur X_test
    Xc_test = sm.add_constant(X_test, has_constant="add")
    y_hat_global = np.asarray(global_model.predict(Xc_test))

    # Safely get models, defaulting to Center if a specific group model is None
    mdl_C_info = group_models["Center"]
    mdl_L_info = group_models["Lower"] if group_models["Lower"] else mdl_C_info
    mdl_U_info = group_models["Upper"] if group_models["Upper"] else mdl_C_info

    y_preds = []
    for i, y_group in enumerate(y_hat_global):
        x_i = X_test.iloc[i:i+1]
        try:
            # 1) Zone Lower pure
            if y_group < low_thr - alpha:
                pred = predict_with_model(mdl_L_info["model"], x_i, mdl_L_info["name"])

            # 2) Transition Lower–Center
            elif low_thr - alpha <= y_group <= low_thr + alpha:
                w = (low_thr + alpha - y_group) / (2 * alpha)
                yL = predict_with_model(mdl_L_info["model"], x_i, mdl_L_info["name"])
                y_center = predict_with_model(mdl_C_info["model"], x_i, mdl_C_info["name"])
                pred = w * yL + (1 - w) * y_center

            # 3) Zone Center pure
            elif low_thr + alpha < y_group < up_thr - alpha:
                pred = predict_with_model(mdl_C_info["model"], x_i, mdl_C_info["name"])

            # 4) Transition Center–Upper
            elif up_thr - alpha <= y_group <= up_thr + alpha:
                w = (up_thr + alpha - y_group) / (2 * alpha)
                y_center = predict_with_model(mdl_C_info["model"], x_i, mdl_C_info["name"])
                yU = predict_with_model(mdl_U_info["model"], x_i, mdl_U_info["name"])
                pred = w * y_center + (1 - w) * yU

            # 5) Zone Upper pure
            else:
                pred = predict_with_model(mdl_U_info["model"], x_i, mdl_U_info["name"])

        except Exception as e:
            print(f"Erreur sur observation {i}: {e}")
            pred = np.nan

        y_preds.append(pred)

    return np.array(y_preds)

# ==========================================================
# SER Pipeline
# ==========================================================

def run_pipeline(
    X_train, y_train,
    X_test,  y_test,
    method,
    quantile_bounds=(0.15, 0.85),
    alpha_ratio = 0.05,
    verbose=True
):
    """
    Run the complete SER pipeline:
    1. Train global baseline model
    2. Apply target-space segmentation
    3. Train local experts
    4. Perform alpha-blended prediction
    5. Evaluate predictive performance

    Parameters
    ----------
    method : str
    Segmentation strategy:
    A = MAD-based
    B = Quantile-based
    C = Iterative trimming
    D = Bias-variance optimization
    """


    if verbose:
        print(f"\n==============================")
        print(f" PIPELINE – method {method}")
        print(f"==============================")
        print(f"Train: {len(X_train)}, Test: {len(X_test)}\n")

    # ==============================
    # 2. Modèle global naïf (OLS)
    # ==============================
    Xc_train = sm.add_constant(X_train, has_constant="add")
    mdl_global = sm.OLS(y_train, Xc_train).fit(cov_type="HC3")

    y_hat_train_global = pd.Series(mdl_global.predict(Xc_train)).reset_index(drop=True)
    y_train_ = pd.Series(y_train).reset_index(drop=True)

    resid_global = (y_train_ - y_hat_train_global).reset_index(drop=True)


    # ==============================
    # 3. SEGMENTATION
    # ==============================
    if method == "A":
        seg = segmentation_3conditions_target(y_train)
    elif method == "B":
        ql, qu = quantile_bounds
        seg = segmentation_target_quantiles(y_train)
    elif method == "C":
        # itérative gauche + droite
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
    elif method == "D":
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
        raise ValueError("method inconnue. Choix: 'A','B','C','D'.")
    alpha = alpha_ratio* (seg["up_thr"] - seg["low_thr"])  # zone de transition = 5% de l’intervalle global

    


    # ==============================
    # 4. EXTRACTION DES GROUPES
    # ==============================
    idxL, idxC, idxU = seg["idxL"], seg["idxC"], seg["idxU"]

    X_L = X_train.iloc[idxL]
    y_L = y_train.iloc[idxL]

    X_C = X_train.iloc[idxC]
    y_C = y_train.iloc[idxC]

    X_U = X_train.iloc[idxU]
    y_U = y_train.iloc[idxU]

    if verbose:
        print("\n Tailles des groupes :")
        print(f"Lower : {len(X_L)} / Center : {len(X_C)} / Upper : {len(X_U)}")


    # ==============================
    # 5. TRAIN DES EXPERTS
    # ==============================
   # Résidus DU GROUPE (pas tous les résidus du train)
    resid_L = resid_global.iloc[idxL].reset_index(drop=True)
    resid_C = resid_global.iloc[idxC].reset_index(drop=True)
    resid_U = resid_global.iloc[idxU].reset_index(drop=True)

    best_L, perf_L = safe_fit_group(X_L, y_L, "Lower",  tau=0.10, resid_from_ols=resid_L, verbose=verbose)
    best_C, perf_C = safe_fit_group(X_C, y_C, "Center", tau=0.50, resid_from_ols=resid_C, verbose=verbose)
    best_U, perf_U = safe_fit_group(X_U, y_U, "Upper",  tau=0.90, resid_from_ols=resid_U, verbose=verbose)

    group_models = {
        "Lower": best_L,
        "Center": best_C,
        "Upper": best_U
    }

    # Collect group performance dataframes
    group_perf_dfs = {
        "Lower": perf_L,
        "Center": perf_C,
        "Upper": perf_U
    }

    # === PATCH IMPORTANT ===


    if group_models["Lower"] is None:
        group_models["Lower"] = group_models["Center"]

    if group_models["Upper"] is None:
        group_models["Upper"] = group_models["Center"]


    # ==============================
    # 6. PREDICTION ()
    # ==============================
    y_pred_naive = mdl_global.predict(sm.add_constant(X_test, has_constant="add"))
    y_pred_blend = blended_predict_full_noleak(X_test, mdl_global, seg, group_models, alpha)

    # ==============================
    # 7. MÉTRIQUES
    # ==============================
    
    df_blend = metrics_table(y_test, y_pred_blend, "SER")
   

    

    if verbose:
        print("\n==============================")
        print(f" Final prediction (test set) - method : {method}")
        print("==============================")
        print(df_blend)

    return {
        "global_model": mdl_global,
        "seg": seg,
        "group_models": group_models,
        "pred_naive": y_pred_naive,
        "pred_blend": y_pred_blend,
        "metrics": df_blend,
        "group_perf_dfs": group_perf_dfs, # Added this line
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }