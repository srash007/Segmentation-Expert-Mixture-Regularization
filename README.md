# Segmentation Expert-Mixture Regularization (SER)

Implementation of the **Segmentation Expert-Mixture Regularization (SER)** framework proposed in:

> **Rashiwa, S. E., Branco, P.**  
> *Segmentation Expert-Mixture Regularization:  
> An Adaptive Learning Method for Imbalanced Regression Problems*  
> CAIAC 2026 — University of Ottawa

---

## Overview

Segmentation Expert-Mixture Regularization (SER) is an adaptive regression framework designed for **imbalanced regression problems** and heterogeneous target distributions.

The method combines:

- target-space segmentation;
- specialized local regression experts;
- adaptive expert selection;
- continuous α-blending regularization.

Unlike traditional global regression models, SER aims to improve predictive performance in underrepresented or extreme regions of the target space while maintaining strong overall generalization.

---

## SER Framework

```text
Target Distribution
        ↓
Target-Space Segmentation
        ↓
Local Expert Training
        ↓
Adaptive Expert Selection
        ↓
α-Blending Regularization
        ↓
Final Prediction
```

---

## Repository Structure

```text
Segmentation-Expert-Mixture-Regularization/
│
├── src/
│   └── ser.py
│
├── notebooks/
│   ├── SER_cpuSm.ipynb
│   ├── SER_Bank8FM.ipynb
│   └── SER_Boston.ipynb
│
├── data/
│
├── figures/
│
├── results/
│
├── requirements.txt
└── README.md
```

---

## Core Features

### Target-Space Segmentation

SER partitions the target variable into:

- Lower region;
- Center region;
- Upper region.

Four segmentation strategies are implemented:

| Method | Description |
|---|---|
| SER-A | MAD-based segmentation |
| SER-B | Quantile-based segmentation |
| SER-C | Iterative trimming segmentation |
| SER-D | Bias-variance optimized segmentation |

---

### Local Experts

Each target region is modeled using specialized regression experts.

Implemented models include:

- OLS Regression;
- Weighted Least Squares (WLS);
- Polynomial Regression;
- Huber Regression;
- Quantile Regression;
- Ridge Regression;
- Lasso Regression;
- Poisson Regression.

---

### α-Blending Regularization

SER uses continuous alpha-blending between neighboring regions to avoid abrupt prediction discontinuities.

The blending process improves:
- smoothness;
- robustness;
- transition stability.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/srash007/Segmentation-Expert-Mixture-Regularization.git
cd Segmentation-Expert-Mixture-Regularization
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

Main dependencies:

- Python 3.10+
- NumPy
- Pandas
- StatsModels
- Scikit-learn
- Matplotlib

---

## Running Experiments

Example using the cpuSm dataset:

```python
from ser import run_pipeline

results = run_pipeline(
    X_train,
    y_train,
    X_test,
    y_test,
    method="A"
)

print(results["metrics"])
```

---

## Experimental Datasets

Experiments were conducted on datasets from:

> Branco, P., Torgo, L., Ribeiro, R.  
> *Pre-processing approaches for imbalanced distributions in regression.*  
> Neurocomputing, 343, 76–99, 2019.

Repository:

- https://github.com/paobranco/Imbalanced-Regression-DataSets

Datasets used include:

- cpuSm
- Bank8FM
- Boston Housing

---

## Example Results

| Dataset | Model | R² |
|---|---|---|
| cpuSm | OLS | 0.7063 |
| cpuSm | SER-A | 0.9623 |
| Bank8FM | OLS | 0.6412 |
| Bank8FM | SER-A | 0.9174 |

---

## Notebooks

The notebooks reproduce the experiments and figures presented in the paper:

- `SER_cpuSm.ipynb`
- `SER_Bank8FM.ipynb`
- `SER_Boston.ipynb`

---

## Citation

If you use this work, please cite:

```bibtex
@article{rashiwa2026ser,
  title={Segmentation Expert-Mixture Regularization:
         An Adaptive Learning Method for Imbalanced Regression Problems},
  author={Rashiwa, Sarah Elyane and Branco, Paula},
  journal={CAIAC 2026},
  year={2026}
}
```

---

## Author

**Sarah Elyane Rashiwa**  
University of Ottawa  
Joint Honours B.Sc. Computer Science and Mathematics

---

## License

