# Segmentation-Expert-Mixture-Regularization : An Adaptive Learning Method for Imbalanced Regression Problems - CAIAC2026

This repository has all the code used in the experiments carried out in the paper **"Segmentation Expert-Mixture Regularization: An Adaptive Learning Method for Imbalanced Regression Problems"** [1].

Segmentation Expert-Mixture Regularization (**SER**) is an algorithm-level framework for imbalanced regression. SER partitions the target space into lower, central, and upper regions, trains specialized regression experts for each region, and applies an alpha-blending regularization mechanism to obtain smooth predictions across segmentation boundaries.

This repository is organized as follows:

* `Code` folder - contains all the code for reproducing the experiments presented in the paper;
* `Data` folder - contains the regression data sets used in the experiments carried out;
* `Figures` folder - contains the figures obtained from the experimental evaluation;
* `Tables` folder - contains the tables obtained from the experimental evaluation;
* `Results` folder - contains the experimental outputs, metrics, and saved results.

### Requirements

The experimental design was implemented in Python. Both code and data are in a format suitable for a Python environment.

In order to replicate these experiments you will need a working installation of Python. Python 3.10 or higher is recommended.

In your Python installation you also need to install the following additional packages:

* numpy
* pandas
* scikit-learn
* scipy
* statsmodels
* matplotlib
* seaborn

All the above packages can be installed from PyPI. Essentially, you need to issue the following command:

    pip install numpy pandas scikit-learn scipy statsmodels matplotlib seaborn

If you are using a virtual environment, you can create and activate it before installing the required packages:

    python -m venv venv
    source venv/bin/activate

On Windows, use:

    python -m venv venv
    venv\Scripts\activate

Then install the required packages:

    pip install numpy pandas scikit-learn scipy statsmodels matplotlib seaborn

### Method

SER is designed for regression problems where the target variable has an imbalanced or heterogeneous distribution. In these settings, standard global regression models tend to prioritize high-density regions of the target space and may perform poorly on rare or extreme target values.

The SER framework combines three main components:

* target-space segmentation;
* region-specific expert models;
* alpha-blending regularization across segment boundaries.

The target variable is divided into three regions: lower, central, and upper. A separate regression expert is trained for each region. During prediction, a preliminary global model is used as a gating mechanism to determine which expert should be used. Near the segmentation boundaries, predictions from adjacent experts are blended to avoid discontinuities.

The paper considers four segmentation strategies:

* MAD-based segmentation;
* quantile-based segmentation;
* iterative trimming-based segmentation;
* bias-variance search-based segmentation.

### Data sets

The experiments are carried out using the following imbalanced regression data sets:

* Bank8FM;
* cpuSm;
* Boston Housing.

The datasets were obtained from the Imbalanced Regression Data Sets repository:

[2] Paula Branco, Luís Torgo, and Rita Ribeiro. **Imbalanced Regression Data Sets.**  
https://github.com/paobranco/Imbalanced-Regression-DataSets

The experiments use a fixed 70/30 train-test split and are repeated over different random seeds to improve reproducibility.
### Evaluation

The predictive performance of SER is compared against standard global regression baselines, including Ordinary Least Squares regression and Huber regression.

The evaluation metrics used are:

* coefficient of determination, R²;
* mean absolute error, MAE;
* root mean squared error, RMSE.

The experimental results show that SER improves predictive performance on larger imbalanced regression data sets, especially when rare or extreme target regions exhibit different predictive behavior from the central region. The results also show that small data sets may limit the effectiveness of local expert estimation when tail regions contain too few observations.

### Reproducing the experiments

To reproduce the experiments, clone this repository:

    git clone https://github.com/srash007/Segmentation-Expert-Mixture-Regularization.git
    cd Segmentation-Expert-Mixture-Regularization

Install the required packages:

    pip install numpy pandas scikit-learn scipy statsmodels matplotlib seaborn

Then run the experiment scripts contained in the `Code` folder.

Check the README files in each folder to see more detailed instructions on how to run the experiments.

---

### References

[1] Sarah Elyane Rashiwa and Paula Branco. **"Segmentation Expert-Mixture Regularization: An Adaptive Learning Method for Imbalanced Regression Problems."** The 39th Canadian Conference on Artificial Intelligence, CAIAC 2026.
[2] P. Branco, L. Torgo, and R. Ribeiro. **Imbalanced Regression Data Sets.**  
Available at: https://github.com/paobranco/Imbalanced-Regression-DataSets

