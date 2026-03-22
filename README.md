# Customer-Churn-Uplift-Model

## Live Demo
🚀 [Launch Dashboard](https://customerchurandupliftbypurva.streamlit.app/)

End-to-end **customer churn + uplift modeling** on a Telco dataset: from EDA and feature engineering, through churn prediction, to causal uplift modeling and ROI analysis for retention campaigns.

## Business Problem and Goals

- **Problem**: Telco providers lose a significant share of recurring revenue due to customer churn. Blanket discounts or offers are expensive and often wasted on customers who would stay anyway.
- **Goal**:  
  - Predict **who is likely to churn** (classification).  
  - Estimate **who actually changes behaviour when treated** (uplift / CATE), so retention budget is focused on persuadable customers only.
- **Key questions** this project answers:
  - Which customers are at highest risk of churn?
  - Among them, who should we **target with an offer**, and who should we **avoid** (\"Sleeping Dogs\" / negative uplift)?
  - What is the **expected incremental retention and ROI** if we target the top uplift segment (e.g. top 10–20%)?

## Environment and Prerequisites

- **Python version**: 3.11.9 (project virtual environment: `venv311`)
- **Recommended install method**: use the provided `requirements.txt` inside an isolated virtual environment.

### 1. Create and activate virtual environment

- **Windows (PowerShell)**:

```bash
cd path\to\Customer-Churn-Uplift-Model
python -m venv venv311
venv311\Scripts\Activate.ps1
```

If `Activate.ps1` is blocked, you may need to temporarily relax the execution policy in PowerShell (run as Administrator):

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Install dependencies

All required libraries are listed in `requirements.txt`. Run:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Key libraries used in the project include:

- **Core scientific stack**: `numpy`, `pandas`, `scipy`, `statsmodels`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Machine learning**: `scikit-learn`, `xgboost`, `imblearn`
- **Causal / uplift modeling**: `scikit-uplift`, `causalml`, `dowhy`
- **Notebooks and Jupyter stack**: `jupyter`, `notebook`, `jupyterlab`, `ipykernel`, `ipywidgets`
- **Model explainability**: `shap`

> **Note**: `requirements.txt` contains some pinned versions used for experimentation (for example duplicate entries for `pandas`, `numpy`, `scikit-learn`, `xgboost`, and `matplotlib`). When `pip install -r requirements.txt` is executed, the **last occurrence of each package** in the file wins. This means the effective versions will be:
> - `pandas==2.0.3`
> - `numpy==1.24.3`
> - `scikit-learn==1.3.0`
> - `xgboost==1.7.6`
> - `matplotlib==3.7.2`

If you run into binary / compilation issues on a different OS or Python version, consider:

- Creating a fresh environment with the same Python version (3.11.9).
- Installing heavy packages one-by-one, for example:

```bash
pip install numpy==1.24.3 pandas==2.0.3
pip install scikit-learn==1.3.0 xgboost==1.7.6
pip install scikit-uplift==0.5.1 causalml==0.13.0 dowhy==0.8
```

## Project Structure

project-root/
├── data/                  # Raw and processed datasets
│   ├── raw/               # Original downloads
│   └── processed/         # Cleaned CSVs (incl. telco_churn_clean_v2.csv)
├── notebooks/             # Jupyter notebooks for exploration & modeling
│   ├── 01_eda.ipynb
│   ├── 02_churn_prediction_modelling.ipynb
│   └── 03_uplift_analysis.ipynb
├── src/                   # Python scripts (for production / reuse)
│   ├── data_prep.py
│   ├── models.py
│   └── app.py             # For deployment / scoring
├── docs/                  # Images, write-ups, reports (screenshots of SHAP, Qini, uplift dist, etc.)
├── requirements.txt       # List of dependencies (see versions above)
├── README.md              # Main documentation
└── .gitignore             # Ignore temp files, venv, etc.

## Running the notebooks

1. **Activate the virtual environment** (`venv311`) and install dependencies as described above.
2. Start Jupyter Lab or Notebook from the project root:

```bash
jupyter lab
# or
jupyter notebook
```

3. Open the notebooks in order:
   - `01_eda.ipynb` – exploratory data analysis and feature engineering.
   - `02_churn_prediction_modelling.ipynb` – churn prediction models, SMOTE, XGBoost, SHAP, etc.
   - `03_uplift_analysis.ipynb` – uplift / treatment effect analysis (using `causalml`, `scikit-uplift`, etc.).

Make sure the selected kernel points to the `venv311` environment (Python 3.11.9) so that all imports resolve correctly.

## Modeling Overview and Key Ideas

- **EDA & Feature Engineering (`01_eda.ipynb`)**
  - Churn rate analysis by contract type, tenure, monthly charges, add-on services.
  - Business-features: `tenure_bin`, `TotalServices`, and RFM-inspired scores (`Recency_score`, `Frequency_score`, `Monetary_score`).
  - Cleaned dataset exported as `data/processed/telco_churn_clean_v2.csv`.

- **Churn Prediction (`02_churn_prediction_modelling.ipynb`)**
  - Train/validation split with **class imbalance handling** (e.g. SMOTE).
  - Baseline models (Logistic Regression) and gradient-boosted models (XGBoost).
  - Evaluation via ROC-AUC, confusion matrix, and calibration.
  - **Model explainability** with SHAP (feature importance, waterfall plots for individual customers).

- **Uplift Modeling (`03_uplift_analysis.ipynb`)**
  - Simulate a binary `treatment` flag (retention offer vs no offer) and introduce a heterogeneous treatment effect:
    - Short-tenure, month-to-month, high-charge customers are more likely to **respond** to treatment.
  - Build a **Two-Model uplift baseline**:
    - Separate churn models for treated vs control using `XGBClassifier`.
    - Individual uplift score = predicted churn if control − predicted churn if treated.
  - Train an **X-Learner** using `causalml`'s `BaseXRegressor` with `XGBRegressor`:
    - Estimates customer-level CATE (uplift) and captures non-linear heterogeneity.
  - Evaluate uplift models using:
    - Uplift curves, **Qini curves**, and KPIs such as **Qini AUC** and **AUUC** (via `sklift.metrics`).
    - **Top-decile / top-quantile analysis**: compare churn rates for treated vs control in the highest uplift segment.
  - **Cost-aware ROI simulation**:
    - Assume an offer cost (e.g. ₹/€ discount) and margin per retained customer.
    - Compute expected incremental revenue, campaign cost, incremental profit and ROI when targeting top X% ranked by uplift.
  - **Sensitivity analysis**:
    - Scale the assumed treatment effect (e.g. 0.5×, 1.0×, 1.5×) to see how ROI changes under weaker/stronger effects.
  - **CATE / uplift visualisation**:
    - Plotly histogram for the distribution of individual uplift scores.
    - Highlights not only high-uplift customers but also **\"Sleeping Dogs\"** (negative uplift customers who may react badly to offers).

## Why This Project Is Portfolio-Ready

- **End-to-end**: covers data cleaning, EDA, supervised modeling, causal uplift modeling, evaluation and business translation.
- **Modern stack**: XGBoost, SHAP, CausalML, scikit-uplift metrics, interactive Plotly visualisation.
- **Business focus**: moves beyond \"who will churn\" to **\"who should we target, and what is the ROI?\"**, explicitly handling negative uplift / Sleeping Dogs risk.

You can adapt the notebooks to your own churn or CRM data by replacing the processed Telco dataset with your own engineered dataset and re-running the modeling pipeline.
