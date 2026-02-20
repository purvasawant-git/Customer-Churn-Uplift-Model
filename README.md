# Customer-Churn-Uplift-Model

Building a churn prediction and uplift model using Telco dataset.

## Project Structure

project-root/
├── data/                  # Raw and processed datasets
│   ├── raw/               # Original downloads
│   └── processed/         # Cleaned CSVs
├── notebooks/             # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_uplift.ipynb
├── src/                   # Clean Python scripts (for production)
│   ├── data_prep.py
│   ├── models.py
│   └── app.py             # For deployment
├── docs/                  # Images, write-ups, reports
├── requirements.txt       # List of dependencies
├── README.md              # Main documentation
└── .gitignore             # Ignore temp files, venv, etc.