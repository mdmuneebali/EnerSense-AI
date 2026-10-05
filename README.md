# ⚡ EnerSense AI — Intelligent Building Energy Optimizer

EnerSense AI is a machine-learning prototype for short-term building energy forecasting, unexpected high-consumption detection, and operational recommendations.

## What it does

- Analyzes appliance energy consumption over time
- Extracts hour, day-of-week, weekend, and month features
- Uses a chronological train/test split to respect time order
- Compares a naive time-series baseline with Linear Regression, Random Forest, and Gradient Boosting
- Adds 10-, 20-, and 30-minute historical energy lag features
- Selects **Linear Regression + 3 lag features** as the final model used by the dashboard
- Detects unexpected high consumption from large positive prediction residuals
- Produces NORMAL / HIGH / VERY HIGH alerts
- Generates operational recommendations
- Provides a Streamlit dashboard

## Key result from the current notebook

| Model | MAE (Wh) | RMSE (Wh) | R² |
|---|---:|---:|---:|
| Naive Previous Value | 26.65 | 66.44 | 0.467 |
| Linear Regression | 50.46 | 86.07 | 0.106 |
| Random Forest | 168.87 | 207.04 | -4.172 |
| Gradient Boosting | 134.49 | 173.80 | -2.644 |
| Linear Regression + 1 Lag | 28.41 | 61.93 | 0.537 |
| **Linear Regression + 3 Lags** | **28.43** | **60.78** | **0.554** |

The final model improves on the naive baseline on this chronological test split. The tree-based models are retained as benchmarks because they performed worse on this particular split.

## Project structure

```text
EnerSense-AI/
├── EnerSense_AI_Energy_Optimization_GitHub.ipynb
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── README.md
└── outputs/
    └── energy_results.csv   # generated locally; ignored by Git
```

## Run locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd EnerSense-AI
```

### 2. Create an environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download the UCI Appliances Energy Prediction dataset and place:

```text
data/energydata_complete.csv
```

See `data/README.md`.

### 5. Run the notebook

Open:

```text
EnerSense_AI_Energy_Optimization_GitHub.ipynb
```

Run the cells from top to bottom. The notebook creates:

```text
outputs/energy_results.csv
```

### 6. Launch the dashboard

```bash
streamlit run app.py
```

## Important modeling note

The anomaly logic is **residual-based**. It identifies consumption that is unexpectedly high compared with the model prediction. It does not prove that an appliance or electrical system is faulty.

## Future improvements

- Hyperparameter tuning with time-series cross-validation
- Compare stronger forecasting models such as XGBoost/LightGBM where appropriate
- Add explainability (SHAP)
- Add energy-cost and carbon-emission estimation
- Add configurable alert thresholds
- Deploy the dashboard publicly
