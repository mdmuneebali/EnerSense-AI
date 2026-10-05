import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EnerSense AI",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

HIGH_THRESHOLD = 196
VERY_HIGH_THRESHOLD = 330


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "lights",

    "T1",
    "RH_1",

    "T2",
    "RH_2",

    "T3",
    "RH_3",

    "T4",
    "RH_4",

    "T5",
    "RH_5",

    "T6",
    "RH_6",

    "T7",
    "RH_7",

    "T8",
    "RH_8",

    "T9",
    "RH_9",

    "T_out",
    "Press_mm_hg",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",

    "hour",
    "day_of_week",
    "is_weekend",
    "month",

    "Appliances_lag_1",
    "Appliances_lag_2",
    "Appliances_lag_3",
]


# ============================================================
# REQUIRED DATASET COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "date",
    "Appliances",

    "lights",

    "T1",
    "RH_1",

    "T2",
    "RH_2",

    "T3",
    "RH_3",

    "T4",
    "RH_4",

    "T5",
    "RH_5",

    "T6",
    "RH_6",

    "T7",
    "RH_7",

    "T8",
    "RH_8",

    "T9",
    "RH_9",

    "T_out",
    "Press_mm_hg",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",

    "rv1",
    "rv2",
]


# ============================================================
# ALERT AND RECOMMENDATION LOGIC
# ============================================================

def energy_decision(predicted_energy, is_anomaly):

    if predicted_energy >= VERY_HIGH_THRESHOLD:

        alert = "VERY HIGH"

        recommendation = (
            "Check major electrical loads"
        )

    elif predicted_energy >= HIGH_THRESHOLD:

        alert = "HIGH"

        recommendation = (
            "Monitor building energy usage"
        )

    else:

        alert = "NORMAL"

        recommendation = (
            "Normal operation"
        )

    # Override recommendation when
    # an unexpected high-consumption event occurs.

    if is_anomaly:

        recommendation = (
            "Investigate unexpected energy increase"
        )

    return alert, recommendation


# ============================================================
# LOAD UCI DATASET
# ============================================================

@st.cache_data(
    show_spinner="Loading energy dataset..."
)
def load_dataset():

    try:

        from ucimlrepo import fetch_ucirepo

        # UCI Appliances Energy Prediction
        # Dataset ID = 374

        dataset = fetch_ucirepo(
            id=374
        )

        features = (
            dataset.data.features.copy()
        )

        targets = (
            dataset.data.targets.copy()
        )

        # Make sure target column is called Appliances

        if "Appliances" not in targets.columns:

            targets = targets.rename(
                columns={
                    targets.columns[0]:
                    "Appliances"
                }
            )

        # Combine features and target

        data = pd.concat(
            [
                features,
                targets[["Appliances"]]
            ],
            axis=1
        )

        return data

    except Exception as exc:

        raise RuntimeError(
            "The UCI dataset could not be loaded "
            "automatically. Please check the "
            "internet connection or download the "
            "UCI Appliances Energy Prediction "
            "dataset manually."
        ) from exc


# ============================================================
# BUILD MODEL RESULTS
# ============================================================

@st.cache_data(
    show_spinner="Running EnerSense AI model..."
)
def build_results(df):

    data = df.copy()

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # Convert date column
    # --------------------------------------------------------

    data["date"] = pd.to_datetime(
        data["date"]
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    data = (
        data
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Time-based feature engineering
    # --------------------------------------------------------

    data["hour"] = (
        data["date"].dt.hour
    )

    data["day_of_week"] = (
        data["date"].dt.dayofweek
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    )

    data["month"] = (
        data["date"].dt.month
    )

    # --------------------------------------------------------
    # Historical energy lag features
    # --------------------------------------------------------

    data["Appliances_lag_1"] = (
        data["Appliances"].shift(1)
    )

    data["Appliances_lag_2"] = (
        data["Appliances"].shift(2)
    )

    data["Appliances_lag_3"] = (
        data["Appliances"].shift(3)
    )

    # --------------------------------------------------------
    # Remove rows created by lag features
    # --------------------------------------------------------

    model_data = data.dropna(
        subset=[
            "Appliances_lag_1",
            "Appliances_lag_2",
            "Appliances_lag_3",
        ]
    ).copy()

    # --------------------------------------------------------
    # Prepare X and y
    # --------------------------------------------------------

    X = model_data[
        FEATURE_COLUMNS
    ]

    y = model_data[
        "Appliances"
    ]

    # --------------------------------------------------------
    # Chronological 80/20 train-test split
    # --------------------------------------------------------

    split_index = int(
        len(model_data) * 0.80
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    # --------------------------------------------------------
    # Train Linear Regression model
    # --------------------------------------------------------

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Model evaluation
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # Residual calculation
    # --------------------------------------------------------

    residuals = (
        y_test.to_numpy()
        - predictions
    )

    absolute_errors = np.abs(
        residuals
    )

    # --------------------------------------------------------
    # 95th percentile anomaly threshold
    # --------------------------------------------------------

    anomaly_threshold = np.percentile(
        absolute_errors,
        95
    )

    # Positive residual means actual energy
    # was higher than predicted energy.

    anomaly_flags = (
        residuals >= anomaly_threshold
    )

    # --------------------------------------------------------
    # Create results dataframe
    # --------------------------------------------------------

    results = model_data.iloc[
        split_index:
    ].copy()

    results["Actual_Energy"] = (
        y_test.to_numpy()
    )

    results["Predicted_Energy"] = (
        predictions
    )

    results["Prediction_Error"] = (
        residuals
    )

    results["Absolute_Error"] = (
        absolute_errors
    )

    results["Anomaly"] = (
        anomaly_flags
    )

    # --------------------------------------------------------
    # Generate alerts and recommendations
    # --------------------------------------------------------

    decisions = [
        energy_decision(
            prediction,
            anomaly
        )

        for prediction, anomaly
        in zip(
            predictions,
            anomaly_flags
        )
    ]

    results["Final_Alert"] = [
        decision[0]
        for decision in decisions
    ]

    results["Final_Recommendation"] = [
        decision[1]
        for decision in decisions
    ]

    # --------------------------------------------------------
    # Store model metrics
    # --------------------------------------------------------

    metrics = {

        "mae": mae,

        "rmse": rmse,

        "r2": r2,

        "anomaly_threshold":
            anomaly_threshold,

        "train_rows":
            len(X_train),

        "test_rows":
            len(X_test),
    }

    return results, metrics


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title(
    "⚡ EnerSense AI"
)

st.subheader(
    "Intelligent Building Energy Optimizer"
)

st.write(
    "Forecast appliance energy consumption, "
    "identify unexpected high-consumption events, "
    "and generate operational recommendations."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Dashboard Filters"
    )

    st.caption(
        "Model: Linear Regression with "
        "10-, 20-, and 30-minute energy lags."
    )

    st.caption(
        "Alert thresholds: "
        "196 Wh (HIGH) and "
        "330 Wh (VERY HIGH)."
    )


# ============================================================
# LOAD DATA AND RUN MODEL
# ============================================================

try:

    raw_data = load_dataset()

    results, metrics = (
        build_results(
            raw_data
        )
    )

except Exception as exc:

    st.error(
        str(exc)
    )

    st.stop()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

with st.sidebar:

    selected_alerts = st.multiselect(
        "Alert levels",

        [
            "NORMAL",
            "HIGH",
            "VERY HIGH"
        ],

        default=[
            "NORMAL",
            "HIGH",
            "VERY HIGH"
        ],
    )

    show_anomalies_only = (
        st.checkbox(
            "Show unexpected high-consumption events only"
        )
    )


# ============================================================
# APPLY FILTERS
# ============================================================

filtered = results[
    results["Final_Alert"].isin(
        selected_alerts
    )
].copy()


if show_anomalies_only:

    filtered = filtered[
        filtered["Anomaly"]
    ]


if filtered.empty:

    st.warning(
        "No observations match "
        "the selected filters."
    )

    st.stop()


# ============================================================
# KPI CALCULATIONS
# ============================================================

avg_actual = (
    filtered["Actual_Energy"].mean()
)

avg_predicted = (
    filtered["Predicted_Energy"].mean()
)

filtered_mae = (
    filtered["Absolute_Error"].mean()
)

high_alerts = int(
    (
        filtered["Final_Alert"]
        == "HIGH"
    ).sum()
)

very_high_alerts = int(
    (
        filtered["Final_Alert"]
        == "VERY HIGH"
    ).sum()
)

anomalies = int(
    filtered["Anomaly"].sum()
)

anomaly_rate = (
    anomalies
    / len(filtered)
    * 100
)


# ============================================================
# KPI DISPLAY
# ============================================================

c1, c2, c3 = st.columns(3)

c1.metric(
    "Average Actual Energy",
    f"{avg_actual:.2f} Wh"
)

c2.metric(
    "Average Predicted Energy",
    f"{avg_predicted:.2f} Wh"
)

c3.metric(
    "Filtered MAE",
    f"{filtered_mae:.2f} Wh"
)


c4, c5, c6, c7 = st.columns(4)

c4.metric(
    "High Alerts",
    high_alerts
)

c5.metric(
    "Very High Alerts",
    very_high_alerts
)

c6.metric(
    "Unexpected High Events",
    anomalies
)

c7.metric(
    "Anomaly Rate",
    f"{anomaly_rate:.2f}%"
)


st.divider()


# ============================================================
# ACTUAL VS PREDICTED ENERGY
# ============================================================

st.subheader(
    "Actual vs Predicted Energy"
)

chart = (
    filtered
    .set_index("date")
    [
        [
            "Actual_Energy",
            "Predicted_Energy"
        ]
    ]
)

st.line_chart(
    chart
)


# ============================================================
# ALERT DISTRIBUTION
# ============================================================

left, right = st.columns(2)


with left:

    st.subheader(
        "Energy Alert Distribution"
    )

    alert_counts = (
        filtered["Final_Alert"]
        .value_counts()
        .reindex(
            [
                "NORMAL",
                "HIGH",
                "VERY HIGH"
            ],
            fill_value=0
        )
    )

    st.bar_chart(
        alert_counts
    )


# ============================================================
# ANOMALY EVENTS
# ============================================================

with right:

    st.subheader(
        "Unexpected High-Consumption Events"
    )

    anomaly_data = (
        filtered[
            filtered["Anomaly"]
        ]
        .set_index("date")
    )

    if anomaly_data.empty:

        st.info(
            "No unexpected high-consumption "
            "events in the current filter."
        )

    else:

        st.line_chart(
            anomaly_data[
                ["Actual_Energy"]
            ]
        )


# ============================================================
# OPERATIONAL RECOMMENDATIONS
# ============================================================

st.subheader(
    "Operational Recommendations"
)

recommendation_counts = (
    filtered[
        "Final_Recommendation"
    ]
    .value_counts()
    .rename_axis(
        "Recommendation"
    )
    .reset_index(
        name="Count"
    )
)

st.dataframe(
    recommendation_counts,

    use_container_width=True,

    hide_index=True
)


# ============================================================
# RECENT EVENTS
# ============================================================

st.subheader(
    "Recent Events"
)

display_columns = [
    "date",
    "Actual_Energy",
    "Predicted_Energy",
    "Final_Alert",
    "Anomaly",
    "Final_Recommendation",
]

recent_events = (
    filtered[
        display_columns
    ]
    .tail(25)
    .sort_values(
        "date",
        ascending=False
    )
)

st.dataframe(
    recent_events,

    use_container_width=True,

    hide_index=True
)


# ============================================================
# MODEL DETAILS
# ============================================================

with st.expander(
    "Model details"
):

    st.write(
        f"Chronological split: "
        f"80% training / 20% test."
    )

    st.write(
        f"Training rows: "
        f"{metrics['train_rows']:,}"
    )

    st.write(
        f"Test rows: "
        f"{metrics['test_rows']:,}"
    )

    st.write(
        f"MAE: "
        f"{metrics['mae']:.2f} Wh"
    )

    st.write(
        f"RMSE: "
        f"{metrics['rmse']:.2f} Wh"
    )

    st.write(
        f"R²: "
        f"{metrics['r2']:.3f}"
    )

    st.write(
        f"95th-percentile absolute "
        f"residual threshold: "
        f"{metrics['anomaly_threshold']:.2f} Wh"
    )

    st.write(
        "An unexpected high-consumption "
        "event is flagged when the actual "
        "energy consumption is at least "
        "the anomaly threshold above the "
        "predicted consumption."
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Dataset: UCI Appliances Energy Prediction. "
    "Anomaly alerts are analytical indicators, "
    "not proof of equipment failure."
)
