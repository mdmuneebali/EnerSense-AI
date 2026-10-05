import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="EnerSense AI",
    page_icon="⚡",
    layout="wide"
)

RESULTS_PATH = Path("outputs/energy_results.csv")

@st.cache_data
def load_results():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "outputs/energy_results.csv was not found. Run the notebook first "
            "to generate the dashboard output."
        )
    data = pd.read_csv(RESULTS_PATH)
    data["Date"] = pd.to_datetime(data["Date"])
    return data.sort_values("Date")

st.title("⚡ EnerSense AI")
st.caption("Intelligent Building Energy Optimizer")

try:
    results = load_results()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

# Sidebar filters
st.sidebar.header("Dashboard Filters")
alert_options = ["NORMAL", "HIGH", "VERY HIGH"]
selected_alerts = st.sidebar.multiselect(
    "Alert levels",
    alert_options,
    default=alert_options
)
show_anomalies_only = st.sidebar.checkbox("Show unexpected high-consumption events only")

filtered = results[results["Final_Alert"].isin(selected_alerts)].copy()
if show_anomalies_only:
    filtered = filtered[filtered["Anomaly"] == True]

if filtered.empty:
    st.warning("No observations match the selected filters.")
    st.stop()

# KPIs
avg_actual = filtered["Actual_Energy"].mean()
avg_predicted = filtered["Predicted_Energy"].mean()
mae = filtered["Absolute_Error"].mean()
high_alerts = (filtered["Final_Alert"] == "HIGH").sum()
very_high_alerts = (filtered["Final_Alert"] == "VERY HIGH").sum()
anomalies = filtered["Anomaly"].sum()
anomaly_rate = anomalies / len(filtered) * 100

c1, c2, c3 = st.columns(3)
c1.metric("Average Actual Energy", f"{avg_actual:.2f} Wh")
c2.metric("Average Predicted Energy", f"{avg_predicted:.2f} Wh")
c3.metric("MAE", f"{mae:.2f} Wh")

c4, c5, c6, c7 = st.columns(4)
c4.metric("High Alerts", int(high_alerts))
c5.metric("Very High Alerts", int(very_high_alerts))
c6.metric("Unexpected High Events", int(anomalies))
c7.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

st.divider()

# Forecast chart
st.subheader("Actual vs Predicted Energy")
chart = filtered.set_index("Date")[["Actual_Energy", "Predicted_Energy"]]
st.line_chart(chart)

# Alert distribution
left, right = st.columns(2)

with left:
    st.subheader("Alert Distribution")
    alert_counts = filtered["Final_Alert"].value_counts().reindex(alert_options, fill_value=0)
    st.bar_chart(alert_counts)

with right:
    st.subheader("Unexpected High-Consumption Events")
    anomaly_chart = filtered[filtered["Anomaly"]].set_index("Date")[["Actual_Energy"]]
    if anomaly_chart.empty:
        st.info("No unexpected high-consumption events in the current filter.")
    else:
        st.line_chart(anomaly_chart)

# Recommendations
st.subheader("Operational Recommendations")
recommendation_counts = (
    filtered["Final_Recommendation"]
    .value_counts()
    .rename_axis("Recommendation")
    .reset_index(name="Count")
)
st.dataframe(recommendation_counts, use_container_width=True, hide_index=True)

st.subheader("Recent Events")
display_cols = [
    "Date", "Actual_Energy", "Predicted_Energy",
    "Final_Alert", "Anomaly", "Final_Recommendation"
]
st.dataframe(
    filtered[display_cols].tail(25).sort_values("Date", ascending=False),
    use_container_width=True,
    hide_index=True
)

with st.expander("About the model"):
    st.write(
        "The selected forecasting model is Linear Regression using 10-, 20-, "
        "and 30-minute historical appliance-energy lags plus environmental and "
        "time-based features. Unexpected high-consumption events are detected "
        "from large positive prediction residuals."
    )
