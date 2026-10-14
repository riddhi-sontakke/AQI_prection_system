from __future__ import annotations

import datetime as dt
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in [_ROOT, _HERE]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pandas as pd

try:
    import streamlit as st
    import plotly.graph_objects as go
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Install: pip install streamlit plotly")

from data.city_data import available_cities
from models.artifact_loader import load_pipeline_for_city
from utils.aqi_advisor import AQIAdvisor, AQI_CATEGORIES


if STREAMLIT_AVAILABLE:
    st.set_page_config(
        page_title="AQI Prediction System",
        page_icon="AQI",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _aqi_color(aqi: float) -> str:
    if aqi > 400:
        return "#7E0023"
    if aqi > 300:
        return "#FF0000"
    if aqi > 200:
        return "#FF7E00"
    if aqi > 100:
        return "#FFFF00"
    if aqi > 50:
        return "#92D050"
    return "#00E400"


def _aqi_category_name(aqi: float) -> str:
    for cat in AQI_CATEGORIES:
        if cat["min"] <= aqi <= cat["max"]:
            return cat["name"]
    return "Severe / Hazardous"


def aqi_gauge(aqi_value: float, title: str = "Predicted AQI") -> "go.Figure":
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=aqi_value,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 500]},
                "bar": {"color": _aqi_color(aqi_value)},
                "steps": [
                    {"range": [0, 50], "color": "#00E400"},
                    {"range": [51, 100], "color": "#92D050"},
                    {"range": [101, 200], "color": "#FFFF00"},
                    {"range": [201, 300], "color": "#FF7E00"},
                    {"range": [301, 400], "color": "#FF0000"},
                    {"range": [401, 500], "color": "#7E0023"},
                ],
            },
        )
    )
    fig.update_layout(height=320, margin=dict(t=50, b=20, l=20, r=20))
    return fig


def forecast_chart(forecast: list, city: str) -> "go.Figure":
    times = [item["time"] for item in forecast]
    aqis = [item["aqi"] for item in forecast]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=aqis,
            mode="lines+markers+text",
            text=[f"{aqi:.0f}" for aqi in aqis],
            textposition="top center",
            marker=dict(size=10, color=[_aqi_color(aqi) for aqi in aqis]),
            line=dict(width=3, color="#2563eb"),
            name="Predicted AQI",
        )
    )
    fig.update_layout(
        title=f"Model Forecast: Next {len(forecast)} Hours ({city})",
        xaxis_title="Time",
        yaxis_title="Predicted AQI",
        yaxis=dict(range=[0, max(500, max(aqis, default=0) + 40)]),
        height=360,
        margin=dict(t=60, b=40, l=50, r=30),
        showlegend=False,
    )
    return fig


if STREAMLIT_AVAILABLE:

    @st.cache_resource(show_spinner="Loading trained model artifacts")
    def load_pipeline(city: str):
        return load_pipeline_for_city(city=city)


def run_dashboard():
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not installed. Run: pip install streamlit plotly")
        return

    st.title("AQI Prediction & Health Advisory System")
    # st.caption("Local CSV data only. No API data. No mock data.")

    with st.sidebar:
        city = st.selectbox("City", available_cities())
        forecast_steps = st.slider("Forecast horizon", 3, 12, 6, 1)
        if st.button("Run Prediction", type="primary"):
            st.cache_resource.clear()
            st.rerun()

    try:
        pipeline = load_pipeline(city)
        forecast_result = pipeline.predict_with_forecast(steps=forecast_steps)
    except Exception as exc:
        st.error(str(exc))
        st.code(f"python train.py --city {city} --no-lstm", language="powershell")
        st.stop()

    if forecast_result["source"] != "model":
        st.error(f"No trained model artifacts found for {city}. Train the city first.")
        st.code(f"python train.py --city {city} --no-lstm", language="powershell")
        st.stop()

    current_aqi = forecast_result["current_aqi"]
    confidence = forecast_result["confidence"]
    current_data = forecast_result["current_data"]
    forecast = forecast_result["forecast"]

    advisor = AQIAdvisor()
    now = dt.datetime.now()
    advice = advisor.generate_advice(
        aqi=current_aqi,
        confidence=confidence / 100,
        city=city,
        hour=now.hour,
        month=now.month,
    )

    st.success(f"Model prediction complete for {city} using {current_data.get('source')}")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Predicted AQI", f"{current_aqi:.0f}")
    metric_cols[1].metric("Category", advice.aqi_category)
    metric_cols[2].metric("Confidence", f"{confidence:.1f}%")
    metric_cols[3].metric("Source Row", current_data.get("timestamp", "latest"))

    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(aqi_gauge(current_aqi, f"{city} Predicted AQI"), width="stretch")
    with right:
        st.subheader("Latest Local Row Used")
        pollutant_cols = [
            "PM2.5",
            "PM10",
            "NO2",
            "SO2",
            "CO",
            "O3",
            "Temperature",
            "Humidity",
            "WindSpeed",
            "Pressure_MSL_hPa",
        ]
        rows = [
            {"Feature": col, "Value": current_data.get(col)}
            for col in pollutant_cols
            if col in current_data
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if forecast:
        st.subheader("Model Forecast")
        st.plotly_chart(forecast_chart(forecast, city), width="stretch")
        forecast_df = pd.DataFrame(forecast)
        forecast_df["category"] = forecast_df["aqi"].apply(_aqi_category_name)
        st.dataframe(forecast_df, width="stretch", hide_index=True)

    st.subheader("Health Advisory")
    for item in advice.general_advice:
        st.write(f"- {item}")
    if advice.mask_required:
        st.warning(advice.mask_type)
    if advice.city_alert:
        st.warning(advice.city_alert)

    st.caption("Model: trained city artifacts in models/<city>. Data: bundled local CSV files.")


if __name__ == "__main__":
    run_dashboard()
