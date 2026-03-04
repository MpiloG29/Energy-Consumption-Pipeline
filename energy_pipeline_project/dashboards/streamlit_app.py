"""
================================================================================
SOUTH AFRICA ENERGY CONSUMPTION ANALYTICS DASHBOARD (STREAMLIT)
Advanced ML-Powered Insights for Load Shedding, Consumption Patterns & Forecasting
================================================================================
"""

import os
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================================
# THEME
# ============================================================================
THEME = {
    "bg_dark": "#0d1117",
    "bg_card": "#161b22",
    "bg_secondary": "#21262d",
    "border": "#30363d",
    "text_primary": "#f0f6fc",
    "text_secondary": "#8b949e",
    "text_muted": "#6e7681",
    "accent_blue": "#58a6ff",
    "accent_green": "#3fb950",
    "accent_red": "#f85149",
    "accent_orange": "#d29922",
    "accent_purple": "#a371f7",
}

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "raw_data", "energy_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "ml_models")

sys.path.insert(0, BASE_DIR)
from ml_models.feature_engineering import engineer_features


# ============================================================================
# STREAMLIT CONFIG
# ============================================================================
st.set_page_config(
    page_title="SA Energy Analytics | Streamlit",
    layout="wide",
)

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {THEME['bg_dark']};
            color: {THEME['text_primary']};
        }}
        .block-container {{
            padding-top: 1.5rem;
        }}
        .kpi-card {{
            background: {THEME['bg_card']};
            padding: 18px 20px;
            border-radius: 12px;
            border: 1px solid {THEME['border']};
        }}
        .kpi-label {{
            color: {THEME['text_secondary']};
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .kpi-value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: {THEME['text_primary']};
            margin: 6px 0 2px 0;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["date"] = df["timestamp"].dt.date
    df["day_name"] = df["timestamp"].dt.day_name()
    df["month"] = df["timestamp"].dt.month
    df["month_name"] = df["timestamp"].dt.month_name()
    df["week"] = df["timestamp"].dt.isocalendar().week
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    def get_season(month):
        if month in [6, 7, 8]:
            return "Winter"
        if month in [12, 1, 2]:
            return "Summer"
        if month in [3, 4, 5]:
            return "Autumn"
        return "Spring"

    df["season"] = df["month"].apply(get_season)

    def get_time_period(hour):
        if 6 <= hour < 9:
            return "Morning Peak"
        if 9 <= hour < 17:
            return "Daytime"
        if 17 <= hour < 21:
            return "Evening Peak"
        return "Off-Peak"

    df["time_period"] = df["hour"].apply(get_time_period)

    df["outage_severity"] = pd.cut(
        df["load_shedding_stage"],
        bins=[-1, 0, 2, 4, 6],
        labels=["No Outage", "Low (1-2)", "Medium (3-4)", "High (5-6)"],
    )

    return df


@st.cache_resource(show_spinner=False)
def load_models():
    gb_model = rf_model = lr_model = None
    try:
        gb_model = joblib.load(os.path.join(MODEL_DIR, "energy_model_gradient_boosting.pkl"))
        rf_model = joblib.load(os.path.join(MODEL_DIR, "energy_model_random_forest.pkl"))
        lr_model = joblib.load(os.path.join(MODEL_DIR, "energy_model_linear.pkl"))
    except Exception:
        pass
    return gb_model, rf_model, lr_model


@st.cache_data(show_spinner=False)
def build_features(df):
    features_df, target, df_eng = engineer_features(df.copy())
    return features_df, target, df_eng


# ============================================================================
# HEADER
# ============================================================================
st.title("South Africa Energy Analytics")
st.caption("ML-Powered Load Shedding & Consumption Intelligence")
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ============================================================================
# LOAD DATA
# ============================================================================
df = load_data()
features_df, target, df_eng = build_features(df)

gb_model, rf_model, lr_model = load_models()
if gb_model is not None:
    df_eng["predicted"] = gb_model.predict(features_df)
    df_eng["residual"] = df_eng["energy_consumption"] - df_eng["predicted"]
    df_eng["abs_error"] = np.abs(df_eng["residual"])
    df_eng["pct_error"] = (df_eng["abs_error"] / df_eng["energy_consumption"].clip(lower=0.01)) * 100

# ============================================================================
# METRICS
# ============================================================================
total_consumption = df["energy_consumption"].sum()
peak_consumption = df["energy_consumption"].max()
avg_consumption = df["energy_consumption"].mean()

outage_hours = (df["load_shedding_stage"] > 0).sum()
outage_pct = (outage_hours / len(df)) * 100

if gb_model is not None:
    rmse = np.sqrt(np.mean(df_eng["residual"] ** 2))
    mae = np.mean(df_eng["abs_error"])
    r2 = 1 - (
        np.sum(df_eng["residual"] ** 2)
        / np.sum((df_eng["energy_consumption"] - df_eng["energy_consumption"].mean()) ** 2)
    )
else:
    rmse = mae = r2 = 0.0

TARIFF_PEAK = 2.85
TARIFF_OFFPEAK = 1.45

df["estimated_cost"] = np.where(
    df["time_period"].isin(["Morning Peak", "Evening Peak"]),
    df["energy_consumption"] * TARIFF_PEAK,
    df["energy_consumption"] * TARIFF_OFFPEAK,
)

kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Consumption</div>
            <div class="kpi-value">{total_consumption:,.0f}</div>
            <div style="color:{THEME['text_muted']};font-size:0.8rem;">kWh (12 months)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[1]:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Model R2 Score</div>
            <div class="kpi-value" style="color:{THEME['accent_green']};">{r2:.4f}</div>
            <div style="color:{THEME['text_muted']};font-size:0.8rem;">RMSE: {rmse:.3f} kWh</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[2]:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Load Shedding Hours</div>
            <div class="kpi-value" style="color:{THEME['accent_orange']};">{outage_hours:,}</div>
            <div style="color:{THEME['text_muted']};font-size:0.8rem;">{outage_pct:.1f}% of time affected</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[3]:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Estimated Cost</div>
            <div class="kpi-value" style="color:{THEME['accent_purple']};">R{df['estimated_cost'].sum():,.0f}</div>
            <div style="color:{THEME['text_muted']};font-size:0.8rem;">Based on Eskom TOU tariffs</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Model accuracy cards (focused, no performance tab)
if gb_model is not None:
    mape = np.mean(df_eng["pct_error"])
else:
    mape = 0.0

st.markdown("### Model Accuracy")
acc_cols = st.columns(4)

with acc_cols[0]:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">R2 Score</div>
            <div class="kpi-value" style="color:{THEME['accent_green']};">{r2:.4f}</div>
            <div style="color:{THEME['text_muted']};font-size:0.8rem;">Higher is better</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with acc_cols[1]:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">RMSE</div>
            <div class="kpi-value" style="color:{THEME['accent_blue']};">{rmse:.3f}</div>
            <div style="color:{THEME['text_muted']};font-size:0.8rem;">kWh</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with acc_cols[2]:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">MAE</div>
            <div class="kpi-value" style="color:{THEME['accent_orange']};">{mae:.3f}</div>
            <div style="color:{THEME['text_muted']};font-size:0.8rem;">kWh</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with acc_cols[3]:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">MAPE</div>
            <div class="kpi-value" style="color:{THEME['accent_purple']};">{mape:.2f}%</div>
            <div style="color:{THEME['text_muted']};font-size:0.8rem;">Avg % error</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================================
# TABS
# ============================================================================
tab_bi, tab_patterns, tab_loadshed, tab_cost, tab_predict = st.tabs(
    [
        "BI Overview",
        "Consumption Patterns",
        "Load Shedding Impact",
        "Cost Analysis",
        "Predictive Insights",
    ]
)

# ============================================================================
# TAB: OVERVIEW
# ============================================================================
with tab_bi:
    st.subheader("BI-Style Overview")

    min_date = pd.to_datetime(df["date"]).min().date()
    max_date = pd.to_datetime(df["date"]).max().date()

    filter_cols = st.columns(3)
    with filter_cols[0]:
        date_range = st.date_input("Date Range", (min_date, max_date))
    with filter_cols[1]:
        day_filter = st.multiselect(
            "Day of Week",
            options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        )
    with filter_cols[2]:
        hour_range = st.slider("Hour Range", min_value=0, max_value=23, value=(0, 23))

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    filtered_df = df[
        (pd.to_datetime(df["date"]) >= pd.to_datetime(start_date))
        & (pd.to_datetime(df["date"]) <= pd.to_datetime(end_date))
        & (df["day_name"].isin(day_filter))
        & (df["hour"] >= hour_range[0])
        & (df["hour"] <= hour_range[1])
    ].copy()

    if filtered_df.empty:
        st.warning("No data for the selected filters. Showing full dataset.")
        filtered_df = df.copy()

    daily_summary = (
        filtered_df.groupby("date")["energy_consumption"]
        .agg(total_kwh="sum", avg_kwh="mean", max_kwh="max", min_kwh="min")
        .reset_index()
    )

    daily_summary["date"] = pd.to_datetime(daily_summary["date"])
    daily_summary["day_name"] = daily_summary["date"].dt.day_name()

    overview_cols = st.columns(4)
    with overview_cols[0]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Consumption</div>
                <div class="kpi-value">{daily_summary['total_kwh'].sum():,.0f}</div>
                <div style="color:{THEME['text_muted']};font-size:0.8rem;">kWh</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with overview_cols[1]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Average Daily</div>
                <div class="kpi-value">{daily_summary['total_kwh'].mean():,.2f}</div>
                <div style="color:{THEME['text_muted']};font-size:0.8rem;">kWh</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with overview_cols[2]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Peak Consumption</div>
                <div class="kpi-value">{daily_summary['max_kwh'].max():,.2f}</div>
                <div style="color:{THEME['text_muted']};font-size:0.8rem;">kWh</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with overview_cols[3]:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Minimum Consumption</div>
                <div class="kpi-value">{daily_summary['min_kwh'].min():,.2f}</div>
                <div style="color:{THEME['text_muted']};font-size:0.8rem;">kWh</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    fig_daily = go.Figure()
    fig_daily.add_trace(
        go.Scatter(
            x=daily_summary["date"],
            y=daily_summary["total_kwh"],
            mode="lines",
            name="Daily Total",
            line=dict(color=THEME["accent_blue"], width=2),
        )
    )
    fig_daily.update_layout(
        title="Daily Consumption Trend",
        template="plotly_dark",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Total kWh"),
        margin=dict(l=40, r=20, t=50, b=30),
    )

    hourly_avg = filtered_df.groupby("hour")["energy_consumption"].mean().reset_index()
    fig_hourly = go.Figure()
    fig_hourly.add_trace(
        go.Bar(
            x=hourly_avg["hour"],
            y=hourly_avg["energy_consumption"],
            marker_color=THEME["accent_orange"],
        )
    )
    fig_hourly.update_layout(
        title="Average Consumption by Hour",
        template="plotly_dark",
        xaxis=dict(title="Hour", dtick=1),
        yaxis=dict(title="Avg kWh"),
        margin=dict(l=40, r=20, t=50, b=30),
    )

    scatter_data = (
        filtered_df.groupby(["day_name", "hour"])["energy_consumption"]
        .agg(avg_kwh="mean", total_kwh="sum")
        .reset_index()
    )
    fig_scatter = px.scatter(
        scatter_data,
        x="hour",
        y="avg_kwh",
        size="total_kwh",
        color="day_name",
        title="Consumption Distribution",
        template="plotly_dark",
        labels={"avg_kwh": "Avg kWh", "hour": "Hour"},
    )
    fig_scatter.update_layout(margin=dict(l=40, r=20, t=50, b=30))

    day_totals = filtered_df.groupby("day_name")["energy_consumption"].sum().reset_index()
    fig_donut = go.Figure(
        data=[
            go.Pie(
                labels=day_totals["day_name"],
                values=day_totals["energy_consumption"],
                hole=0.55,
                marker=dict(colors=px.colors.qualitative.Set2),
            )
        ]
    )
    fig_donut.update_layout(
        title="Consumption by Day of Week",
        template="plotly_dark",
        margin=dict(l=40, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    heatmap_table = filtered_df.pivot_table(
        values="energy_consumption", index="day_name", columns="hour", aggfunc="mean"
    )
    heatmap_table = heatmap_table.reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )

    fig_table_heatmap = go.Figure(
        data=go.Heatmap(
            z=heatmap_table.values,
            x=heatmap_table.columns,
            y=heatmap_table.index,
            colorscale="Viridis",
            colorbar=dict(title="Avg kWh"),
        )
    )
    fig_table_heatmap.update_layout(
        title="Consumption Heatmap: Day vs Hour",
        template="plotly_dark",
        margin=dict(l=80, r=20, t=50, b=30),
        xaxis=dict(title="Hour"),
    )

    forecast_series = daily_summary[["date", "total_kwh"]].copy()
    if len(forecast_series) >= 7:
        baseline = forecast_series["total_kwh"].tail(7).mean()
        last_date = forecast_series["date"].max()
        forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7, freq="D")
        forecast_values = [baseline] * len(forecast_dates)
        fig_forecast = go.Figure()
        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_series["date"],
                y=forecast_series["total_kwh"],
                mode="lines",
                name="Actual",
                line=dict(color=THEME["accent_blue"], width=2),
            )
        )
        fig_forecast.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                mode="lines",
                name="Forecast (7d baseline)",
                line=dict(color=THEME["accent_orange"], dash="dash", width=2),
            )
        )
        fig_forecast.update_layout(
            title="Daily Consumption Forecast (7 Days)",
            template="plotly_dark",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Total kWh"),
            margin=dict(l=40, r=20, t=50, b=30),
        )
    else:
        fig_forecast = None

    st.plotly_chart(fig_daily, width="stretch")
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(fig_hourly, width="stretch")
    with chart_cols[1]:
        st.plotly_chart(fig_scatter, width="stretch")

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(fig_table_heatmap, width="stretch")
    with chart_cols[1]:
        st.plotly_chart(fig_donut, width="stretch")

    if fig_forecast is not None:
        st.plotly_chart(fig_forecast, width="stretch")
    else:
        st.info("Not enough data to generate a 7-day forecast.")

    st.markdown("---")

with tab_patterns:
    daily = (
        df.groupby("date")
        .agg({"energy_consumption": "sum", "load_shedding_stage": "max"})
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily["rolling_7d"] = daily["energy_consumption"].rolling(7).mean()
    daily["rolling_30d"] = daily["energy_consumption"].rolling(30).mean()

    fig_ts = go.Figure()
    fig_ts.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["energy_consumption"],
            mode="lines",
            name="Daily Total",
            line=dict(color=THEME["text_muted"], width=1),
            opacity=0.5,
        )
    )
    fig_ts.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["rolling_7d"],
            mode="lines",
            name="7-Day MA",
            line=dict(color=THEME["accent_blue"], width=2),
        )
    )
    fig_ts.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["rolling_30d"],
            mode="lines",
            name="30-Day MA",
            line=dict(color=THEME["accent_purple"], width=2),
        )
    )

    outage_days = daily[daily["load_shedding_stage"] > 0]
    fig_ts.add_trace(
        go.Scatter(
            x=outage_days["date"],
            y=outage_days["energy_consumption"],
            mode="markers",
            name="Outage Days",
            marker=dict(color=THEME["accent_red"], size=4, symbol="x"),
        )
    )

    fig_ts.update_layout(
        title="Consumption Trend with Load Shedding Overlay",
        template="plotly_dark",
        xaxis=dict(showgrid=True),
        yaxis=dict(title="kWh", showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=50, b=30),
        hovermode="x unified",
    )

    heatmap_data = df.pivot_table(
        values="energy_consumption", index="day_name", columns="hour", aggfunc="mean"
    )
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = heatmap_data.reindex(day_order)

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale=[[0, THEME["bg_dark"]], [0.3, THEME["accent_blue"]], [0.7, THEME["accent_orange"]], [1, THEME["accent_red"]]],
            colorbar=dict(title="kWh"),
            hovertemplate="Hour: %{x}<br>Day: %{y}<br>Avg: %{z:.2f} kWh<extra></extra>",
        )
    )

    for day in day_order:
        max_hour = heatmap_data.loc[day].idxmax()
        fig_heatmap.add_annotation(x=max_hour, y=day, text="●", showarrow=False, font=dict(color="white", size=8))

    fig_heatmap.update_layout(
        title="Consumption Heatmap: Hour x Day (dot = Daily Peak)",
        template="plotly_dark",
        xaxis=dict(title="Hour of Day", dtick=1),
        margin=dict(l=80, r=20, t=50, b=30),
    )

    fig_seasonal = go.Figure()
    season_colors = {
        "Winter": THEME["accent_blue"],
        "Summer": THEME["accent_orange"],
        "Spring": THEME["accent_green"],
        "Autumn": THEME["accent_red"],
    }

    for season in ["Summer", "Autumn", "Winter", "Spring"]:
        season_data = df[df["season"] == season]["energy_consumption"]
        fig_seasonal.add_trace(
            go.Box(
                y=season_data,
                name=season,
                marker_color=season_colors[season],
                boxmean="sd",
                hovertemplate=f"{season}<br>Value: %{{y:.2f}} kWh<extra></extra>",
            )
        )

    fig_seasonal.update_layout(
        title="Consumption Distribution by Season",
        template="plotly_dark",
        yaxis=dict(title="kWh"),
        showlegend=False,
        margin=dict(l=40, r=20, t=50, b=30),
    )

    weekend_comp = df.groupby(["hour", "is_weekend"])["energy_consumption"].mean().unstack()

    fig_weekend = go.Figure()
    fig_weekend.add_trace(
        go.Scatter(
            x=weekend_comp.index,
            y=weekend_comp[0],
            mode="lines+markers",
            name="Weekday",
            line=dict(color=THEME["accent_blue"], width=2),
            marker=dict(size=6),
        )
    )
    fig_weekend.add_trace(
        go.Scatter(
            x=weekend_comp.index,
            y=weekend_comp[1],
            mode="lines+markers",
            name="Weekend",
            line=dict(color=THEME["accent_orange"], width=2),
            marker=dict(size=6),
        )
    )

    fig_weekend.add_vrect(x0=6, x1=9, fillcolor=THEME["accent_red"], opacity=0.1, line_width=0, annotation_text="AM Peak")
    fig_weekend.add_vrect(x0=17, x1=21, fillcolor=THEME["accent_red"], opacity=0.1, line_width=0, annotation_text="PM Peak")

    fig_weekend.update_layout(
        title="Weekday vs Weekend Load Profile",
        template="plotly_dark",
        xaxis=dict(title="Hour of Day", dtick=2),
        yaxis=dict(title="Avg Consumption (kWh)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=50, b=30),
    )

    st.plotly_chart(fig_ts, width="stretch")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(fig_heatmap, width="stretch")
    with col_b:
        st.plotly_chart(fig_seasonal, width="stretch")
    st.plotly_chart(fig_weekend, width="stretch")

# ============================================================================
# TAB: LOAD SHEDDING IMPACT
# ============================================================================
with tab_loadshed:
    stage_impact = df.groupby("load_shedding_stage")["energy_consumption"].agg(["mean", "std", "count"]).reset_index()
    stage_impact.columns = ["stage", "mean", "std", "count"]

    fig_stages = go.Figure()
    fig_stages.add_trace(
        go.Bar(
            x=[f"Stage {int(s)}" if s > 0 else "No Outage" for s in stage_impact["stage"]],
            y=stage_impact["mean"],
            error_y=dict(type="data", array=stage_impact["std"], visible=True),
            marker=dict(
                color=[
                    THEME["accent_green"] if s == 0 else THEME["accent_orange"] if s <= 2 else THEME["accent_red"]
                    for s in stage_impact["stage"]
                ]
            ),
            text=[f"n={c:,}" for c in stage_impact["count"]],
            textposition="outside",
        )
    )

    fig_stages.update_layout(
        title="Consumption by Load Shedding Stage (Mean +- Std Dev)",
        template="plotly_dark",
        yaxis=dict(title="Avg Consumption (kWh)"),
        margin=dict(l=40, r=20, t=50, b=30),
        showlegend=False,
    )

    backup_data = df.groupby(["load_shedding_stage", "backup_power"])["energy_consumption"].mean().unstack()

    fig_backup = go.Figure()
    fig_backup.add_trace(
        go.Bar(
            name="No Backup",
            x=[f"Stage {int(s)}" if s > 0 else "Normal" for s in backup_data.index],
            y=backup_data[False] if False in backup_data.columns else [0] * len(backup_data),
            marker_color=THEME["accent_red"],
        )
    )
    fig_backup.add_trace(
        go.Bar(
            name="With Backup",
            x=[f"Stage {int(s)}" if s > 0 else "Normal" for s in backup_data.index],
            y=backup_data[True] if True in backup_data.columns else [0] * len(backup_data),
            marker_color=THEME["accent_green"],
        )
    )

    fig_backup.update_layout(
        title="Backup Power Effect: Consumption During Outages",
        template="plotly_dark",
        yaxis=dict(title="Avg Consumption (kWh)"),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=50, b=30),
    )

    df_sorted = df.sort_values(["household_id", "timestamp"]).copy()
    df_sorted["prev_outage"] = df_sorted.groupby("household_id")["load_shedding_stage"].shift(1).fillna(0)
    df_sorted["next_outage"] = df_sorted.groupby("household_id")["load_shedding_stage"].shift(-1).fillna(0)

    pre_outage = df_sorted[(df_sorted["load_shedding_stage"] == 0) & (df_sorted["next_outage"] > 0)]["energy_consumption"].mean()
    during_outage = df_sorted[df_sorted["load_shedding_stage"] > 0]["energy_consumption"].mean()
    post_outage = df_sorted[(df_sorted["load_shedding_stage"] == 0) & (df_sorted["prev_outage"] > 0)]["energy_consumption"].mean()
    normal = df_sorted[
        (df_sorted["load_shedding_stage"] == 0)
        & (df_sorted["prev_outage"] == 0)
        & (df_sorted["next_outage"] == 0)
    ]["energy_consumption"].mean()

    fig_transition = go.Figure()
    categories = ["Normal", "Pre-Outage", "During Outage", "Post-Outage"]
    values = [normal, pre_outage, during_outage, post_outage]

    fig_transition.add_trace(
        go.Waterfall(
            x=categories,
            y=[normal, pre_outage - normal, during_outage - pre_outage, post_outage - during_outage],
            measure=["absolute", "relative", "relative", "relative"],
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            connector=dict(line=dict(color=THEME["text_muted"])),
        )
    )

    fig_transition.update_layout(
        title="Load Shedding Transition Impact (Waterfall)",
        template="plotly_dark",
        yaxis=dict(title="Consumption (kWh)"),
        margin=dict(l=40, r=20, t=50, b=30),
        showlegend=False,
    )

    outage_by_hour = df[df["load_shedding_stage"] > 0].groupby("hour").size()

    fig_outage_dist = make_subplots(specs=[[{"secondary_y": True}]])

    fig_outage_dist.add_trace(
        go.Bar(
            x=outage_by_hour.index,
            y=outage_by_hour.values,
            name="Outage Hours",
            marker_color=THEME["accent_red"],
            opacity=0.7,
        ),
        secondary_y=False,
    )

    fig_outage_dist.add_trace(
        go.Scatter(
            x=df.groupby("hour")["energy_consumption"].mean().index,
            y=df.groupby("hour")["energy_consumption"].mean().values,
            name="Avg Consumption",
            line=dict(color=THEME["accent_blue"], width=3),
            mode="lines+markers",
        ),
        secondary_y=True,
    )

    fig_outage_dist.update_layout(
        title="Outage Frequency vs Consumption by Hour",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=30),
    )
    fig_outage_dist.update_yaxes(title_text="Outage Count", secondary_y=False)
    fig_outage_dist.update_yaxes(title_text="Avg kWh", secondary_y=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(fig_stages, width="stretch")
    with col_b:
        st.plotly_chart(fig_backup, width="stretch")

    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(fig_transition, width="stretch")
    with col_d:
        st.plotly_chart(fig_outage_dist, width="stretch")

# ============================================================================
# TAB: COST ANALYSIS
# ============================================================================
with tab_cost:
    monthly_cost = df.groupby(["month_name", "time_period"])["estimated_cost"].sum().unstack()
    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    monthly_cost = monthly_cost.reindex([m for m in month_order if m in monthly_cost.index])

    fig_monthly = go.Figure()
    period_colors = {
        "Morning Peak": THEME["accent_orange"],
        "Daytime": THEME["accent_blue"],
        "Evening Peak": THEME["accent_red"],
        "Off-Peak": THEME["accent_green"],
    }

    for period in ["Off-Peak", "Daytime", "Morning Peak", "Evening Peak"]:
        if period in monthly_cost.columns:
            fig_monthly.add_trace(
                go.Bar(
                    name=period,
                    x=monthly_cost.index,
                    y=monthly_cost[period],
                    marker_color=period_colors.get(period, THEME["text_muted"]),
                )
            )

    fig_monthly.update_layout(
        title="Monthly Cost by Time-of-Use Period",
        template="plotly_dark",
        yaxis=dict(title="Cost (R)"),
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=50, b=30),
    )

    household_cost = (
        df.groupby("household_id")
        .agg({"estimated_cost": "sum", "energy_consumption": "sum", "backup_power": "first"})
        .reset_index()
    )
    household_cost["cost_per_kwh"] = household_cost["estimated_cost"] / household_cost["energy_consumption"]

    fig_household_cost = go.Figure()
    colors = [THEME["accent_green"] if bp else THEME["accent_red"] for bp in household_cost["backup_power"]]

    fig_household_cost.add_trace(
        go.Bar(
            x=household_cost["household_id"],
            y=household_cost["estimated_cost"],
            marker_color=colors,
            text=[f"R{c:,.0f}" for c in household_cost["estimated_cost"]],
            textposition="outside",
            hovertemplate="%{x}<br>Total: R%{y:,.0f}<br>R/kWh: %{customdata:.2f}<extra></extra>",
            customdata=household_cost["cost_per_kwh"],
        )
    )

    fig_household_cost.update_layout(
        title="Annual Cost by Household (Green = Has Backup)",
        template="plotly_dark",
        yaxis=dict(title="Total Cost (R)"),
        margin=dict(l=40, r=20, t=50, b=30),
    )

    peak_consumption = df[df["time_period"].isin(["Morning Peak", "Evening Peak"])]["energy_consumption"].sum()
    offpeak_consumption = df[~df["time_period"].isin(["Morning Peak", "Evening Peak"])]["energy_consumption"].sum()
    current_cost = peak_consumption * TARIFF_PEAK + offpeak_consumption * TARIFF_OFFPEAK

    shift_pct = 0.20
    shifted_peak = peak_consumption * (1 - shift_pct)
    shifted_offpeak = offpeak_consumption + (peak_consumption * shift_pct)
    potential_cost = shifted_peak * TARIFF_PEAK + shifted_offpeak * TARIFF_OFFPEAK
    savings = current_cost - potential_cost

    fig_savings = go.Figure()
    fig_savings.add_trace(
        go.Indicator(
            mode="number+delta",
            value=savings,
            number=dict(prefix="R ", valueformat=",.0f"),
            delta=dict(reference=0, relative=False, valueformat=".0f"),
            title=dict(text="Potential Annual Savings (20% peak shifted)"),
        )
    )

    fig_savings.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=80, b=20))

    daily_data = (
        df.groupby("date")
        .agg({"estimated_cost": "sum", "load_shedding_stage": lambda x: (x > 0).sum(), "energy_consumption": "sum"})
        .reset_index()
    )
    daily_data.columns = ["date", "cost", "outage_hours", "consumption"]

    fig_cost_outage = go.Figure()
    fig_cost_outage.add_trace(
        go.Scatter(
            x=daily_data["outage_hours"],
            y=daily_data["cost"],
            mode="markers",
            marker=dict(
                color=daily_data["consumption"],
                colorscale=[[0, THEME["accent_blue"]], [1, THEME["accent_red"]]],
                size=8,
                opacity=0.6,
                colorbar=dict(title="kWh"),
            ),
            hovertemplate="Outage Hours: %{x}<br>Daily Cost: R%{y:.0f}<extra></extra>",
        )
    )

    if len(daily_data) > 10:
        z = np.polyfit(daily_data["outage_hours"], daily_data["cost"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(daily_data["outage_hours"].min(), daily_data["outage_hours"].max(), 100)
        fig_cost_outage.add_trace(
            go.Scatter(
                x=x_line,
                y=p(x_line),
                mode="lines",
                name="Trend",
                line=dict(color=THEME["accent_orange"], dash="dash", width=2),
            )
        )

    fig_cost_outage.update_layout(
        title="Daily Cost vs Outage Hours",
        template="plotly_dark",
        xaxis=dict(title="Outage Hours per Day"),
        yaxis=dict(title="Daily Cost (R)"),
        showlegend=False,
        margin=dict(l=40, r=20, t=50, b=30),
    )

    st.plotly_chart(fig_monthly, width="stretch")
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(fig_household_cost, width="stretch")
    with col_b:
        st.plotly_chart(fig_savings, width="stretch")
    st.plotly_chart(fig_cost_outage, width="stretch")

# ============================================================================
# TAB: PREDICTIVE INSIGHTS
# ============================================================================
with tab_predict:
    if gb_model is None:
        st.warning("Models not loaded. Run train_model.py first.")
    else:
        daily_pred = df_eng.groupby("date").agg({"energy_consumption": "sum", "predicted": "sum", "residual": ["mean", "std"]}).reset_index()
        daily_pred.columns = ["date", "actual", "predicted", "residual_mean", "residual_std"]
        daily_pred["date"] = pd.to_datetime(daily_pred["date"])
        daily_pred["upper"] = daily_pred["predicted"] + 1.96 * daily_pred["residual_std"] * np.sqrt(24)
        daily_pred["lower"] = daily_pred["predicted"] - 1.96 * daily_pred["residual_std"] * np.sqrt(24)

        fig_confidence = go.Figure()
        fig_confidence.add_trace(
            go.Scatter(
                x=pd.concat([daily_pred["date"], daily_pred["date"][::-1]]),
                y=pd.concat([daily_pred["upper"], daily_pred["lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(88, 166, 255, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% CI",
                hoverinfo="skip",
            )
        )

        fig_confidence.add_trace(
            go.Scatter(
                x=daily_pred["date"],
                y=daily_pred["actual"],
                mode="lines",
                name="Actual",
                line=dict(color=THEME["text_secondary"], width=1),
                opacity=0.7,
            )
        )

        fig_confidence.add_trace(
            go.Scatter(
                x=daily_pred["date"],
                y=daily_pred["predicted"],
                mode="lines",
                name="Predicted",
                line=dict(color=THEME["accent_blue"], width=2),
            )
        )

        fig_confidence.update_layout(
            title="Daily Consumption: Actual vs Predicted with 95% Confidence",
            template="plotly_dark",
            yaxis=dict(title="Daily Consumption (kWh)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=50, b=30),
            hovermode="x unified",
        )

        df_eng["z_score"] = (df_eng["energy_consumption"] - df_eng["energy_consumption"].mean()) / df_eng["energy_consumption"].std()
        df_eng["is_anomaly"] = np.abs(df_eng["z_score"]) > 2.5
        anomalies = df_eng[df_eng["is_anomaly"]].copy()

        fig_anomaly = go.Figure()
        normal = df_eng[~df_eng["is_anomaly"]].sample(min(2000, len(df_eng[~df_eng["is_anomaly"]])), random_state=42)
        fig_anomaly.add_trace(
            go.Scattergl(
                x=normal["hour"],
                y=normal["energy_consumption"],
                mode="markers",
                name="Normal",
                marker=dict(color=THEME["text_muted"], size=3, opacity=0.3),
            )
        )

        if len(anomalies) > 0:
            fig_anomaly.add_trace(
                go.Scatter(
                    x=anomalies["hour"],
                    y=anomalies["energy_consumption"],
                    mode="markers",
                    name=f"Anomalies ({len(anomalies)})",
                    marker=dict(color=THEME["accent_red"]),
                    hovertemplate="Hour: %{x}<br>Consumption: %{y:.2f} kWh<br>Z-score: %{customdata:.2f}<extra></extra>",
                    customdata=anomalies["z_score"],
                )
            )

        fig_anomaly.update_layout(
            title=f"Anomaly Detection: {len(anomalies)} Unusual Readings (|Z| > 2.5)",
            template="plotly_dark",
            xaxis=dict(title="Hour of Day"),
            yaxis=dict(title="Consumption (kWh)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=50, b=30),
        )

        st.plotly_chart(fig_confidence, width="stretch")

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(fig_anomaly, width="stretch")

        with col_b:
            st.subheader("Scenario Simulator")
            hour = st.slider("Hour of Day", min_value=0, max_value=23, value=18, step=1)
            season = st.selectbox("Season", ["Winter", "Summer", "Spring", "Autumn"], index=0)
            stage = st.slider("Load Shedding Stage", min_value=0, max_value=6, value=0, step=1)
            backup = st.radio("Backup Power", ["No", "Yes"], horizontal=True)

            day_of_week = 2
            is_weekend = 0
            month = {"Winter": 7, "Summer": 1, "Spring": 10, "Autumn": 4}.get(season, 1)

            feature_vector = np.array(
                [
                    hour,
                    day_of_week,
                    is_weekend,
                    month,
                    np.sin(2 * np.pi * hour / 24),
                    np.cos(2 * np.pi * hour / 24),
                    np.sin(2 * np.pi * day_of_week / 7),
                    np.cos(2 * np.pi * day_of_week / 7),
                    1 if season == "Autumn" else 0,
                    1 if season == "Spring" else 0,
                    1 if season == "Summer" else 0,
                    1 if season == "Winter" else 0,
                    1 if stage > 0 else 0,
                    0,
                    stage / 6,
                    1 if backup == "Yes" else 0,
                    1,
                    0,
                    0,
                    0,
                    0,
                ]
            ).reshape(1, -1)

            prediction = float(gb_model.predict(feature_vector)[0])

            if prediction < 0.5:
                category, color = "Low", THEME["accent_green"]
            elif prediction < 1.0:
                category, color = "Normal", THEME["accent_blue"]
            elif prediction < 1.5:
                category, color = "Above Average", THEME["accent_orange"]
            else:
                category, color = "High", THEME["accent_red"]

            st.markdown(
                f"<div style='font-size:2.2rem;font-weight:700;color:{color};'>{prediction:.2f} kWh</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Category: {category}")
            st.caption(f"Scenario: {season}, Stage {stage}, {'With' if backup == 'Yes' else 'No'} Backup")
