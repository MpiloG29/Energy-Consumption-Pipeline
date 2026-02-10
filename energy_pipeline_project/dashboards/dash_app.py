"""
================================================================================
SOUTH AFRICA ENERGY CONSUMPTION ANALYTICS DASHBOARD
Advanced ML-Powered Insights for Load Shedding, Consumption Patterns & Forecasting
================================================================================
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import os
import sys
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PROFESSIONAL DARK THEME COLOR PALETTE
# ============================================================================
THEME = {
    # Core colors
    'bg_dark': '#0d1117',
    'bg_card': '#161b22',
    'bg_secondary': '#21262d',
    'border': '#30363d',
    
    # Text colors
    'text_primary': '#f0f6fc',
    'text_secondary': '#8b949e',
    'text_muted': '#6e7681',
    
    # Accent colors
    'accent_blue': '#58a6ff',
    'accent_green': '#3fb950',
    'accent_red': '#f85149',
    'accent_orange': '#d29922',
    'accent_purple': '#a371f7',
    'accent_cyan': '#39c5cf',
    'accent_pink': '#db61a2',
    
    # Chart colors
    'chart_grid': '#21262d',
    'chart_line': '#30363d',
    
    # Seasonal colors
    'winter': '#58a6ff',
    'summer': '#d29922',
    'spring': '#3fb950',
    'autumn': '#f85149',
    
    # Gradient for heatmaps
    'heatmap_scale': [[0, '#0d1117'], [0.5, '#58a6ff'], [1, '#f85149']],
}

# ============================================================================
# DATA LOADING & FEATURE ENGINEERING
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "raw_data", "energy_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "ml_models")

sys.path.insert(0, BASE_DIR)
from ml_models.feature_engineering import engineer_features, get_feature_descriptions

print(f"Loading data from: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"Loaded {len(df):,} records")

# Time-based features
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["date"] = df["timestamp"].dt.date
df["day_name"] = df["timestamp"].dt.day_name()
df["month"] = df["timestamp"].dt.month
df["month_name"] = df["timestamp"].dt.month_name()
df["week"] = df["timestamp"].dt.isocalendar().week
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# Season mapping (Southern Hemisphere)
def get_season(month):
    if month in [6, 7, 8]: return "Winter"
    elif month in [12, 1, 2]: return "Summer"
    elif month in [3, 4, 5]: return "Autumn"
    else: return "Spring"

df["season"] = df["month"].apply(get_season)

# Time of day categories
def get_time_period(hour):
    if 6 <= hour < 9: return "Morning Peak"
    elif 9 <= hour < 17: return "Daytime"
    elif 17 <= hour < 21: return "Evening Peak"
    else: return "Off-Peak"

df["time_period"] = df["hour"].apply(get_time_period)

# Load shedding categories
df["outage_severity"] = pd.cut(
    df["load_shedding_stage"], 
    bins=[-1, 0, 2, 4, 6],
    labels=["No Outage", "Low (1-2)", "Medium (3-4)", "High (5-6)"]
)

# ============================================================================
# LOAD ML MODELS
# ============================================================================
gb_model = rf_model = lr_model = None
try:
    gb_model = joblib.load(os.path.join(MODEL_DIR, "energy_model_gradient_boosting.pkl"))
    rf_model = joblib.load(os.path.join(MODEL_DIR, "energy_model_random_forest.pkl"))
    lr_model = joblib.load(os.path.join(MODEL_DIR, "energy_model_linear.pkl"))
    print("Models loaded successfully")
except Exception as e:
    print(f"Could not load models: {e}")

# Engineer features and get predictions
features_df, target, df_eng = engineer_features(df.copy())
if gb_model is not None:
    df_eng["predicted"] = gb_model.predict(features_df)
    df_eng["residual"] = df_eng["energy_consumption"] - df_eng["predicted"]
    df_eng["abs_error"] = np.abs(df_eng["residual"])
    df_eng["pct_error"] = (df_eng["abs_error"] / df_eng["energy_consumption"].clip(lower=0.01)) * 100

# ============================================================================
# CALCULATE KEY METRICS
# ============================================================================
total_consumption = df["energy_consumption"].sum()
avg_consumption = df["energy_consumption"].mean()
peak_consumption = df["energy_consumption"].max()
outage_hours = (df["load_shedding_stage"] > 0).sum()
total_hours = len(df)
outage_pct = (outage_hours / total_hours) * 100

# Model metrics
if gb_model is not None:
    rmse = np.sqrt(np.mean(df_eng["residual"]**2))
    mae = np.mean(df_eng["abs_error"])
    r2 = 1 - (np.sum(df_eng["residual"]**2) / np.sum((df_eng["energy_consumption"] - df_eng["energy_consumption"].mean())**2))
else:
    rmse = mae = r2 = 0

# Cost estimates (South African electricity tariffs)
TARIFF_PEAK = 2.85  # R/kWh during peak
TARIFF_OFFPEAK = 1.45  # R/kWh off-peak
df["estimated_cost"] = np.where(
    df["time_period"].isin(["Morning Peak", "Evening Peak"]),
    df["energy_consumption"] * TARIFF_PEAK,
    df["energy_consumption"] * TARIFF_OFFPEAK
)

# ============================================================================
# INITIALIZE DASH APP
# ============================================================================
app = dash.Dash(__name__, title="SA Energy Analytics | ML Dashboard")

# ============================================================================
# LAYOUT
# ============================================================================
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1("South Africa Energy Analytics", style={
                'margin': '0', 'fontSize': '1.75rem', 'fontWeight': '700',
                'background': f'linear-gradient(90deg, {THEME["accent_blue"]}, {THEME["accent_purple"]})',
                '-webkit-background-clip': 'text', '-webkit-text-fill-color': 'transparent',
                'backgroundClip': 'text'
            }),
            html.P("ML-Powered Load Shedding & Consumption Intelligence", style={
                'margin': '5px 0 0 0', 'color': THEME['text_secondary'], 'fontSize': '0.9rem'
            })
        ], style={'flex': '1'}),
        html.Div([
            html.Span("●", style={'color': THEME['accent_green'], 'marginRight': '8px'}),
            html.Span("Live Dashboard", style={'color': THEME['text_secondary'], 'fontSize': '0.85rem'}),
            html.Span(f" | {datetime.now().strftime('%Y-%m-%d %H:%M')}", style={
                'color': THEME['text_muted'], 'fontSize': '0.85rem', 'marginLeft': '10px'
            })
        ])
    ], style={
        'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
        'padding': '20px 30px', 'borderBottom': f'1px solid {THEME["border"]}',
        'backgroundColor': THEME['bg_card']
    }),
    
    # KPI Cards Row
    html.Div([
        # Total Consumption
        html.Div([
            html.Div([
                html.Span("▲", style={'color': THEME['accent_blue'], 'marginRight': '8px', 'fontSize': '0.9rem'}),
                html.Span("Total Consumption", style={'color': THEME['text_secondary'], 'fontSize': '0.8rem', 'textTransform': 'uppercase', 'letterSpacing': '1px'})
            ]),
            html.Div(f"{total_consumption:,.0f}", style={
                'fontSize': '2rem', 'fontWeight': '700', 'color': THEME['text_primary'],
                'margin': '10px 0 5px 0', 'fontFamily': 'monospace'
            }),
            html.Div("kWh (12 months)", style={'color': THEME['text_muted'], 'fontSize': '0.8rem'})
        ], className='kpi-card'),
        
        # Model Accuracy
        html.Div([
            html.Div([
                html.Span("◉", style={'color': THEME['accent_green'], 'marginRight': '8px', 'fontSize': '0.9rem'}),
                html.Span("Model R² Score", style={'color': THEME['text_secondary'], 'fontSize': '0.8rem', 'textTransform': 'uppercase', 'letterSpacing': '1px'})
            ]),
            html.Div(f"{r2:.4f}", style={
                'fontSize': '2rem', 'fontWeight': '700', 'color': THEME['accent_green'],
                'margin': '10px 0 5px 0', 'fontFamily': 'monospace'
            }),
            html.Div(f"RMSE: {rmse:.3f} kWh", style={'color': THEME['text_muted'], 'fontSize': '0.8rem'})
        ], className='kpi-card'),
        
        # Outage Impact
        html.Div([
            html.Div([
                html.Span("⚡", style={'color': THEME['accent_orange'], 'marginRight': '8px', 'fontSize': '0.9rem'}),
                html.Span("Load Shedding Hours", style={'color': THEME['text_secondary'], 'fontSize': '0.8rem', 'textTransform': 'uppercase', 'letterSpacing': '1px'})
            ]),
            html.Div(f"{outage_hours:,}", style={
                'fontSize': '2rem', 'fontWeight': '700', 'color': THEME['accent_orange'],
                'margin': '10px 0 5px 0', 'fontFamily': 'monospace'
            }),
            html.Div(f"{outage_pct:.1f}% of time affected", style={'color': THEME['text_muted'], 'fontSize': '0.8rem'})
        ], className='kpi-card'),
        
        # Cost Estimate
        html.Div([
            html.Div([
                html.Span("R", style={'color': THEME['accent_purple'], 'marginRight': '8px', 'fontSize': '0.9rem', 'fontWeight': 'bold'}),
                html.Span("Estimated Cost", style={'color': THEME['text_secondary'], 'fontSize': '0.8rem', 'textTransform': 'uppercase', 'letterSpacing': '1px'})
            ]),
            html.Div(f"R{df['estimated_cost'].sum():,.0f}", style={
                'fontSize': '2rem', 'fontWeight': '700', 'color': THEME['accent_purple'],
                'margin': '10px 0 5px 0', 'fontFamily': 'monospace'
            }),
            html.Div("Based on Eskom TOU tariffs", style={'color': THEME['text_muted'], 'fontSize': '0.8rem'})
        ], className='kpi-card'),
    ], style={
        'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '20px',
        'padding': '25px 30px', 'backgroundColor': THEME['bg_dark']
    }),
    
    # Main Content Tabs
    dcc.Tabs(id="main-tabs", value="tab-overview", children=[
        dcc.Tab(label="📊 Consumption Patterns", value="tab-overview"),
        dcc.Tab(label="⚡ Load Shedding Impact", value="tab-loadshed"),
        dcc.Tab(label="🧠 Model Performance", value="tab-model"),
        dcc.Tab(label="💰 Cost Analysis", value="tab-cost"),
        dcc.Tab(label="🔮 Predictive Insights", value="tab-predict"),
    ], style={'backgroundColor': THEME['bg_card'], 'borderBottom': f'1px solid {THEME["border"]}'}),
    
    # Tab Content
    html.Div(id='tab-content', style={'padding': '25px 30px', 'backgroundColor': THEME['bg_dark']})
    
], style={
    'backgroundColor': THEME['bg_dark'], 'minHeight': '100vh',
    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
})

# ============================================================================
# TAB CONTENT CALLBACK
# ============================================================================
@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tab(tab):
    if tab == "tab-overview":
        return render_overview_tab()
    elif tab == "tab-loadshed":
        return render_loadshed_tab()
    elif tab == "tab-model":
        return render_model_tab()
    elif tab == "tab-cost":
        return render_cost_tab()
    elif tab == "tab-predict":
        return render_predict_tab()
    return html.Div("Select a tab")

# ============================================================================
# TAB 1: CONSUMPTION PATTERNS
# ============================================================================
def render_overview_tab():
    # 1. Time Series with Rolling Average
    daily = df.groupby('date').agg({
        'energy_consumption': 'sum',
        'load_shedding_stage': 'max'
    }).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    daily['rolling_7d'] = daily['energy_consumption'].rolling(7).mean()
    daily['rolling_30d'] = daily['energy_consumption'].rolling(30).mean()
    
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=daily['date'], y=daily['energy_consumption'],
        mode='lines', name='Daily Total',
        line=dict(color=THEME['text_muted'], width=1),
        opacity=0.5
    ))
    fig_ts.add_trace(go.Scatter(
        x=daily['date'], y=daily['rolling_7d'],
        mode='lines', name='7-Day MA',
        line=dict(color=THEME['accent_blue'], width=2)
    ))
    fig_ts.add_trace(go.Scatter(
        x=daily['date'], y=daily['rolling_30d'],
        mode='lines', name='30-Day MA',
        line=dict(color=THEME['accent_purple'], width=2)
    ))
    
    # Add outage markers
    outage_days = daily[daily['load_shedding_stage'] > 0]
    fig_ts.add_trace(go.Scatter(
        x=outage_days['date'], y=outage_days['energy_consumption'],
        mode='markers', name='Outage Days',
        marker=dict(color=THEME['accent_red'], size=4, symbol='x')
    ))
    
    fig_ts.update_layout(
        title=dict(text='Consumption Trend with Load Shedding Overlay', font=dict(color=THEME['text_primary'])),
        plot_bgcolor=THEME['bg_card'], paper_bgcolor=THEME['bg_card'],
        font=dict(color=THEME['text_secondary']),
        xaxis=dict(gridcolor=THEME['chart_grid'], showgrid=True),
        yaxis=dict(gridcolor=THEME['chart_grid'], showgrid=True, title='kWh'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=20, t=60, b=40),
        hovermode='x unified'
    )
    
    # 2. Hour × Day Heatmap with Annotations
    heatmap_data = df.pivot_table(
        values='energy_consumption', index='day_name', columns='hour', aggfunc='mean'
    )
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=[[0, THEME['bg_dark']], [0.3, THEME['accent_blue']], [0.7, THEME['accent_orange']], [1, THEME['accent_red']]],
        colorbar=dict(title='kWh', tickfont=dict(color=THEME['text_secondary'])),
        hovertemplate='Hour: %{x}<br>Day: %{y}<br>Avg: %{z:.2f} kWh<extra></extra>'
    ))
    
    # Add peak indicators
    max_val = heatmap_data.max().max()
    for i, day in enumerate(day_order):
        max_hour = heatmap_data.loc[day].idxmax()
        fig_heatmap.add_annotation(
            x=max_hour, y=day, text='●', showarrow=False,
            font=dict(color='white', size=8)
        )
    
    fig_heatmap.update_layout(
        title=dict(text='Consumption Heatmap: Hour × Day (● = Daily Peak)', font=dict(color=THEME['text_primary'])),
        plot_bgcolor=THEME['bg_card'], paper_bgcolor=THEME['bg_card'],
        font=dict(color=THEME['text_secondary']),
        xaxis=dict(title='Hour of Day', dtick=1),
        yaxis=dict(title=''),
        margin=dict(l=100, r=20, t=60, b=40)
    )
    
    # 3. Seasonal Distributions (Box plots)
    fig_seasonal = go.Figure()
    season_colors = {'Winter': THEME['winter'], 'Summer': THEME['summer'], 
                     'Spring': THEME['spring'], 'Autumn': THEME['autumn']}
    
    for season in ['Summer', 'Autumn', 'Winter', 'Spring']:
        season_data = df[df['season'] == season]['energy_consumption']
        fig_seasonal.add_trace(go.Box(
            y=season_data, name=season,
            marker_color=season_colors[season],
            boxmean='sd',
            hovertemplate=f'{season}<br>Value: %{{y:.2f}} kWh<extra></extra>'
        ))
    
    fig_seasonal.update_layout(
        title=dict(text='Consumption Distribution by Season (South African Calendar)', font=dict(color=THEME['text_primary'])),
        plot_bgcolor=THEME['bg_card'], paper_bgcolor=THEME['bg_card'],
        font=dict(color=THEME['text_secondary']),
        yaxis=dict(gridcolor=THEME['chart_grid'], title='kWh'),
        showlegend=False,
        margin=dict(l=50, r=20, t=60, b=40)
    )
    
    # 4. Weekday vs Weekend comparison
    weekend_comp = df.groupby(['hour', 'is_weekend'])['energy_consumption'].mean().unstack()
    
    fig_weekend = go.Figure()
    fig_weekend.add_trace(go.Scatter(
        x=weekend_comp.index, y=weekend_comp[0],
        mode='lines+markers', name='Weekday',
        line=dict(color=THEME['accent_blue'], width=2),
        marker=dict(size=6)
    ))
    fig_weekend.add_trace(go.Scatter(
        x=weekend_comp.index, y=weekend_comp[1],
        mode='lines+markers', name='Weekend',
        line=dict(color=THEME['accent_orange'], width=2),
        marker=dict(size=6)
    ))
    
    # Add peak period shading
    fig_weekend.add_vrect(x0=6, x1=9, fillcolor=THEME['accent_red'], opacity=0.1, line_width=0, annotation_text='AM Peak')
    fig_weekend.add_vrect(x0=17, x1=21, fillcolor=THEME['accent_red'], opacity=0.1, line_width=0, annotation_text='PM Peak')
    
    fig_weekend.update_layout(
        title=dict(text='Weekday vs Weekend Load Profile', font=dict(color=THEME['text_primary'])),
        plot_bgcolor=THEME['bg_card'], paper_bgcolor=THEME['bg_card'],
        font=dict(color=THEME['text_secondary']),
        xaxis=dict(gridcolor=THEME['chart_grid'], title='Hour of Day', dtick=2),
        yaxis=dict(gridcolor=THEME['chart_grid'], title='Avg Consumption (kWh)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=20, t=60, b=40)
    )
    
    return html.Div([
        html.Div([
            html.Div([dcc.Graph(figure=fig_ts)], className='chart-card', style={'gridColumn': 'span 2'}),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px', 'marginBottom': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(figure=fig_heatmap)], className='chart-card'),
            html.Div([dcc.Graph(figure=fig_seasonal)], className='chart-card'),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px', 'marginBottom': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(figure=fig_weekend)], className='chart-card', style={'gridColumn': 'span 2'}),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'})
    ])

# ============================================================================
# TAB 2: LOAD SHEDDING IMPACT
# ============================================================================
def render_loadshed_tab():
    # 1. Consumption by Load Shedding Stage
    stage_impact = df.groupby('load_shedding_stage')['energy_consumption'].agg(['mean', 'std', 'count']).reset_index()
    stage_impact.columns = ['stage', 'mean', 'std', 'count']
    
    fig_stages = go.Figure()
    fig_stages.add_trace(go.Bar(
        x=[f'Stage {int(s)}' if s > 0 else 'No Outage' for s in stage_impact['stage']],
        y=stage_impact['mean'],
        error_y=dict(type='data', array=stage_impact['std'], visible=True),
        marker=dict(
            color=[THEME['accent_green'] if s == 0 else 
                   THEME['accent_orange'] if s <= 2 else 
                   THEME['accent_red'] for s in stage_impact['stage']]
        ),
        text=[f'n={c:,}' for c in stage_impact['count']],
        textposition='outside',
        hovertemplate='%{x}<br>Mean: %{y:.3f} kWh<br>Std: %{error_y.array:.3f}<extra></extra>'
    ))
    
    fig_stages.update_layout(
        title=dict(text='Consumption by Load Shedding Stage (Mean ± Std Dev)', font=dict(color=THEME['text_primary'])),
        plot_bgcolor=THEME['bg_card'], paper_bgcolor=THEME['bg_card'],
        font=dict(color=THEME['text_secondary']),
        yaxis=dict(gridcolor=THEME['chart_grid'], title='Avg Consumption (kWh)'),
        xaxis=dict(title=''),
        margin=dict(l=50, r=20, t=60, b=40),
        showlegend=False
    )
    
    # 2. Backup Power Effect
    backup_data = df.groupby(['load_shedding_stage', 'backup_power'])['energy_consumption'].mean().unstack()
    
    fig_backup = go.Figure()
    fig_backup.add_trace(go.Bar(
        name='No Backup',
        x=[f'Stage {int(s)}' if s > 0 else 'Normal' for s in backup_data.index],
        y=backup_data[False] if False in backup_data.columns else [0]*len(backup_data),
        marker_color=THEME['accent_red'],
        text=[f'{v:.2f}' for v in (backup_data[False] if False in backup_data.columns else [0]*len(backup_data))],
        textposition='auto'
    ))
    fig_backup.add_trace(go.Bar(
        name='With Backup',
        x=[f'Stage {int(s)}' if s > 0 else 'Normal' for s in backup_data.index],
        y=backup_data[True] if True in backup_data.columns else [0]*len(backup_data),
        marker_color=THEME['accent_green'],
        text=[f'{v:.2f}' for v in (backup_data[True] if True in backup_data.columns else [0]*len(backup_data))],
        textposition='auto'
    ))
    
    fig_backup.update_layout(
        title=dict(text='Backup Power Effect: Consumption During Outages', font=dict(color=THEME['text_primary'])),
        plot_bgcolor=THEME['bg_card'], paper_bgcolor=THEME['bg_card'],
        font=dict(color=THEME['text_secondary']),
        yaxis=dict(gridcolor=THEME['chart_grid'], title='Avg Consumption (kWh)'),
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=20, t=60, b=40)
    )
    
    # 3. Pre/During/Post Outage Analysis
    df_sorted = df.sort_values(['household_id', 'timestamp']).copy()
    df_sorted['prev_outage'] = df_sorted.groupby('household_id')['load_shedding_stage'].shift(1).fillna(0)
    df_sorted['next_outage'] = df_sorted.groupby('household_id')['load_shedding_stage'].shift(-1).fillna(0)
    
    pre_outage = df_sorted[(df_sorted['load_shedding_stage'] == 0) & (df_sorted['next_outage'] > 0)]['energy_consumption'].mean()
    during_outage = df_sorted[df_sorted['load_shedding_stage'] > 0]['energy_consumption'].mean()
    post_outage = df_sorted[(df_sorted['load_shedding_stage'] == 0) & (df_sorted['prev_outage'] > 0)]['energy_consumption'].mean()
    normal = df_sorted[(df_sorted['load_shedding_stage'] == 0) & (df_sorted['prev_outage'] == 0) & (df_sorted['next_outage'] == 0)]['energy_consumption'].mean()
    
    fig_transition = go.Figure()
    categories = ['Normal', 'Pre-Outage', 'During Outage', 'Post-Outage']
    values = [normal, pre_outage, during_outage, post_outage]
    colors = [THEME['accent_blue'], THEME['accent_orange'], THEME['accent_red'], THEME['accent_purple']]
    
    fig_transition.add_trace(go.Waterfall(
        x=categories,
        y=[normal, pre_outage - normal, during_outage - pre_outage, post_outage - during_outage],
        measure=['absolute', 'relative', 'relative', 'relative'],
        text=[f'{v:.3f}' for v in values],
        textposition='outside',
        connector=dict(line=dict(color=THEME['text_muted'])),
        increasing=dict(marker=dict(color=THEME['accent_green'])),
        decreasing=dict(marker=dict(color=THEME['accent_red'])),
        totals=dict(marker=dict(color=THEME['accent_blue']))
    ))
    
    fig_transition.update_layout(
        title=dict(text='Load Shedding Transition Impact (Waterfall)', font=dict(color=THEME['text_primary'])),
        plot_bgcolor=THEME['bg_card'], paper_bgcolor=THEME['bg_card'],
        font=dict(color=THEME['text_secondary']),
        yaxis=dict(gridcolor=THEME['chart_grid'], title='Consumption (kWh)'),
        margin=dict(l=50, r=20, t=60, b=40),
        showlegend=False
    )
    
    # 4. Outage Distribution by Hour
    outage_by_hour = df[df['load_shedding_stage'] > 0].groupby('hour').size()
    normal_by_hour = df[df['load_shedding_stage'] == 0].groupby('hour').size()
    
    fig_outage_dist = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_outage_dist.add_trace(go.Bar(
        x=outage_by_hour.index,
        y=outage_by_hour.values,
        name='Outage Hours',
        marker_color=THEME['accent_red'],
        opacity=0.7
    ), secondary_y=False)
    
    fig_outage_dist.add_trace(go.Scatter(
        x=df.groupby('hour')['energy_consumption'].mean().index,
        y=df.groupby('hour')['energy_consumption'].mean().values,
        name='Avg Consumption',
        line=dict(color=THEME['accent_blue'], width=3),
        mode='lines+markers'
    ), secondary_y=True)
    
    fig_outage_dist.update_layout(
        title=dict(text='Outage Frequency vs Consumption by Hour', font=dict(color=THEME['text_primary'])),
        plot_bgcolor=THEME['bg_card'], paper_bgcolor=THEME['bg_card'],
        font=dict(color=THEME['text_secondary']),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=50, t=60, b=40)
    )
    fig_outage_dist.update_yaxes(title_text="Outage Count", gridcolor=THEME['chart_grid'], secondary_y=False)
    fig_outage_dist.update_yaxes(title_text="Avg kWh", gridcolor=THEME['chart_grid'], secondary_y=True)
    
    return html.Div([
        html.Div([
            html.Div([dcc.Graph(figure=fig_stages)], className='chart-card'),
            html.Div([dcc.Graph(figure=fig_backup)], className='chart-card'),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px', 'marginBottom': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(figure=fig_transition)], className='chart-card'),
            html.Div([dcc.Graph(figure=fig_outage_dist)], className='chart-card'),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'})
    ])

# ============================================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================================
def render_model_tab():
    if gb_model is None:
        return html.Div([
            html.P("Models not loaded. Run train_model.py first.", style={'color': THEME['accent_red']})
        ])
    
    # 1. Actual vs Predicted Scatter
    sample = df_eng.sample(min(5000, len(df_eng)), random_state=42)
    
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scattergl(
        x=sample['energy_consumption'],
        y=sample['predicted'],
        mode='markers',
        marker=dict(
            color=sample['abs_error'],
            colorscale=[[0, THEME['accent_green']], [0.5, THEME['accent_orange']], [1, THEME['accent_red']]],
            size=4,
            opacity=0.6,
            colorbar=dict(title='Error', tickfont=dict(color=THEME['text_secondary']))
        ),
        hovertemplate='Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>'
    ))
    
    # Perfect prediction line
    max_val = max(sample['energy_consumption'].max(), sample['predicted'].max())
    fig_scatter.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode='lines', name='Perfect Prediction',
        line=dict(color=THEME['accent_blue'], dash='dash', width=2)
    ))
    
    fig_scatter.update_layout(
        title=dict(text=f'Actual vs Predicted (R² = {r2:.4f})', font=dict(color=THEME['text_primary'])),
        plot_bgcolor=THEME['bg_card'], paper_bgcolor=THEME['bg_card'],
        font=dict(color=THEME['text_secondary']),
        xaxis=dict(gridcolor=THEME['chart_grid'], title='Actual (kWh)'),
        yaxis=dict(gridcolor=THEME['chart_grid'], title='Predicted (kWh)'),
        margin=dict(l=50, r=20, t=60, b=40),
        showlegend=False
    )
    
    # 2. Residuals Distribution
    fig_residuals = make_subplots(rows=1, cols=2, subplot_titles=('Residual Distribution', 'Residuals vs Predicted'))
    
    fig_residuals.add_trace(go.Histogram(
        x=df_eng['residual'],
        nbinsx=50,
        marker_color=THEME['accent_purple'],
        opacity=0.7,
        name='Residuals'
    ), row=1, col=1)
    
    # Add normal distribution overlay
    x_range = np.linspace(df_eng['residual'].min(), df_eng['residual'].max(), 100)
    normal_pdf = stats.norm.pdf(x_range, df_eng['residual'].mean(), df_eng['residual'].std())
    fig_residuals.add_trace(go.Scatter(
        x=x_range, y=normal_pdf * len(df_eng['residual']) * (df_eng['residual'].max() - df_eng['residual'].min()) / 50,
        mode='lines', name='Normal Fit',
        line=dict(color=THEME['accent_orange'], width=2)
    ), row=1, col=1)
    
    # Residuals vs Predicted
    sample_res = df_eng.sample(min(3000, len(df_eng)), random_state=42)
    fig_residuals.add_trace(go.Scattergl(
        x=sample_res['predicted'],
        y=sample_res['residual'],
        mode='markers',
        marker=dict(color=THEME['accent_blue'], size=3, opacity=0.4),
        name='Residuals'
    ), row=1, col=2)
    
    fig_residuals.add_hline(y=0, line=dict(color=THEME['accent_red'], dash='dash'), row=1, col=2)
    
    fig_residuals.update_layout(
        plot_bgcolor=THEME['bg_card'], paper_bgcolor=THEME['bg_card'],
        font=dict(color=THEME['text_secondary']),
        showlegend=False,
        margin=dict(l=50, r=20, t=60, b=40)
    )
    fig_residuals.update_xaxes(gridcolor=THEME['chart_grid'])
    fig_residuals.update_yaxes(gridcolor=THEME['chart_grid'])
    
    # 3. Feature Importance (from Random Forest)
    if rf_model is not None:
        feature_names = [
            "hour", "day_of_week", "is_weekend", "month",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "season_Autumn", "season_Spring", "season_Summer", "season_Winter",
            "is_load_shedding", "post_load_shedding", "load_shedding_stage_normalized",
            "has_backup_power",
            "household_House_1", "household_House_2", "household_House_3",
            "household_House_4", "household_House_5"
        ]
        importances = rf_model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:15]
        top_features = [feature_names[i] for i in indices]
        top_importances = [importances[i] for i in indices]
        
        # Color by category
        colors = []
        for f in top_features:
            if 'hour' in f.lower() or 'day' in f.lower() or 'weekend' in f.lower():
                colors.append(THEME['accent_blue'])
            elif 'season' in f.lower() or 'month' in f.lower():
                colors.append(THEME['accent_green'])
            elif 'load_shedding' in f.lower() or 'backup' in f.lower():
                colors.append(THEME['accent_red'])
            elif 'household' in f.lower():
                colors.append(THEME['accent_purple'])
            else:
                colors.append(THEME['accent_orange'])
        
        fig_importance = go.Figure()
        fig_importance.add_trace(go.Bar(
            x=top_importances[::-1],
            y=top_features[::-1],
            orientation='h',
            marker_color=colors[::-1],
            text=[f'{v:.3f}' for v in top_importances[::-1]],
            textposition='outside'
        ))
        
        fig_importance.update_layout(
            title=dict(text='Top 15 Feature Importances (Random Forest)', font=dict(color=THEME['text_primary'])),
            plot_bgcolor=THEME['bg_card'], paper_bgcolor=THEME['bg_card'],
            font=dict(color=THEME['text_secondary']),
            xaxis=dict(gridcolor=THEME['chart_grid'], title='Importance'),
            yaxis=dict(title=''),
            margin=dict(l=180, r=60, t=60, b=40)
        )
    else:
        fig_importance = go.Figure()
        fig_importance.add_annotation(text="Random Forest model not available", x=0.5, y=0.5)
    
    # 4. Error by Scenario
    scenarios = {
        'Normal Hours': df_eng[df_eng['load_shedding_stage'] == 0],
        'During Outage': df_eng[df_eng['is_load_shedding'] == 1],
        'Winter Peak': df_eng[(df_eng['season'] == 'Winter') & (df_eng['hour'].between(17, 21))],
        'Summer Midday': df_eng[(df_eng['season'] == 'Summer') & (df_eng['hour'].between(11, 14))],
        'Weekend': df_eng[df_eng['is_weekend'] == 1],
        'With Backup': df_eng[df_eng['has_backup_power'] == 1]
    }
    
    scenario_metrics = []
    for name, data in scenarios.items():
        if len(data) > 0:
            rmse_s = np.sqrt(np.mean(data['residual']**2))
            mae_s = np.mean(data['abs_error'])
            scenario_metrics.append({'Scenario': name, 'RMSE': rmse_s, 'MAE': mae_s, 'n': len(data)})
    
    scenario_df = pd.DataFrame(scenario_metrics)
    
    fig_scenarios = go.Figure()
    fig_scenarios.add_trace(go.Bar(
        name='RMSE',
        x=scenario_df['Scenario'],
        y=scenario_df['RMSE'],
        marker_color=THEME['accent_blue'],
        text=[f'{v:.3f}' for v in scenario_df['RMSE']],
        textposition='outside'
    ))
    fig_scenarios.add_trace(go.Bar(
        name='MAE',
        x=scenario_df['Scenario'],
        y=scenario_df['MAE'],
        marker_color=THEME['accent_purple'],
        text=[f'{v:.3f}' for v in scenario_df['MAE']],
        textposition='outside'
    ))
    
    fig_scenarios.update_layout(
        title=dict(text='Model Error by Scenario', font=dict(color=THEME['text_primary'])),
        plot_bgcolor=THEME['bg_card'], paper_bgcolor=THEME['bg_card'],
        font=dict(color=THEME['text_secondary']),
        xaxis=dict(gridcolor=THEME['chart_grid']),
        yaxis=dict(gridcolor=THEME['chart_grid'], title='Error (kWh)'),
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=20, t=60, b=40)
    )
    
    return html.Div([
        html.Div([
            html.Div([dcc.Graph(figure=fig_scatter)], className='chart-card'),
            html.Div([dcc.Graph(figure=fig_residuals)], className='chart-card'),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px', 'marginBottom': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(figure=fig_importance)], className='chart-card'),
            html.Div([dcc.Graph(figure=fig_scenarios)], className='chart-card'),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'})
    ])

# ============================================================================
# TAB 4: COST ANALYSIS
# ============================================================================
def render_cost_tab():
    # 1. Monthly Cost Breakdown
    monthly_cost = df.groupby(['month_name', 'time_period'])['estimated_cost'].sum().unstack()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_cost = monthly_cost.reindex([m for m in month_order if m in monthly_cost.index])
    
    fig_monthly = go.Figure()
    period_colors = {
        'Morning Peak': THEME['accent_orange'],
        'Daytime': THEME['accent_blue'],
        'Evening Peak': THEME['accent_red'],
        'Off-Peak': THEME['accent_green']
    }
    
    for period in ['Off-Peak', 'Daytime', 'Morning Peak', 'Evening Peak']:
        if period in monthly_cost.columns:
            fig_monthly.add_trace(go.Bar(
                name=period,
                x=monthly_cost.index,
                y=monthly_cost[period],
                marker_color=period_colors.get(period, THEME['text_muted'])
            ))
    
    fig_monthly.update_layout(
        title=dict(text='Monthly Cost by Time-of-Use Period', font=dict(color=THEME['text_primary'])),
        plot_bgcolor=THEME['bg_card'], paper_bgcolor=THEME['bg_card'],
        font=dict(color=THEME['text_secondary']),
        xaxis=dict(gridcolor=THEME['chart_grid'], title=''),
        yaxis=dict(gridcolor=THEME['chart_grid'], title='Cost (R)'),
        barmode='stack',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=20, t=60, b=40)
    )
    
    # 2. Household Cost Comparison
    household_cost = df.groupby('household_id').agg({
        'estimated_cost': 'sum',
        'energy_consumption': 'sum',
        'backup_power': 'first'
    }).reset_index()
    household_cost['cost_per_kwh'] = household_cost['estimated_cost'] / household_cost['energy_consumption']
    
    fig_household_cost = go.Figure()
    colors = [THEME['accent_green'] if bp else THEME['accent_red'] for bp in household_cost['backup_power']]
    
    fig_household_cost.add_trace(go.Bar(
        x=household_cost['household_id'],
        y=household_cost['estimated_cost'],
        marker_color=colors,
        text=[f'R{c:,.0f}' for c in household_cost['estimated_cost']],
        textposition='outside',
        hovertemplate='%{x}<br>Total: R%{y:,.0f}<br>R/kWh: %{customdata:.2f}<extra></extra>',
        customdata=household_cost['cost_per_kwh']
    ))
    
    fig_household_cost.update_layout(
        title=dict(text='Annual Cost by Household (Green = Has Backup)', font=dict(color=THEME['text_primary'])),
        plot_bgcolor=THEME['bg_card'], paper_bgcolor=THEME['bg_card'],
        font=dict(color=THEME['text_secondary']),
        xaxis=dict(gridcolor=THEME['chart_grid'], title=''),
        yaxis=dict(gridcolor=THEME['chart_grid'], title='Total Cost (R)'),
        margin=dict(l=50, r=20, t=60, b=40)
    )
    
    # 3. Peak vs Off-Peak Savings Opportunity
    peak_consumption = df[df['time_period'].isin(['Morning Peak', 'Evening Peak'])]['energy_consumption'].sum()
    offpeak_consumption = df[~df['time_period'].isin(['Morning Peak', 'Evening Peak'])]['energy_consumption'].sum()
    
    current_cost = peak_consumption * TARIFF_PEAK + offpeak_consumption * TARIFF_OFFPEAK
    
    # If 20% of peak shifted to off-peak
    shift_pct = 0.20
    shifted_peak = peak_consumption * (1 - shift_pct)
    shifted_offpeak = offpeak_consumption + (peak_consumption * shift_pct)
    potential_cost = shifted_peak * TARIFF_PEAK + shifted_offpeak * TARIFF_OFFPEAK
    savings = current_cost - potential_cost
    
    fig_savings = go.Figure()
    fig_savings.add_trace(go.Indicator(
        mode="number+delta",
        value=savings,
        number=dict(prefix="R ", valueformat=",.0f", font=dict(color=THEME['accent_green'], size=48)),
        delta=dict(reference=0, relative=False, valueformat=".0f"),
        title=dict(text="Potential Annual Savings<br><span style='font-size:0.8em;color:gray'>If 20% peak load shifted to off-peak</span>", font=dict(color=THEME['text_primary']))
    ))
    
    fig_savings.update_layout(
        plot_bgcolor=THEME['bg_card'], paper_bgcolor=THEME['bg_card'],
        margin=dict(l=20, r=20, t=100, b=20)
    )
    
    # 4. Cost vs Outage Frequency
    daily_data = df.groupby('date').agg({
        'estimated_cost': 'sum',
        'load_shedding_stage': lambda x: (x > 0).sum(),
        'energy_consumption': 'sum'
    }).reset_index()
    daily_data.columns = ['date', 'cost', 'outage_hours', 'consumption']
    
    fig_cost_outage = go.Figure()
    fig_cost_outage.add_trace(go.Scatter(
        x=daily_data['outage_hours'],
        y=daily_data['cost'],
        mode='markers',
        marker=dict(
            color=daily_data['consumption'],
            colorscale=[[0, THEME['accent_blue']], [1, THEME['accent_red']]],
            size=8,
            opacity=0.6,
            colorbar=dict(title='kWh', tickfont=dict(color=THEME['text_secondary']))
        ),
        hovertemplate='Outage Hours: %{x}<br>Daily Cost: R%{y:.0f}<extra></extra>'
    ))
    
    # Add trend line
    if len(daily_data) > 10:
        z = np.polyfit(daily_data['outage_hours'], daily_data['cost'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(daily_data['outage_hours'].min(), daily_data['outage_hours'].max(), 100)
        fig_cost_outage.add_trace(go.Scatter(
            x=x_line, y=p(x_line),
            mode='lines', name='Trend',
            line=dict(color=THEME['accent_orange'], dash='dash', width=2)
        ))
    
    fig_cost_outage.update_layout(
        title=dict(text='Daily Cost vs Outage Hours', font=dict(color=THEME['text_primary'])),
        plot_bgcolor=THEME['bg_card'], paper_bgcolor=THEME['bg_card'],
        font=dict(color=THEME['text_secondary']),
        xaxis=dict(gridcolor=THEME['chart_grid'], title='Outage Hours per Day'),
        yaxis=dict(gridcolor=THEME['chart_grid'], title='Daily Cost (R)'),
        showlegend=False,
        margin=dict(l=50, r=20, t=60, b=40)
    )
    
    return html.Div([
        html.Div([
            html.Div([dcc.Graph(figure=fig_monthly)], className='chart-card', style={'gridColumn': 'span 2'}),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px', 'marginBottom': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(figure=fig_household_cost)], className='chart-card'),
            html.Div([dcc.Graph(figure=fig_savings)], className='chart-card'),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px', 'marginBottom': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(figure=fig_cost_outage)], className='chart-card', style={'gridColumn': 'span 2'}),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'})
    ])

# ============================================================================
# TAB 5: PREDICTIVE INSIGHTS
# ============================================================================
def render_predict_tab():
    if gb_model is None:
        return html.Div([
            html.P("Models not loaded. Run train_model.py first.", style={'color': THEME['accent_red']})
        ])
    
    # 1. Prediction Confidence Bands
    daily_pred = df_eng.groupby('date').agg({
        'energy_consumption': 'sum',
        'predicted': 'sum',
        'residual': ['mean', 'std']
    }).reset_index()
    daily_pred.columns = ['date', 'actual', 'predicted', 'residual_mean', 'residual_std']
    daily_pred['date'] = pd.to_datetime(daily_pred['date'])
    daily_pred['upper'] = daily_pred['predicted'] + 1.96 * daily_pred['residual_std'] * np.sqrt(24)
    daily_pred['lower'] = daily_pred['predicted'] - 1.96 * daily_pred['residual_std'] * np.sqrt(24)
    
    fig_confidence = go.Figure()
    
    # Confidence band
    fig_confidence.add_trace(go.Scatter(
        x=pd.concat([daily_pred['date'], daily_pred['date'][::-1]]),
        y=pd.concat([daily_pred['upper'], daily_pred['lower'][::-1]]),
        fill='toself',
        fillcolor=f'rgba(88, 166, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% CI',
        hoverinfo='skip'
    ))
    
    fig_confidence.add_trace(go.Scatter(
        x=daily_pred['date'], y=daily_pred['actual'],
        mode='lines', name='Actual',
        line=dict(color=THEME['text_secondary'], width=1),
        opacity=0.7
    ))
    
    fig_confidence.add_trace(go.Scatter(
        x=daily_pred['date'], y=daily_pred['predicted'],
        mode='lines', name='Predicted',
        line=dict(color=THEME['accent_blue'], width=2)
    ))
    
    fig_confidence.update_layout(
        title=dict(text='Daily Consumption: Actual vs Predicted with 95% Confidence', font=dict(color=THEME['text_primary'])),
        plot_bgcolor=THEME['bg_card'], paper_bgcolor=THEME['bg_card'],
        font=dict(color=THEME['text_secondary']),
        xaxis=dict(gridcolor=THEME['chart_grid'], title=''),
        yaxis=dict(gridcolor=THEME['chart_grid'], title='Daily Consumption (kWh)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=20, t=60, b=40),
        hovermode='x unified'
    )
    
    # 2. Anomaly Detection
    df_eng['z_score'] = (df_eng['energy_consumption'] - df_eng['energy_consumption'].mean()) / df_eng['energy_consumption'].std()
    df_eng['is_anomaly'] = np.abs(df_eng['z_score']) > 2.5
    
    anomalies = df_eng[df_eng['is_anomaly']].copy()
    
    fig_anomaly = go.Figure()
    
    # Normal points
    normal = df_eng[~df_eng['is_anomaly']].sample(min(2000, len(df_eng[~df_eng['is_anomaly']])), random_state=42)
    fig_anomaly.add_trace(go.Scattergl(
        x=normal['hour'],
        y=normal['energy_consumption'],
        mode='markers',
        name='Normal',
        marker=dict(color=THEME['text_muted'], size=3, opacity=0.3)
    ))
    
    # Anomalies
    if len(anomalies) > 0:
        fig_anomaly.add_trace(go.Scatter(
            x=anomalies['hour'],
            y=anomalies['energy_consumption'],
            mode='markers',
            name=f'Anomalies ({len(anomalies)})',
            marker=dict(color=THEME['accent_red'], size=8, symbol='x'),
            hovertemplate='Hour: %{x}<br>Consumption: %{y:.2f} kWh<br>Z-score: %{customdata:.2f}<extra></extra>',
            customdata=anomalies['z_score']
        ))
    
    fig_anomaly.update_layout(
        title=dict(text=f'Anomaly Detection: {len(anomalies)} Unusual Readings (|Z| > 2.5)', font=dict(color=THEME['text_primary'])),
        plot_bgcolor=THEME['bg_card'], paper_bgcolor=THEME['bg_card'],
        font=dict(color=THEME['text_secondary']),
        xaxis=dict(gridcolor=THEME['chart_grid'], title='Hour of Day'),
        yaxis=dict(gridcolor=THEME['chart_grid'], title='Consumption (kWh)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=20, t=60, b=40)
    )
    
    # 3. Scenario Simulator
    return html.Div([
        html.Div([
            html.Div([dcc.Graph(figure=fig_confidence)], className='chart-card', style={'gridColumn': 'span 2'}),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px', 'marginBottom': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(figure=fig_anomaly)], className='chart-card'),
            
            # Scenario Simulator Panel
            html.Div([
                html.H3("Scenario Simulator", style={'color': THEME['text_primary'], 'marginBottom': '20px'}),
                
                html.Div([
                    html.Label("Hour of Day", style={'color': THEME['text_secondary'], 'fontSize': '0.85rem'}),
                    dcc.Slider(id='sim-hour', min=0, max=23, value=18, 
                              marks={i: {'label': str(i), 'style': {'color': THEME['text_muted']}} for i in range(0, 24, 4)},
                              className='custom-slider')
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Label("Season", style={'color': THEME['text_secondary'], 'fontSize': '0.85rem'}),
                    dcc.Dropdown(
                        id='sim-season',
                        options=[{'label': s, 'value': s} for s in ['Winter', 'Summer', 'Spring', 'Autumn']],
                        value='Winter',
                        style={'backgroundColor': THEME['bg_secondary'], 'color': THEME['text_primary']}
                    )
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Label("Load Shedding Stage", style={'color': THEME['text_secondary'], 'fontSize': '0.85rem'}),
                    dcc.Slider(id='sim-stage', min=0, max=6, value=0,
                              marks={i: {'label': str(i), 'style': {'color': THEME['text_muted']}} for i in range(0, 7)},
                              className='custom-slider')
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Label("Backup Power", style={'color': THEME['text_secondary'], 'fontSize': '0.85rem', 'marginRight': '15px'}),
                    dcc.RadioItems(
                        id='sim-backup',
                        options=[{'label': ' No', 'value': 0}, {'label': ' Yes', 'value': 1}],
                        value=0,
                        inline=True,
                        style={'color': THEME['text_primary']}
                    )
                ], style={'marginBottom': '25px'}),
                
                html.Div(id='sim-output', style={
                    'backgroundColor': THEME['bg_secondary'],
                    'padding': '20px',
                    'borderRadius': '8px',
                    'border': f'1px solid {THEME["border"]}'
                })
            ], className='chart-card')
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'})
    ])

# Scenario Simulator Callback
@app.callback(
    Output('sim-output', 'children'),
    [Input('sim-hour', 'value'),
     Input('sim-season', 'value'),
     Input('sim-stage', 'value'),
     Input('sim-backup', 'value')]
)
def update_simulation(hour, season, stage, backup):
    if gb_model is None:
        return html.P("Model not loaded", style={'color': THEME['accent_red']})
    
    # Build feature vector
    day_of_week = 2  # Wednesday
    is_weekend = 0
    month = {'Winter': 7, 'Summer': 1, 'Spring': 10, 'Autumn': 4}.get(season, 1)
    
    feature_vector = np.array([
        hour, day_of_week, is_weekend, month,
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * day_of_week / 7), np.cos(2 * np.pi * day_of_week / 7),
        1 if season == 'Autumn' else 0, 1 if season == 'Spring' else 0,
        1 if season == 'Summer' else 0, 1 if season == 'Winter' else 0,
        1 if stage > 0 else 0, 0, stage / 6,
        backup,
        1, 0, 0, 0, 0  # House_1 default
    ]).reshape(1, -1)
    
    prediction = float(gb_model.predict(feature_vector)[0])
    
    # Categorize
    if prediction < 0.5:
        category, color = "Low", THEME['accent_green']
    elif prediction < 1.0:
        category, color = "Normal", THEME['accent_blue']
    elif prediction < 1.5:
        category, color = "Above Average", THEME['accent_orange']
    else:
        category, color = "High", THEME['accent_red']
    
    return html.Div([
        html.Div([
            html.Span(f"{prediction:.2f}", style={
                'fontSize': '2.5rem', 'fontWeight': '700', 'color': color,
                'fontFamily': 'monospace'
            }),
            html.Span(" kWh", style={'fontSize': '1rem', 'color': THEME['text_secondary']})
        ]),
        html.P(f"Category: {category}", style={'color': THEME['text_secondary'], 'margin': '10px 0'}),
        html.P(f"Scenario: {season}, Stage {stage}, {'With' if backup else 'No'} Backup", 
               style={'color': THEME['text_muted'], 'fontSize': '0.85rem'})
    ])

# ============================================================================
# CSS STYLING
# ============================================================================
app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            * {{
                box-sizing: border-box;
            }}
            body {{
                margin: 0;
                padding: 0;
                background-color: {THEME['bg_dark']};
                color: {THEME['text_primary']};
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            }}
            
            .kpi-card {{
                background: {THEME['bg_card']};
                padding: 20px 25px;
                border-radius: 12px;
                border: 1px solid {THEME['border']};
                transition: transform 0.2s, box-shadow 0.2s;
            }}
            .kpi-card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            }}
            
            .chart-card {{
                background: {THEME['bg_card']};
                padding: 20px;
                border-radius: 12px;
                border: 1px solid {THEME['border']};
            }}
            
            /* Tab styling */
            .tab {{
                background: {THEME['bg_card']} !important;
                color: {THEME['text_secondary']} !important;
                border: none !important;
                padding: 15px 25px !important;
                font-weight: 500 !important;
            }}
            .tab--selected {{
                background: {THEME['bg_secondary']} !important;
                color: {THEME['accent_blue']} !important;
                border-bottom: 2px solid {THEME['accent_blue']} !important;
            }}
            
            /* Dropdown styling */
            .Select-control {{
                background-color: {THEME['bg_secondary']} !important;
                border-color: {THEME['border']} !important;
            }}
            .Select-value-label {{
                color: {THEME['text_primary']} !important;
            }}
            .Select-menu-outer {{
                background-color: {THEME['bg_card']} !important;
                border-color: {THEME['border']} !important;
            }}
            
            /* Slider styling */
            .rc-slider-track {{
                background-color: {THEME['accent_blue']} !important;
            }}
            .rc-slider-handle {{
                border-color: {THEME['accent_blue']} !important;
                background-color: {THEME['bg_card']} !important;
            }}
            .rc-slider-rail {{
                background-color: {THEME['bg_secondary']} !important;
            }}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
    </body>
</html>
'''

# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("SA ENERGY ANALYTICS DASHBOARD")
    print("="*60)
    print(f"Open: http://127.0.0.1:8050")
    print("="*60 + "\n")
    app.run(debug=True, port=8050)
