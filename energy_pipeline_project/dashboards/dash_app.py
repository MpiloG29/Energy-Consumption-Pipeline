import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os

# MUTED COLOR SCHEME
COLORS = {
    'background': '#f8f9fa',
    'card_bg': '#ffffff',
    'text': '#495057',
    'text_light': '#6c757d',
    'primary': '#4a6fa5',
    'secondary': '#6c757d',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'accent1': '#8e9aaf',
    'accent2': '#c9ada7',
    'grid': '#e9ecef',
    'border': '#dee2e6',
}

# Use correct paths - go up one directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "raw_data", "energy_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "energy_predictor.pkl")

print(f"Looking for data at: {DATA_PATH}")
print(f"Looking for model at: {MODEL_PATH}")

# Load data
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"✅ Loaded {len(df)} records")
else:
    print(f"❌ Data file not found: {DATA_PATH}")
    # Create sample data for demo
    print("Creating sample data for demo...")
    dates = pd.date_range("2025-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        'timestamp': dates,
        'energy_consumption': 2 + np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 0.2, len(dates))
    })

# Create aggregated data
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["date"] = df["timestamp"].dt.date
df["day_name"] = df["timestamp"].dt.day_name()

hourly_agg = df.groupby(["hour", "day_name"]).agg({
    "energy_consumption": "mean"
}).reset_index()

daily_agg = df.groupby("date").agg({
    "energy_consumption": "sum"
}).reset_index()

# Initialize Dash app with muted theme
app = dash.Dash(__name__, title="Energy Consumption Dashboard")

app.layout = html.Div([
    html.H1("Energy Consumption Dashboard", style={
        'textAlign': 'center', 
        'color': COLORS['text'],
        'marginBottom': '10px',
        'fontWeight': '600'
    }),
    
    html.P("Analytics and monitoring of energy usage patterns", style={
        'textAlign': 'center',
        'color': COLORS['text_light'],
        'marginBottom': '30px'
    }),
    
    html.Div([
        html.Div([
            html.Div([
                html.H3("Total Consumption", className="stat-title"),
                html.H2(f"{df['energy_consumption'].sum():,.2f} kWh", style={
                    'color': COLORS['primary'],
                    'fontSize': '1.8rem',
                    'margin': '0',
                    'fontWeight': '600'
                })
            ], className="stat-card"),
            
            html.Div([
                html.H3("Average Hourly", className="stat-title"),
                html.H2(f"{df['energy_consumption'].mean():.2f} kWh", style={
                    'color': COLORS['secondary'],
                    'fontSize': '1.8rem',
                    'margin': '0',
                    'fontWeight': '600'
                })
            ], className="stat-card"),
            
            html.Div([
                html.H3("Peak Consumption", className="stat-title"),
                html.H2(f"{df['energy_consumption'].max():.2f} kWh", style={
                    'color': COLORS['danger'],
                    'fontSize': '1.8rem',
                    'margin': '0',
                    'fontWeight': '600'
                })
            ], className="stat-card"),
            
            html.Div([
                html.H3("Lowest Consumption", className="stat-title"),
                html.H2(f"{df['energy_consumption'].min():.2f} kWh", style={
                    'color': COLORS['success'],
                    'fontSize': '1.8rem',
                    'margin': '0',
                    'fontWeight': '600'
                })
            ], className="stat-card")
        ], className="stats-container"),
    ]),
    
    html.Div([
        dcc.Graph(id='time-series-chart'),
        dcc.Graph(id='hourly-pattern-chart')
    ], className="charts-row"),
    
    html.Div([
        dcc.Graph(id='heatmap-chart')
    ], className="heatmap-container"),
    
    html.Div([
        html.H2("Consumption Predictor", style={
            'marginTop': '40px',
            'color': COLORS['text'],
            'paddingBottom': '10px',
            'borderBottom': f'1px solid {COLORS["border"]}'
        }),
        html.Div([
            html.Label("Hour of Day (0-23):", style={
                'color': COLORS['text'],
                'display': 'block',
                'marginBottom': '10px',
                'fontWeight': '500'
            }),
            dcc.Slider(
                id='hour-slider',
                min=0,
                max=23,
                value=12,
                marks={i: str(i) for i in range(0, 24, 3)},
                step=1
            ),
            
            html.Label("Day of Week:", style={
                'color': COLORS['text'],
                'display': 'block',
                'marginTop': '20px',
                'marginBottom': '10px',
                'fontWeight': '500'
            }),
            dcc.Dropdown(
                id='day-dropdown',
                options=[
                    {'label': 'Monday', 'value': 1},
                    {'label': 'Tuesday', 'value': 2},
                    {'label': 'Wednesday', 'value': 3},
                    {'label': 'Thursday', 'value': 4},
                    {'label': 'Friday', 'value': 5},
                    {'label': 'Saturday', 'value': 6},
                    {'label': 'Sunday', 'value': 7}
                ],
                value=3,
                clearable=False
            ),
            
            html.Button('Predict', id='predict-button', n_clicks=0, style={
                'backgroundColor': COLORS['primary'],
                'color': 'white',
                'border': 'none',
                'padding': '10px 20px',
                'borderRadius': '4px',
                'marginTop': '20px',
                'cursor': 'pointer',
                'fontWeight': '500'
            }),
            
            html.Div(id='prediction-output', style={
                'marginTop': '20px',
                'padding': '15px',
                'backgroundColor': COLORS['background'],
                'borderRadius': '5px',
                'border': f'1px solid {COLORS["border"]}',
                'borderLeft': f'4px solid {COLORS["primary"]}'
            })
        ], className="prediction-form")
    ])
], className="dashboard-container", style={
    'backgroundColor': COLORS['background'],
    'padding': '20px',
    'minHeight': '100vh'
})

# Callbacks with updated colors
@app.callback(
    Output('time-series-chart', 'figure'),
    Input('hour-slider', 'value')
)
def update_time_series(hour):
    filtered_df = df[df['hour'] == hour] if hour is not None else df
    daily_data = filtered_df.groupby('date')['energy_consumption'].sum().reset_index()
    
    fig = px.line(daily_data, x='date', y='energy_consumption',
                  title=f'Daily Energy Consumption{" (Filtered by Hour)" if hour is not None else ""}',
                  color_discrete_sequence=[COLORS['primary']])
    
    # Apply muted theme
    fig.update_layout(
        plot_bgcolor=COLORS['card_bg'],
        paper_bgcolor=COLORS['card_bg'],
        font_color=COLORS['text'],
        xaxis=dict(
            gridcolor=COLORS['grid'],
            linecolor=COLORS['border'],
            showgrid=True
        ),
        yaxis=dict(
            gridcolor=COLORS['grid'],
            linecolor=COLORS['border'],
            showgrid=True
        ),
        margin=dict(l=40, r=20, t=40, b=40)
    )
    
    return fig

@app.callback(
    Output('hourly-pattern-chart', 'figure'),
    Input('day-dropdown', 'value')
)
def update_hourly_pattern(day):
    filtered_df = df[df['day_of_week'] == day-1] if day is not None else df
    hourly_data = filtered_df.groupby('hour')['energy_consumption'].mean().reset_index()
    
    fig = px.bar(hourly_data, x='hour', y='energy_consumption',
                 title=f'Average Consumption by Hour{" (Filtered by Day)" if day is not None else ""}',
                 color_discrete_sequence=[COLORS['secondary']])
    
    # Apply muted theme
    fig.update_layout(
        plot_bgcolor=COLORS['card_bg'],
        paper_bgcolor=COLORS['card_bg'],
        font_color=COLORS['text'],
        xaxis=dict(
            gridcolor=COLORS['grid'],
            linecolor=COLORS['border'],
            showgrid=True
        ),
        yaxis=dict(
            gridcolor=COLORS['grid'],
            linecolor=COLORS['border'],
            showgrid=True
        ),
        margin=dict(l=40, r=20, t=40, b=40)
    )
    
    return fig

@app.callback(
    Output('heatmap-chart', 'figure'),
    Input('predict-button', 'n_clicks')
)
def update_heatmap(n_clicks):
    heatmap_data = df.groupby(['hour', 'day_name'])['energy_consumption'].mean().reset_index()
    pivot_data = heatmap_data.pivot(index='day_name', columns='hour', values='energy_consumption')
    
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_data = pivot_data.reindex(days_order)
    
    # Use a muted color scale for heatmap
    fig = px.imshow(pivot_data,
                    labels=dict(x="Hour of Day", y="Day of Week", color="Consumption (kWh)"),
                    title="Consumption Heatmap: Day vs Hour",
                    aspect="auto",
                    color_continuous_scale=['#f8f9fa', COLORS['primary'], COLORS['accent1']])
    
    # Apply muted theme
    fig.update_layout(
        plot_bgcolor=COLORS['card_bg'],
        paper_bgcolor=COLORS['card_bg'],
        font_color=COLORS['text'],
        margin=dict(l=40, r=20, t=40, b=40)
    )
    
    return fig

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks'),
     Input('hour-slider', 'value'),
     Input('day-dropdown', 'value')]
)
def update_prediction(n_clicks, hour, day):
    if n_clicks > 0 and hour is not None and day is not None:
        try:
            # Load model and make prediction
            if os.path.exists(MODEL_PATH):
                model = joblib.load(MODEL_PATH)
                # Create DataFrame with feature names to avoid warnings
                pred_df = pd.DataFrame([[hour, day]], columns=['hour', 'day'])
                prediction = model.predict(pred_df)[0]
            else:
                # Simulate prediction if model not found
                prediction = 2.5 + 0.5 * np.sin(2 * np.pi * hour / 24) + 0.2 * np.cos(2 * np.pi * day / 7)
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_name = day_names[day-1]
            
            return [
                html.H4("Prediction Result", style={'color': COLORS['text'], 'marginBottom': '10px'}),
                html.P(f"Time: {hour}:00 on {day_name}", style={'color': COLORS['text_light'], 'margin': '5px 0'}),
                html.P([
                    "Predicted Consumption: ",
                    html.Span(f"{prediction:.2f} kWh", style={
                        'color': COLORS['primary'],
                        'fontSize': '1.2rem',
                        'fontWeight': 'bold',
                        'display': 'inline-block',
                        'marginLeft': '5px'
                    })
                ], style={'color': COLORS['text'], 'margin': '10px 0'})
            ]
        except Exception as e:
            return html.P(f"Error making prediction: {str(e)}", style={'color': COLORS['danger']})
    return html.P("Click 'Predict' to get a consumption prediction", style={'color': COLORS['text_light']})

# Updated CSS styles with muted colors
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f8f9fa;
                color: #495057;
            }
            .dashboard-container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }
            .stats-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            .stat-card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #4a6fa5;
                border: 1px solid #dee2e6;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .stat-title {
                color: #6c757d;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin: 0 0 10px 0;
            }
            .charts-row {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 30px 0;
            }
            .heatmap-container {
                margin: 30px 0;
            }
            .prediction-form {
                background: white;
                padding: 25px;
                border-radius: 8px;
                margin-top: 30px;
                border: 1px solid #dee2e6;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            @media (max-width: 768px) {
                .charts-row {
                    grid-template-columns: 1fr;
                }
                .stats-container {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    print("\n✅ Dashboard starting...")
    print("   Open your browser and go to: http://127.0.0.1:8050")
    print("   Press Ctrl+C to stop the server")
    app.run(debug=True, port=8050)
