# ============================================================================
# CONSUMER-FRIENDLY ENERGY PREDICTOR API
# South Africa household energy consumption prediction
# ============================================================================
# Simple, human-understandable inputs → Interpretable predictions + explanations
# ============================================================================

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import joblib
import traceback
import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import get_feature_descriptions

# ============================================================================
# LOAD MODELS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_models():
    """Load all three trained models."""
    try:
        gb_model = joblib.load(os.path.join(BASE_DIR, "energy_model_gradient_boosting.pkl"))
        rf_model = joblib.load(os.path.join(BASE_DIR, "energy_model_random_forest.pkl"))
        lr_model = joblib.load(os.path.join(BASE_DIR, "energy_model_linear.pkl"))
        return gb_model, rf_model, lr_model
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise

try:
    gb_model, rf_model, lr_model = load_models()
    print("✅ Models loaded successfully")
except Exception:
    gb_model = rf_model = lr_model = None
    print("⚠️ Models not yet trained. Run train_model.py first.")

# ============================================================================
# PREDICTION LOGIC
# ============================================================================

def construct_feature_vector(hour, day_type, season, load_shedding_stage, 
                            has_backup_power, household_id="House_1"):
    """
    Construct feature vector from simple human inputs.
    
    Inputs:
        hour (int): 0-23
        day_type (str): "weekday" or "weekend"
        season (str): "Winter", "Summer", "Spring", "Autumn"
        load_shedding_stage (int): 0-6
        has_backup_power (bool): True/False
        household_id (str): "House_1" to "House_5" (default: "House_1")
    
    Returns:
        feature_array: numpy array matching model input requirements
    """
    
    # Convert day_type to day_of_week (use Monday for weekday, Saturday for weekend)
    day_of_week = 0 if day_type.lower() == "weekday" else 5
    is_weekend = 1 if day_type.lower() != "weekday" else 0
    
    # Determine month from season (use mid-month)
    season_to_month = {
        "Winter": 7,     # July (mid-winter)
        "Summer": 1,     # January (mid-summer)
        "Spring": 10,    # October (spring)
        "Autumn": 4      # April (autumn)
    }
    month = season_to_month.get(season, 1)
    
    # Cyclical encodings
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    # Season one-hot encoding
    season_autumn = 1 if season == "Autumn" else 0
    season_spring = 1 if season == "Spring" else 0
    season_summer = 1 if season == "Summer" else 0
    season_winter = 1 if season == "Winter" else 0
    
    # Load shedding features
    is_load_shedding = 1 if load_shedding_stage > 0 else 0
    load_shedding_normalized = load_shedding_stage / 6
    
    # Backup power
    backup_power = 1 if has_backup_power else 0
    
    # Household one-hot encoding
    household_map = {
        "House_1": (1, 0, 0, 0, 0),
        "House_2": (0, 1, 0, 0, 0),
        "House_3": (0, 0, 1, 0, 0),
        "House_4": (0, 0, 0, 1, 0),
        "House_5": (0, 0, 0, 0, 1),
    }
    household_encoded = household_map.get(household_id, (1, 0, 0, 0, 0))
    
    # Construct feature vector in exact order expected by models
    features = [
        hour, day_of_week, is_weekend, month,
        hour_sin, hour_cos, dow_sin, dow_cos,
        season_autumn, season_spring, season_summer, season_winter,
        is_load_shedding, 0, load_shedding_normalized,  # post_load_shedding = 0 (assumed)
        backup_power,
        *household_encoded
    ]
    
    return np.array(features).reshape(1, -1)


def categorize_consumption(predicted_kwh, season, hour):
    """
    Categorize consumption as Low / Normal / High.
    Thresholds vary by season and time of day.
    """
    # Base thresholds (adjusted by season and time)
    base_low = 0.7
    base_normal = 1.5
    
    # Season adjustment
    season_mult = 1.4 if season == "Winter" else 1.0
    
    # Time adjustment (evening peaks are higher)
    time_mult = 1.5 if 18 <= hour <= 21 else 1.0
    
    low_threshold = base_low * season_mult * time_mult
    normal_threshold = base_normal * season_mult * time_mult
    
    if predicted_kwh < low_threshold:
        return "Low"
    elif predicted_kwh < normal_threshold:
        return "Normal"
    else:
        return "High"


def get_top_drivers(rf_model, feature_vector, top_n=3):
    """
    Extract top N feature importance scores for user explanation.
    """
    if rf_model is None:
        return []
    
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
    top_indices = np.argsort(importances)[::-1][:top_n]
    
    drivers = []
    for idx in top_indices:
        if idx < len(feature_names):
            drivers.append({
                "factor": feature_names[idx],
                "importance": float(importances[idx])
            })
    
    return drivers


def generate_explanation(predicted_kwh, category, hour, day_type, season, 
                        load_shedding_stage, has_backup_power, drivers):
    """
    Generate human-friendly explanation of prediction.
    """
    reasons = []
    
    # Time-based reason
    if 6 <= hour < 9:
        reasons.append("morning peak time")
    elif 18 <= hour <= 21:
        reasons.append("evening peak time")
    elif 0 <= hour < 5:
        reasons.append("night (low consumption)")
    
    # Season reason
    if season == "Winter":
        reasons.append("winter season (higher heating demand)")
    elif season == "Summer":
        reasons.append("summer season (lower heating, more cooling)")
    
    # Load shedding reason
    if load_shedding_stage > 0:
        reasons.append(f"electricity outage (Stage {load_shedding_stage})")
    
    # Backup power reason
    if has_backup_power and load_shedding_stage > 0:
        reasons.append("backup power available during outage")
    
    # Day type reason
    if day_type.lower() == "weekend":
        reasons.append("weekend (different consumption pattern)")
    
    explanation = " + ".join(reasons) if reasons else "standard conditions"
    
    return f"{category} consumption expected due to: {explanation}"


# ============================================================================
# HTTP REQUEST HANDLER
# ============================================================================

class EnergyPredictorHandler(BaseHTTPRequestHandler):
    def _set_headers(self, code=200, content_type="application/json"):
        self.send_response(code)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self._set_headers(200)
    
    def do_POST(self):
        """Handle prediction requests."""
        if self.path != '/predict':
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'Endpoint not found. Use /predict'}).encode())
            return
        
        if gb_model is None or rf_model is None:
            self._set_headers(500)
            self.wfile.write(json.dumps({
                'error': 'Models not loaded. Run train_model.py first.'
            }).encode())
            return
        
        try:
            # Parse request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            request_data = json.loads(body.decode('utf-8'))
            
            # Extract and validate inputs
            hour = int(request_data.get('hour', 12))
            day_type = request_data.get('day_type', 'weekday').lower()
            season = request_data.get('season', 'Summer')
            load_shedding_stage = int(request_data.get('load_shedding_stage', 0))
            has_backup_power = bool(request_data.get('has_backup_power', False))
            household_id = request_data.get('household_id', 'House_1')
            
            # Validate inputs
            if not (0 <= hour <= 23):
                raise ValueError("hour must be 0-23")
            if day_type not in ['weekday', 'weekend']:
                raise ValueError("day_type must be 'weekday' or 'weekend'")
            if season not in ['Winter', 'Summer', 'Spring', 'Autumn']:
                raise ValueError("season must be Winter, Summer, Spring, or Autumn")
            if not (0 <= load_shedding_stage <= 6):
                raise ValueError("load_shedding_stage must be 0-6")
            if household_id not in ['House_1', 'House_2', 'House_3', 'House_4', 'House_5']:
                raise ValueError("household_id must be House_1 to House_5")
            
            # Construct feature vector
            features = construct_feature_vector(
                hour, day_type, season, load_shedding_stage, 
                has_backup_power, household_id
            )
            
            # Make predictions using Gradient Boosting
            predicted_kwh = float(gb_model.predict(features)[0])
            predicted_kwh = max(0, predicted_kwh)  # No negative consumption
            
            # Categorize consumption
            category = categorize_consumption(predicted_kwh, season, hour)
            
            # Get top drivers
            drivers = get_top_drivers(rf_model, features, top_n=3)
            
            # Generate explanation
            explanation = generate_explanation(
                predicted_kwh, category, hour, day_type, season, 
                load_shedding_stage, has_backup_power, drivers
            )
            
            # Build response
            response = {
                "status": "success",
                "prediction": {
                    "expected_consumption_kwh": round(predicted_kwh, 2),
                    "consumption_category": category,
                    "explanation": explanation
                },
                "input_context": {
                    "hour": hour,
                    "day_type": day_type,
                    "season": season,
                    "load_shedding_stage": load_shedding_stage,
                    "has_backup_power": has_backup_power,
                    "household_id": household_id
                },
                "top_drivers": drivers
            }
            
            self._set_headers(200)
            self.wfile.write(json.dumps(response, indent=2).encode())
            
        except json.JSONDecodeError:
            self._set_headers(400)
            self.wfile.write(json.dumps({'error': 'Invalid JSON in request body'}).encode())
        except ValueError as e:
            self._set_headers(400)
            self.wfile.write(json.dumps({'error': str(e)}).encode())
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({'error': f'Server error: {str(e)}'}).encode())
    
    def do_GET(self):
        """Handle GET requests (for documentation / health check)."""
        if self.path == '/':
            self._set_headers(200, "text/html")
            html = """
            <html>
            <head><title>Energy Consumption Predictor</title></head>
            <body style="font-family: Arial; margin: 40px;">
                <h1>🔋 South Africa Energy Consumption Predictor</h1>
                <p>POST to <code>/predict</code> with JSON body:</p>
                <pre style="background: #f0f0f0; padding: 10px; border-radius: 5px;">
{
    "hour": 18,                    // 0-23
    "day_type": "weekday",         // "weekday" or "weekend"
    "season": "Winter",            // "Winter", "Summer", "Spring", "Autumn"
    "load_shedding_stage": 0,      // 0-6
    "has_backup_power": false,     // true/false
    "household_id": "House_1"      // "House_1" to "House_5"
}
                </pre>
                <p><strong>Response:</strong> Predicted consumption (kWh) + category + explanation</p>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        elif self.path == '/health':
            self._set_headers(200)
            self.wfile.write(json.dumps({
                'status': 'healthy',
                'models_loaded': gb_model is not None
            }).encode())
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'Not found'}).encode())
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


# ============================================================================
# START SERVER
# ============================================================================

def run_server(port=5000, host='0.0.0.0'):
    """Start the energy predictor API server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, EnergyPredictorHandler)
    print("=" * 70)
    print(f"🚀 Energy Consumption Predictor API started on port {port}")
    print(f"📡 Access documentation at: http://localhost:{port}/")
    print(f"🔮 POST predictions to: http://localhost:{port}/predict")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped.")
    finally:
        httpd.server_close()


if __name__ == '__main__':
    cli_port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.getenv('PORT', '5000'))
    cli_host = os.getenv('HOST', '0.0.0.0')
    run_server(port=cli_port, host=cli_host)