#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================================="
echo "SOUTH AFRICA ENERGY CONSUMPTION PIPELINE"
echo "=========================================================================="

echo
echo "[1/4] Generating synthetic energy data (12 months, 5 households)..."
python simulate_energy_data.py
echo "      [OK] Data generation complete"

echo
echo "[2/4] Running exploratory analysis..."
python exploratory_analysis.py
echo "      [OK] Analysis complete"

echo
echo "[3/4] Training three-model ensemble..."
echo "      * Linear Regression (baseline)"
echo "      * Random Forest (insights)"
echo "      * Gradient Boosting (accuracy)"
python ml_models/train_model.py
echo "      [OK] All models trained and saved"

echo
echo "[4/4] Starting consumer-friendly prediction API..."
PORT="${PORT:-5000}"
python ml_models/api_server.py "$PORT"