# ============================================================================
# SOUTH AFRICA ENERGY CONSUMPTION PIPELINE - COMPLETE ORCHESTRATION
# ============================================================================
# 1. Generate synthetic, realistic SA energy consumption data
# 2. Perform exploratory analysis & validation
# 3. Train three interpretable models (Linear, Random Forest, Gradient Boosting)
# 4. Start consumer-friendly prediction API
# ============================================================================

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -Path $scriptDir

Write-Output "=========================================================================="
Write-Output "SOUTH AFRICA ENERGY CONSUMPTION PIPELINE"
Write-Output "=========================================================================="

# Step 1: Generate synthetic data
Write-Output ""
Write-Output "[1/4] Generating synthetic energy data (12 months, 5 households)..."
try {
    python .\simulate_energy_data.py
    if ($LASTEXITCODE -eq 0) {
        Write-Output "      [OK] Data generation complete"
    } else {
        Write-Output "      [FAIL] Data generation failed"
        exit 1
    }
} catch {
    Write-Output "      [FAIL] Error: $_"
    exit 1
}

# Step 2: Exploratory Analysis
Write-Output ""
Write-Output "[2/4] Running exploratory analysis..."
try {
    python .\exploratory_analysis.py
    if ($LASTEXITCODE -eq 0) {
        Write-Output "      [OK] Analysis complete"
    } else {
        Write-Output "      [FAIL] Analysis failed"
        exit 1
    }
} catch {
    Write-Output "      [FAIL] Error: $_"
    exit 1
}

# Step 3: Train three-model ensemble
Write-Output ""
Write-Output "[3/4] Training three-model ensemble..."
Write-Output "      * Linear Regression (baseline)"
Write-Output "      * Random Forest (insights)"
Write-Output "      * Gradient Boosting (accuracy)"
try {
    python .\ml_models\train_model.py
    if ($LASTEXITCODE -eq 0) {
        Write-Output "      [OK] All models trained and saved"
    } else {
        Write-Output "      [FAIL] Model training failed"
        exit 1
    }
} catch {
    Write-Output "      [FAIL] Error: $_"
    exit 1
}

# Step 4: Start API server (background)
Write-Output ""
Write-Output "[4/4] Starting consumer-friendly prediction API..."
try {
    Start-Process -FilePath "python" -ArgumentList "ml_models\api_server.py" -NoNewWindow
    Write-Output "      [OK] API server started on http://localhost:5000"
} catch {
    Write-Output "      [WARN] Could not start API server: $_"
}

Write-Output "`n=========================================================================="
Write-Output "✅ PIPELINE COMPLETE"
Write-Output "=========================================================================="
Write-Output "`nNext steps:"
Write-Output "  • View dashboard: python dashboards\dash_app.py"
Write-Output "  • Test API: curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' -d '{\"hour\": 18, \"day_type\": \"weekday\", \"season\": \"Winter\", \"load_shedding_stage\": 0, \"has_backup_power\": false}'"
Write-Output "  • Check data: ./raw_data/energy_data.csv"
Write-Output "  • View plots: ./dashboard_data/plots/"
Write-Output "=========================================================================="
