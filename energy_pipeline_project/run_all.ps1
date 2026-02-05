# run_all.ps1 - Run core project scripts (Windows PowerShell)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -Path $scriptDir

Write-Output "1/4 Generating simulated data..."
python .\simulate_energy_data.py

Write-Output "2/4 Running Spark ETL (requires pyspark)..."
try {
	python -c "import pyspark" 2>$null
	if ($LASTEXITCODE -eq 0) {
		python .\spark_etl.py
	} else {
		Write-Output "Skipping Spark ETL: pyspark not available in the environment."
	}
} catch {
	Write-Output "Skipping Spark ETL: pyspark import raised an error."
}

Write-Output "3/4 Training model..."
python .\ml_models\train_model.py

Write-Output "4/4 Starting API server (background)..."
Start-Process -FilePath "python" -ArgumentList "ml_models\api_server.py"

Write-Output "Run complete. API server started in background."
