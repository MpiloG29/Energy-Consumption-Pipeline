# Energy-Consumption-Pipeline

Build a scalable data pipeline that ingests, processes, and analyzes household energy-consumption data, then serves dashboards and prediction endpoints.

## Project structure

```text
.
├── Procfile                           # PaaS web entrypoint (Streamlit)
├── requirements.txt                   # Root dependency file for deployment tools
├── runtime.txt                        # Python runtime pin for platforms like Render
└── energy_pipeline_project/
    ├── raw_data/
    ├── transformed_data/
    ├── database/
    ├── dashboard_data/
    ├── dashboards/
    │   ├── streamlit_app.py           # Main web dashboard
    │   └── dash_app.py
    ├── ml_models/
    │   ├── api_server.py              # HTTP prediction API
    │   ├── train_model.py
    │   └── *.pkl
    ├── requirements.txt
    ├── run_all.ps1                    # Windows pipeline runner
    └── run_all.sh                     # Linux/macOS pipeline runner