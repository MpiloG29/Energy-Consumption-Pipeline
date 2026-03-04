# Energy-Consumption-Pipeline

Build a scalable data pipeline that ingests, processes, and analyzes household energy-consumption data, then serves dashboards and prediction endpoints.
<img width="1590" height="704" alt="Screenshot (51)" src="https://github.com/user-attachments/assets/6245160f-e2b2-4ca4-ad20-f6ac5293148b" />
<img width="1582" height="717" alt="Screenshot (52)" src="https://github.com/user-attachments/assets/7068a7c2-2194-479f-bf08-51468ee3edd6" />
<img width="1589" height="709" alt="Screenshot (53)" src="https://github.com/user-attachments/assets/0eab6284-73c5-469d-9063-55d9ca719451" />

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
