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
<img width="1590" height="704" alt="Screenshot (51)" src="https://github.com/user-attachments/assets/4d02af1d-a11c-42ce-a2f5-fdd239e0f5da" />
<img width="1582" height="717" alt="Screenshot (52)" src="https://github.com/user-attachments/assets/e8dcfa3f-02ff-45b0-97f4-96789563050b" />
<img width="1589" height="709" alt="Screenshot (53)" src="https://github.com/user-attachments/assets/294e2f5d-80ab-4872-b270-caaf85ad1398" />
