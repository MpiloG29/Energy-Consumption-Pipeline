# Energy-Consumption-Pipeline

In today’s data-driven world, energy consumption patterns hold critical importance for both sustainability and operational efficiency. This project focuses on building a scalable data pipeline that seamlessly ingests, processes, and analyzes real-world energy usage data. By integrating advanced data engineering techniques with predictive modeling, the pipeline not only ensures reliable data flow but also transforms raw information into actionable insights. The deployment of interactive dashboards further empowers stakeholders to monitor consumption trends, optimize resource allocation, and make informed decisions that drive energy efficiency and long-term sustainability.

#  Energy Consumption Pipeline

A practical end-to-end data project for **South African household energy analytics**.

This repository covers the full workflow:
- synthetic data generation,
- exploratory analysis,
- feature engineering,
- ML model training,
- a prediction API,
- and an interactive Streamlit dashboard.

## 🌍 Live Dashboard

 **Deployed App:** https://energy-consumption-dashboard-eo9o.onrender.com/

---

##  Project Highlights

-  Rich analytics dashboard with consumption, outage, and cost views
-  Multiple ML models (Linear Regression, Random Forest, Gradient Boosting)
-  Load-shedding aware features and visual insights
-  Reproducible local pipeline scripts for Windows and Linux/macOS
-  Deployment-friendly config for Render and similar platforms

---

##  Repository Structure

```text
.
├── Procfile
├── render.yaml
├── requirements.txt
├── runtime.txt
└── energy_pipeline_project/
    ├── dashboards/
    │   ├── streamlit_app.py
    │   └── dash_app.py
    ├── ml_models/
    │   ├── api_server.py
    │   ├── feature_engineering.py
    │   ├── train_model.py
    │   └── *.pkl
    ├── raw_data/
    ├── transformed_data/
    ├── dashboard_data/
    ├── database/
    ├── exploratory_analysis.py
    ├── simulate_energy_data.py
    ├── requirements.txt
    ├── run_all.ps1
    └── run_all.sh
```

---

##  Quick Start (Local)

### 1) Clone and enter project

```bash
git clone https://github.com/MpiloG29/Energy-Consumption-Pipeline.git
cd Energy-Consumption-Pipeline
```

### 2) Create environment and install dependencies

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 3) Run the dashboard

```bash
streamlit run energy_pipeline_project/dashboards/streamlit_app.py
```

---

##  Prediction API

Run locally:

```bash
python energy_pipeline_project/ml_models/api_server.py
```

Defaults:
- `HOST=0.0.0.0`
- `PORT=5000`

You can also set `HOST` / `PORT` environment variables for cloud deployment.

---

##  Run Full Pipeline

### Windows

```powershell
cd energy_pipeline_project
./run_all.ps1
```

### Linux/macOS

```bash
cd energy_pipeline_project
./run_all.sh
```

---

##  Deployment Notes (Render)

This repo includes deployment files at the root:
- `render.yaml` (Blueprint deployment)
- `Procfile` (web process command)
- `runtime.txt` (Python version)
- root `requirements.txt` (lightweight web runtime dependencies)

If creating a service manually, use:
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `streamlit run energy_pipeline_project/dashboards/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`

---

##  Tech Stack

- Python
- Pandas, NumPy, SciPy
- scikit-learn
- Plotly
- Streamlit

---
<img width="590" height="450" alt="newplot" src="https://github.com/user-attachments/assets/721607fb-1718-4b8c-ab03-9a237a9445d8" />
<img width="1196" height="450" alt="newplot (2)" src="https://github.com/user-attachments/assets/de2dfa8a-9c2f-4988-aa7a-ca9122d87514" />
<img width="590" height="450" alt="newplot (3)" src="https://github.com/user-attachments/assets/44efbd32-bfc3-4e0c-849a-fcce9345e0d8" />
<img width="590" height="450" alt="newplot (4)" src="https://github.com/user-attachments/assets/ab46342d-206b-4e74-9fcb-f67f909b80fa" />
<img width="590" height="450" alt="newplot (5)" src="https://github.com/user-attachments/assets/92877426-9de2-4f67-a142-e7c4d6233237" />
<img width="590" height="450" alt="newplot (6)" src="https://github.com/user-attachments/assets/9486f025-572d-41f3-ba8c-8acb565d7e01" />

---

## 📄 License

This project is licensed under the terms of the `LICENSE` file in this repository.
