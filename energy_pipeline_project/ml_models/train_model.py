# train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("raw_data/energy_data.csv")
df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
df["day"] = pd.to_datetime(df["timestamp"]).dt.day

X = df[["hour", "day"]]
y = df["energy_consumption"]

model = LinearRegression()
model.fit(X, y)
joblib.dump(model, "ml_models/energy_predictor.pkl")
