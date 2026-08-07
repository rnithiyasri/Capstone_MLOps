from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="Breast Cancer Prediction API")

model = joblib.load("models/best_model.pkl")

@app.get("/")
def home():
    return {"message": "Breast Cancer Prediction API is running"}

@app.post("/predict")
def predict(features: list[float]):
    prediction = model.predict(np.array(features).reshape(1, -1))
    return {"prediction": int(prediction[0])}