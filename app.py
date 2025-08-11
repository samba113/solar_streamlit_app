# main.py
import os
import re
import joblib
import gdown
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Drive direct download links
MODEL_URL = "https://drive.google.com/uc?id=1YTbH3EvF_O0z5ncilUSHtyhSkJdit5Rn"
SCALER_URL = "https://drive.google.com/uc?id=1IFwtBf7fkf0Gr9ZUzLFdYQuMQS4gMI1h"

MODEL_PATH = "solar_power_model.joblib"
SCALER_PATH = "scaler.joblib"

# Download model and scaler if not exist
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

if not os.path.exists(SCALER_PATH):
    gdown.download(SCALER_URL, SCALER_PATH, quiet=False)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

class Payload(BaseModel):
    message: str

@app.post("/predict")
async def predict(payload: Payload):
    text = payload.message.lower()
    try:
        temp = float(re.search(r"(\d+(?:\.\d+)?)\s*°?c", text).group(1))
        hum = float(re.search(r"(\d+(?:\.\d+)?)\s*% humidity", text).group(1))
        pres = float(re.search(r"(\d+(?:\.\d+)?)\s*hpa", text).group(1))
        spd = float(re.search(r"(\d+(?:\.\d+)?)\s*km/h", text).group(1))
    except:
        return {"error": "⚠️ Please follow format: e.g. 32°C, 55% humidity, 1012 hPa, 10 km/h speed"}

    arr = np.array([[temp, hum, pres, spd]])
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled)[0]
    pred = max(0, float(pred))
    return {"prediction": round(pred, 2)}

# ✅ Add this route for health check
@app.get("/")
def read_root():
    return {"message": "Solar Power Prediction API is running"}
