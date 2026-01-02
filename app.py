import os
import pickle
import numpy as np

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


# =========================
# Path Setup
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "svm_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# =========================
# Inisialisasi FastAPI
# =========================
app = FastAPI(
    title="Spam Email Classification API",
    description="API klasifikasi email spam / ham menggunakan SVM",
    version="1.0.0"
)


# =========================
# Templates
# =========================
templates = Jinja2Templates(
    directory=os.path.join(BASE_DIR, "templates")
)


# =========================
# Routes
# =========================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# =========================
# Schemas
# =========================
class EmailRequest(BaseModel):
    email: str


class EmailResponse(BaseModel):
    prediction: str
    confidence: float


# =========================
# Prediction Endpoint
# =========================
@app.post("/predict", response_model=EmailResponse)
async def predict_email(request: EmailRequest):
    X = [request.email]

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X).max()

    return {
        "prediction": str(prediction),
        "confidence": round(float(probability), 4)
    }
