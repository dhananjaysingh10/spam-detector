# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # 👈 NEW
from pydantic import BaseModel
import joblib

app = FastAPI()

# 👇 Allow requests from ANY origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("app/spam_model.pkl")

class MessageInput(BaseModel):
    message: str

@app.get("/")
def root():
    return {"msg": "Spam classifier ready!"}

@app.post("/predict")
def predict(data: MessageInput):
    proba = model.predict_proba([data.message])[0]
    pred = int(proba[1] > 0.5)
    return {
        "prediction": "Spam" if pred else "Not Spam",
        "confidence": f"{round(proba[pred] * 100, 2)}%"
    }
