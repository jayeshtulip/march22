from fastapi import FastAPI
import joblib
from pydantic import BaseModel

# Load trained model
model = joblib.load("house_price_model.pkl")

app = FastAPI()

# Define request format
class HouseFeatures(BaseModel):
    square_feet: float

# Prediction endpoint
@app.post("/predict/")
def predict_price(features: HouseFeatures):
    prediction = model.predict([[features.square_feet]])[0]
    return {"predicted_price": round(prediction, 2)}

# Health check
@app.get("/")
def home():
    return {"message": "House Price Prediction API is running!"}
