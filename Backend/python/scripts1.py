import numpy as np
from fastapi import FastAPI
import joblib
from pydantic import BaseModel

model=joblib.load('python\\xgb.pkl')
app = FastAPI()

class SimilarityInput(BaseModel):
    similarity: float

@app.post("/predict")
async def predict(input:SimilarityInput):
    sim=input.similarity
    x=np.array([[sim]])
    prediction = model.predict(x)
    return {"prediction": float(prediction[0])}