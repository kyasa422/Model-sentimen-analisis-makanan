from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
import joblib
import numpy as np

app = FastAPI()

# Load model and vectorizer
model = joblib.load('model_sentimen_analis2.pkl')
vectorizer = joblib.load('vectorizer2.pkl')

class TextData(BaseModel):
    text: str

@app.get("/")
def home() -> Any:
    return {"message": "Sentiment Analysis API is running!"}

@app.post("/predict")
def predict(data: TextData) -> Any:
    if not data.text:
        raise HTTPException(status_code=400, detail="No text provided")

    text = data.text
    X = vectorizer.transform([text])
    prediction = model.predict(X)
    # Convert numpy.int64 to int
    sentiment = "positive" if prediction[0] == 1 else "negative"

    return {"prediction": sentiment}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)
