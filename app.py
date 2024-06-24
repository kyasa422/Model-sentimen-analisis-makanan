# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Any
# import joblib
# import numpy as np

# app = FastAPI()

# # Load model and vectorizer
# model = joblib.load('model_sentimen_analis2.pkl')
# vectorizer = joblib.load('vectorizer2.pkl')

# class PredictRequest(BaseModel):
#     text: str

# @app.get("/")
# def home() -> Any:
#     return {"message": "Sentiment Analysis API is running!"}

# @app.post("/predict")
# # def predict(data: TextData) -> Any:
# #     if not data.text:
# #         raise HTTPException(status_code=400, detail="No text provided")

# #     text = data.text
# #     X = vectorizer.transform([text])
# #     prediction = model.predict(X)
# # def predict(data: dict):
# #     text = data['text']
# #     text_vector = vectorizer.transform([text])
# #     prediction = model.predict(text_vector)
#     # return {"prediction": prediction[0]}
#     # Convert numpy.int64 to int
    
# def predict(request: PredictRequest):
#     print(request.text)
#     text = request.text
#     text_vector = vectorizer.transform([text])
#     prediction = model.predict(text_vector)
#     sentiment = "positive" if prediction[0] == 1 else "negative"

#     return {"prediction": sentiment}

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app)
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import requests

BASE_URL = "https://your-app-name.up.railway.app"

class PredictRequest(BaseModel):
    text: str

app = FastAPI()

# Load your model and vectorizer using joblib
model = joblib.load("model_sentimen_analis2.pkl")
vectorizer = joblib.load("vectorizer2.pkl")

@app.get("/")
def read_root():

    return {"message": "Sentiment Analysis API is running!"}


@app.route("/predict", methods=['POST'])
@app.post("/predict" )
@app.post("/predict")
def predict(request: PredictRequest):
    text = request.text
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    sentiment = "positive" if prediction[0] == 1 else "negative"
    return {"prediction": sentiment}
    

def send_get_request():
    response = requests.get(f"{BASE_URL}/")
    if response.status_code == 200:
        print("GET request successful!")
        print("Response:", response.json())
    else:
        print(f"GET request failed with status code {response.status_code}")

# Function to send a POST request
def send_post_request(text):
    url = f"{BASE_URL}/predict/"
    headers = {"Content-Type": "application/json"}
    data = {"text": text}
    
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        print("POST request successful!")
        print("Response:", response.json())
    else:
        print(f"POST request failed with status code {response.status_code}")
        print("Response:", response.json())


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    send_get_request()  # Test the GET request
    send_post_request("makanan ini ngga enak")  # Test the POST request
