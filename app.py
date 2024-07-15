from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from nltk.stem import PorterStemmer


class PredictRequest(BaseModel):
    text: str

app = FastAPI()

# Load your model and vectorizer using joblib
model = joblib.load("model_sentimen_analis5.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.get("/")
def read_root():

    return {"message": "Sentiment Analysis API is running!"}



@app.post("/predict")
def predict(request: PredictRequest):
    
    # Preprocessing teks baru (stemming)
    def stem_text(text):
        stemmer = PorterStemmer()
        return ' '.join([stemmer.stem(word) for word in text.split()])
    
    text = request.text
    new_text = stem_text(text)
    text_vector = vectorizer.transform([new_text])
    prediction = model.predict(text_vector)
    if prediction == 1:
        sentiment = "Positif"
    elif prediction == -1:
        sentiment = "Negatif"
    else:
        sentiment = "Netral"
        
    # return {"prediction": prediction}
    return {"prediction": sentiment}
   
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)