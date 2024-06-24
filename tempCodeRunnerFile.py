from flask import Flask, request, jsonify
import joblib
from gevent.pywsgi import WSGIServer
app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('model_sentimen_analis2.pkl')
vectorizer = joblib.load('vectorizer2.pkl')

@app.route('/')
def home():
    return "Sentiment Analysis API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    X = vectorizer.transform([text])
    prediction = model.predict(X)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)    
