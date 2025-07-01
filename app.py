from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Define FastAPI app
app = FastAPI()

# Define request schema
class ReviewRequest(BaseModel):
    review: str

# Prediction endpoint
@app.post("/predict")
def predict_sentiment(data: ReviewRequest):
    review_text = data.review
    X = vectorizer.transform([review_text])
    prediction = model.predict(X)[0]
    label = "positive" if prediction == 1 else "negative"
    return {"sentiment": label}

@app.get("/")
def root():
    return {"message": "Sentiment API is running!"}

