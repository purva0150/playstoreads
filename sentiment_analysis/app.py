import re
import joblib

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="Google Play Store Sentiment Analysis API",
    description="Sentiment prediction using TF-IDF and Linear SVM",
    version="1.0.0"
)


# Load the trained TF-IDF vectorizer and Linear SVM
MODEL_PATH = "models/svm_sentiment_model.joblib"

model_package = joblib.load(MODEL_PATH)

tfidf_vectorizer = model_package["tfidf"]
svm_model = model_package["model"]


# Same text cleaning used during training
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z0-9\s!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Request format
class ReviewRequest(BaseModel):
    review_text: str


@app.get("/")
def home():
    return {
        "message": "Google Play Store Sentiment Analysis API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: ReviewRequest):

    # 1. Clean the review
    cleaned_review = clean_text(request.review_text)

    # 2. Apply the SAME fitted TF-IDF vectorizer
    review_tfidf = tfidf_vectorizer.transform(
        [cleaned_review]
    )

    # 3. Predict using the trained Linear SVM
    prediction = svm_model.predict(
        review_tfidf
    )[0]

    return {
        "review": request.review_text,
        "prediction": prediction
    }