import re
import joblib
import streamlit as st


# --------------------------------------------------
# Load model
# --------------------------------------------------

MODEL_PATH = "models/svm_sentiment_model.joblib"

model_package = joblib.load(MODEL_PATH)

tfidf_vectorizer = model_package["tfidf"]
svm_model = model_package["model"]


# --------------------------------------------------
# Text cleaning
# Same preprocessing used during training
# --------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z0-9\s!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Google Play Store Sentiment",
    page_icon="📱",
    layout="wide"
)


st.title("📱 Google Play Store Sentiment Analysis")
st.write(
    "Enter a review to predict its sentiment "
    "using TF-IDF and Linear SVM."
)


# --------------------------------------------------
# Sentiment Prediction
# --------------------------------------------------

st.header("🔮 Sentiment Prediction")

review = st.text_area(
    "Enter your review:",
    placeholder="Example: This app is really useful and easy to use!"
)


if st.button("Predict Sentiment"):

    if not review.strip():
        st.warning("Please enter a review.")
    else:

        cleaned_review = clean_text(review)

        review_tfidf = tfidf_vectorizer.transform(
            [cleaned_review]
        )

        prediction = svm_model.predict(
            review_tfidf
        )[0]

        st.subheader("Prediction")

        if prediction == "Positive":
            st.success("😊 Positive")

        elif prediction == "Negative":
            st.error("😞 Negative")

        else:
            st.warning("😐 Neutral")


# --------------------------------------------------
# Model Insights
# --------------------------------------------------

st.header("📊 Model Performance")

st.write(
    "The sentiment classifier was trained using TF-IDF features "
    "and a Linear Support Vector Machine (SVM)."
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Model", "Linear SVM")
col2.metric("Features", "TF-IDF")
col3.metric("N-grams", "1–2")
col4.metric("Max Features", "30,000")

st.info(
    "Performance metrics such as Accuracy, Precision, Recall, "
    "and F1-score can be displayed here using the evaluation "
    "results from the training notebook."
)