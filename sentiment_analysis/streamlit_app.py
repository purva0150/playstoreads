import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from lime.lime_text import LimeTextExplainer


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "models/svm_sentiment_model.joblib"

model_package = joblib.load(MODEL_PATH)

tfidf_vectorizer = model_package["tfidf"]
svm_model = model_package["model"]


# ============================================================
# TEXT CLEANING
# Same preprocessing used during training
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'http\S+|www\S+',
        '',
        text
    )

    text = re.sub(
        r'<.*?>',
        '',
        text
    )

    text = re.sub(
        r'[^a-z0-9\s!?]',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text


# ============================================================
# LIME PREDICTION FUNCTION
# ============================================================

def lime_predict(texts):

    cleaned_texts = [
        clean_text(text)
        for text in texts
    ]

    tfidf_features = tfidf_vectorizer.transform(
        cleaned_texts
    )

    decision_scores = svm_model.decision_function(
        tfidf_features
    )

    # Convert SVM decision scores into
    # probability-like values for LIME
    if decision_scores.ndim == 1:

        decision_scores = np.column_stack(
            [
                -decision_scores,
                decision_scores
            ]
        )

    exp_scores = np.exp(
        decision_scores
        - np.max(
            decision_scores,
            axis=1,
            keepdims=True
        )
    )

    probabilities = (
        exp_scores
        / exp_scores.sum(
            axis=1,
            keepdims=True
        )
    )

    return probabilities


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Google Play Store Sentiment Analysis",
    page_icon="📱",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "📱 Google Play Store Sentiment Analysis"
)

st.write(
    "Enter a review to predict its sentiment "
    "using TF-IDF and Linear SVM."
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("📊 Model Performance")

st.write(
    "The sentiment classifier was trained using TF-IDF features "
    "and a Linear Support Vector Machine (SVM)."
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Model",
    "Linear SVM"
)

col2.metric(
    "Features",
    "TF-IDF"
)

col3.metric(
    "N-grams",
    "1–2"
)

col4.metric(
    "Max Features",
    "30,000"
)


# ============================================================
# SENTIMENT PREDICTION
# ============================================================

st.header("🔮 Sentiment Prediction")

review = st.text_area(
    "Enter your review:",
    placeholder=(
        "Example: This app is really useful and easy to use!"
    )
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict Sentiment"):

    if not review.strip():

        st.warning(
            "Please enter a review."
        )

    else:

        # ----------------------------------------------------
        # Clean review
        # ----------------------------------------------------

        cleaned_review = clean_text(
            review
        )

        # ----------------------------------------------------
        # TF-IDF transformation
        # ----------------------------------------------------

        review_tfidf = (
            tfidf_vectorizer.transform(
                [cleaned_review]
            )
        )

        # ----------------------------------------------------
        # Linear SVM prediction
        # ----------------------------------------------------

        prediction = svm_model.predict(
            review_tfidf
        )[0]

        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        st.subheader("Prediction")

        if prediction == "Positive":

            st.success(
                "😊 Positive"
            )

        elif prediction == "Negative":

            st.error(
                "😞 Negative"
            )

        else:

            st.warning(
                "😐 Neutral"
            )


        # ====================================================
        # LIME EXPLANATION
        # ====================================================

        st.subheader(
            "🔍 LIME Explanation"
        )

        st.write(
            "LIME shows which words contributed to this "
            "individual prediction."
        )

        # ----------------------------------------------------
        # Create LIME explainer
        # ----------------------------------------------------

        explainer = LimeTextExplainer(
            class_names=svm_model.classes_
        )

        # ----------------------------------------------------
        # Generate explanation
        # ----------------------------------------------------

        explanation = explainer.explain_instance(
            review,
            lime_predict,
            num_features=10
        )

        # ----------------------------------------------------
        # Get word contributions
        # ----------------------------------------------------

        lime_values = explanation.as_list()

        lime_df = pd.DataFrame(
            lime_values,
            columns=[
                "Word",
                "Contribution"
            ]
        )

        # Sort contributions
        lime_df = lime_df.sort_values(
            "Contribution"
        )

        # ----------------------------------------------------
        # Create colors
        # Green = positive contribution
        # Red = negative contribution
        # ----------------------------------------------------

        bar_colors = [
            "#2ca02c"
            if value > 0
            else "#d62728"
            for value in lime_df[
                "Contribution"
            ]
        ]

        # ----------------------------------------------------
        # Create LIME bar chart
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.barh(
            lime_df["Word"],
            lime_df["Contribution"],
            color=bar_colors
        )

        ax.axvline(
            0,
            color="black",
            linewidth=1
        )

        ax.set_xlabel(
            "Contribution to Predicted Class"
        )

        ax.set_ylabel(
            "Words / Features"
        )

        ax.set_title(
            f"LIME Local Explanation - "
            f"{prediction} Review"
        )

        plt.tight_layout()

        # ----------------------------------------------------
        # Display chart
        # ----------------------------------------------------

        st.pyplot(fig)

        # ----------------------------------------------------
        # Display contribution values
        # ----------------------------------------------------

        st.write(
            "### Word Contributions"
        )

        for word, contribution in lime_values:

            if contribution > 0:

                st.write(
                    f"🟢 **{word}** → "
                    f"+{contribution:.4f}"
                )

            else:

                st.write(
                    f"🔴 **{word}** → "
                    f"{contribution:.4f}"
                )