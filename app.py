# app.py
import streamlit as st
import joblib
import os

# Load vectorizer
vectorizer = joblib.load("vectorizer.pkl")

# Available models
models = {
    "Logistic Regression": "models/LogisticRegression_model.pkl",
    "Naïve Bayes": "models/NaiveBayes_model.pkl",
    "Decision Tree": "models/DecisionTree_model.pkl",
    "Random Forest": "models/RandomForest_model.pkl",
    "XGBoost": "models/XGBoost_model.pkl"
}

st.title("🛍️ Amazon Review Sentiment Analyzer")
st.markdown("Choose your model and input your review to analyze sentiment.")

# Select model
model_choice = st.selectbox("Choose ML model", list(models.keys()))

# Text input
review = st.text_area("Enter your product review")

# Predict
if st.button("Analyze Sentiment"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        # Load selected model
        model_path = models[model_choice]
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            vector = vectorizer.transform([review])
            prediction = model.predict(vector)[0]
            sentiment = "👍 Positive" if prediction == 1 else "👎 Negative"
            st.success(f"{model_choice} predicts: **{sentiment}**")
        else:
            st.error(f"Model file not found: {model_path}")
