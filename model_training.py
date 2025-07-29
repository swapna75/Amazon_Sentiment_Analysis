# model_training.py
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

nltk.download('stopwords')

# Load and preprocess
df = pd.read_csv("Reviews.csv")
df = df[df['Score'] != 3]
df['sentiment'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)
df['combined'] = df['Summary'].fillna('') + " " + df['Text'].fillna('')

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['combined'])
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=6000),
    "NaiveBayes": MultinomialNB(),
    "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Train and save each model
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    joblib.dump(model, f"{name}_model.pkl")

# Save vectorizer once
joblib.dump(vectorizer, "vectorizer.pkl")
