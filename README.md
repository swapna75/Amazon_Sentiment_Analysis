# Amazon Sentiment Analysis

## Introduction

Customer reviews significantly influence purchasing behavior in e-commerce. Analyzing the sentiment behind these reviews helps businesses gain valuable insights into customer satisfaction. This project leverages **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques to automate sentiment analysis of Amazon product reviews.

---

## Key Concepts

- **Sentiment Analysis**: Determining whether a review expresses positive or negative sentiment.
- **Text Classification**: Categorizing textual data into predefined classes.
- **Natural Language Processing (NLP)**: Enabling computers to interpret and analyze human language.

---

## Problem Statement

Manual analysis of large volumes of product reviews is **inefficient and time-consuming**. Automating sentiment classification improves scalability and provides real-time insights. The goal is to build and compare ML models that can accurately classify reviews as **positive or negative**.

---

## Project Objectives

1. Build sentiment classification models using machine learning.
2. Preprocess and analyze Amazon review data.
3. Compare different ML algorithms:
   - Logistic Regression
   - Naïve Bayes
   - Decision Tree
   - XGBoost
   - Random Forest
4. Visualize review insights using **word clouds** and **charts**.
5. Enhance model performance through feature engineering and preprocessing.

---

## Dataset Overview

- **Source**: [Kaggle - Amazon Reviews Dataset](https://www.kaggle.com/datasets/thanmayemajeti/amazon-reviews)
- **Key Features**:
  - **Summary**: Short review titles (used for classification).
  - **Text**: Full review content (provides additional context).
  - **Sentiment**:
    - `1` → Positive
    - `-1` → Negative

---

## NLP Techniques Applied

| Technique            | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| **Tokenization**     | Splits text into individual words or sentences.                            |
| **Stopword Removal** | Removes common words that don't contribute to sentiment (e.g., "the", "is").|
| **Vectorization**    | Converts text into numerical form (CountVectorizer, TF-IDF).               |
| **Word Clouds**      | Visualizes the most frequent words in reviews.                             |

---

## Data Preprocessing

- **StandardScaler** is applied to standardize features:
  - Mean = 0
  - Standard Deviation = 1
- Benefits:
  - Prevents scale dominance.
  - Accelerates model convergence.
  - Improves algorithm performance (e.g., for SVM, Logistic Regression).

---

##  Machine Learning Models

| Model                | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| **Logistic Regression** | Binary classifier using sigmoid function to predict probability.           |
| **Naïve Bayes**         | Probabilistic model assuming feature independence.                         |
| **Decision Tree**       | Tree-based structure with decision nodes and leaf predictions.             |
| **XGBoost**             | Gradient-boosted decision tree ensemble, efficient and highly accurate.    |
| **Random Forest**       | Ensemble of decision trees to reduce overfitting and improve robustness.   |

---

##  Evaluation Metrics

- **True Positive (TP)**: Correctly predicted positive review.
- **True Negative (TN)**: Correctly predicted negative review.
- **False Positive (FP)**: Incorrectly predicted positive review.
- **False Negative (FN)**: Incorrectly predicted negative review.

---

###  ROC Curve

- **True Positive Rate (TPR)** vs. **False Positive Rate (FPR)**.
- **AUC (Area Under Curve)**:
  - `1.0` → Perfect classification
  - `0.5` → Random prediction
  - `< 0.5` → Worse than random

---

##  Visualizations

- Word clouds highlight frequently used terms in positive/negative reviews.
- Charts show sentiment distribution and model performance.

>  *Note: See `/visualizations/` folder for word clouds and model comparison plots.*

---

##  Impact

Automated sentiment analysis using NLP enables businesses to:

- Understand customer opinions at scale.
- Detect product issues early.
- Enhance marketing and product development strategies.

---

##  Conclusion

This project demonstrates how **NLP and machine learning** can be used to classify product reviews effectively. By comparing multiple models and leveraging proper preprocessing techniques, we achieve accurate and scalable sentiment analysis.

---
