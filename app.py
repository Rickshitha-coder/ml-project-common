import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Title
st.title("🕵️ Fake Product Review Detector")

st.write("This app detects whether a product review is **Fake or Genuine**.")

# Load dataset
data = pd.read_csv("fake_reviews_dataset_1000.csv")

# Features and labels
X = data["review_text"]
y = data["label"]

# Convert text to numbers
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

st.success("Model trained successfully!")

# User input
review = st.text_area("Enter a product review")

if st.button("Predict"):

    review_vec = vectorizer.transform([review])
    prediction = model.predict(review_vec)

    if prediction[0] == "fake":
        st.error("⚠️ This review looks FAKE")
    else:
        st.success("✅ This review looks GENUINE")
