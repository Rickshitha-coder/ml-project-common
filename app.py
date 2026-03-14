import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Page config
st.set_page_config(page_title="Fake Review Detector", page_icon="🛒", layout="wide")

# Header
st.markdown("""
# 🛒 E-Commerce Fake Review Detection System
Check whether a product review is **Genuine or Fake** using Machine Learning.
""")

st.divider()

# Sidebar (E-commerce style menu)
st.sidebar.title("🛍️ User Panel")
st.sidebar.write("Review Analysis Tool")

st.sidebar.info("""
Use this tool to verify whether a product review is **authentic or fake**.
Useful for **online shoppers and sellers**.
""")

# Load dataset
data = pd.read_csv("fake_reviews_dataset_1000.csv")

# Features and labels
X = data["review_text"]
y = data["label"]

# Text vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

st.success("✔ Machine Learning Model Ready")

# Layout columns
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("✍️ Write a Product Review")

    review = st.text_area(
        "Enter your review here",
        placeholder="Example: This product quality is amazing and worth the money..."
    )

    predict_button = st.button("🔍 Check Review Authenticity")

with col2:
    st.subheader("📊 Review Statistics")

    st.metric("Total Reviews in Dataset", len(data))
    st.metric("Fake Reviews", len(data[data['label']=="fake"]))
    st.metric("Genuine Reviews", len(data[data['label']=="genuine"]))

st.divider()

# Prediction
if predict_button:

    if review.strip() == "":
        st.warning("⚠ Please enter a review first.")
    else:

        review_vec = vectorizer.transform([review])
        prediction = model.predict(review_vec)

        st.subheader("🧠 Prediction Result")

        if prediction[0] == "fake":
            st.error("⚠ This review appears to be **FAKE**.")
        else:
            st.success("✅ This review appears to be **GENUINE**.")

# Footer
st.divider()

st.markdown("""
### ℹ About This Project
This application uses **Machine Learning and Natural Language Processing** to detect fake product reviews.  
Algorithm used: **Logistic Regression + TF-IDF Text Vectorization**

Developed using **Python, Streamlit, and Scikit-Learn**.
""")
