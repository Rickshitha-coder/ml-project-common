import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Page settings
st.set_page_config(page_title="Fake Review Detector", page_icon="🛒", layout="wide")

# Title
st.title("🛒 E-Commerce Fake Review Detection System")
st.write("Detect whether a product review is **Fake or Genuine** using Machine Learning.")

st.divider()

# Sidebar
st.sidebar.title("🧑 User Panel")
st.sidebar.write("Review Authenticity Checker")

st.sidebar.info("""
Enter a product review and our ML model will analyze whether it is **Fake or Genuine**.
""")

# Load dataset
data = pd.read_csv("fake_reviews_dataset_1000.csv")

X = data["review_text"]
y = data["label"]

# Vectorizer
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

# Layout
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("📦 Product")

    st.image(
        "https://images.unsplash.com/photo-1585386959984-a4155224a1ad",
        width=350
    )

    st.write("**Wireless Bluetooth Headphones**")
    st.write("Price: **$59.99**")

    rating = st.slider("⭐ Product Rating",1,5,4)

    review = st.text_area(
        "Write your review",
        placeholder="Example: This product quality is amazing and worth the money"
    )

    predict_button = st.button("🔍 Analyze Review")

with col2:
    st.subheader("📊 Dataset Insights")

    total = len(data)
    fake = len(data[data['label']=="fake"])
    genuine = len(data[data['label']=="genuine"])

    st.metric("Total Reviews", total)
    st.metric("Fake Reviews", fake)
    st.metric("Genuine Reviews", genuine)

    # Chart
    fig, ax = plt.subplots()
    ax.bar(["Fake","Genuine"],[fake,genuine])
    ax.set_title("Dataset Distribution")
    st.pyplot(fig)

st.divider()

# Prediction
if predict_button:

    if review.strip() == "":
        st.warning("⚠ Please enter a review.")
    else:

        review_vec = vectorizer.transform([review])
        prediction = model.predict(review_vec)

        prob = model.predict_proba(review_vec)

        st.subheader("🧠 Model Prediction")

        if prediction[0] == "fake":
            st.error("⚠ This review appears **FAKE**")
            st.write("Fake Probability:", round(prob[0][0]*100,2),"%")

        else:
            st.success("✅ This review appears **GENUINE**")
            st.write("Genuine Probability:", round(prob[0][1]*100,2),"%")

st.divider()

st.markdown("""
### 📌 About the Project
This application detects **fake product reviews** using:

- **Machine Learning**
- **TF-IDF Text Vectorization**
- **Logistic Regression**

Developed using **Python, Streamlit, and Scikit-Learn**.
""")
