import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Fake Review Detector", page_icon="🛒", layout="wide")

st.title("🛒 E-Commerce Fake Review Detection System")
st.write("Detect whether a product review is **Fake or Genuine**.")

st.divider()

# Sidebar
st.sidebar.title("🧑 User Panel")
st.sidebar.write("Review Authenticity Checker")

# Load dataset
data = pd.read_csv("fake_reviews_dataset_1000.csv")

X = data["review_text"]
y = data["label"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

col1, col2 = st.columns([2,1])

# -----------------------
# Product Section
# -----------------------

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
        "Write a review",
        placeholder="Example: This product is amazing and worth the money"
    )

    predict_button = st.button("🔍 Analyze Review")

# -----------------------
# Dataset Insights
# -----------------------

with col2:

    st.subheader("📊 Dataset Insights")

    total = len(data)
    fake = len(data[data['label']=="fake"])
    genuine = len(data[data['label']=="genuine"])

    st.metric("Total Reviews", total)
    st.metric("Fake Reviews", fake)
    st.metric("Genuine Reviews", genuine)

    fig, ax = plt.subplots()
    ax.bar(["Fake","Genuine"],[fake,genuine])
    ax.set_title("Dataset Distribution")
    st.pyplot(fig)

st.divider()

# -----------------------
# Screenshot Upload
# -----------------------

st.subheader("📷 Upload Review Screenshot")

uploaded_file = st.file_uploader("Upload image of a review", type=["png","jpg","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Screenshot")

    extracted_text = pytesseract.image_to_string(image)

    st.write("Extracted Review Text:")
    st.info(extracted_text)

    review = extracted_text

# -----------------------
# Prediction
# -----------------------

if predict_button and review:

    review_vec = vectorizer.transform([review])

    prediction = model.predict(review_vec)
    prob = model.predict_proba(review_vec)

    st.subheader("🧠 Prediction Result")

    if prediction[0] == "fake":
        st.error("⚠ This review appears FAKE")
        st.write("Fake Probability:", round(prob[0][0]*100,2),"%")
    else:
        st.success("✅ This review appears GENUINE")
        st.write("Genuine Probability:", round(prob[0][1]*100,2),"%")

    # -----------------------
    # AI Chatbot Explanation
    # -----------------------

    st.subheader("🤖 AI Explanation")

    if prediction[0] == "fake":
        st.write("""
        This review may be fake because:

        • It contains promotional language  
        • Very short or repetitive text  
        • Unusual review patterns  
        """)
    else:
        st.write("""
        This review appears genuine because:

        • Natural sentence structure  
        • Balanced opinion  
        • Common user language  
        """)

st.divider()

st.markdown("""
### 📌 About
This system detects fake product reviews using **Machine Learning and NLP**.

Algorithm used:
- TF-IDF Vectorization
- Logistic Regression
""")
