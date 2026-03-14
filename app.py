import streamlit as st
import pickle

# Load model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

st.title("🕵️ Fake Product Review Detector")

st.write("Enter a product review to check if it is Fake or Genuine.")

review = st.text_area("Enter Review")

if st.button("Predict"):

    review_vec = vectorizer.transform([review])

    prediction = model.predict(review_vec)

    if prediction[0] == "fake":
        st.error("⚠️ This review looks FAKE")
    else:
        st.success("✅ This review looks GENUINE")