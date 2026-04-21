import streamlit as st
import joblib

@st.cache_resource
def load_model():
    return joblib.load("emotion_classifier.joblib")

model = load_model()

label_map = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy"
}

st.title("Emotion Classifier")

text_input = st.text_area("Enter text")

if st.button("Predict"):
    if text_input.strip() == "":
        st.warning("Please enter text")
    else:
        pred = model.predict([text_input])[0]
        probs = model.predict_proba([text_input])[0]

        st.subheader("Result")
        st.success(label_map[pred])

        st.subheader("Confidence")
        st.write(f"{max(probs):.2%}")

        st.subheader("All Probabilities")
        for i, p in enumerate(probs):
            st.write(f"{label_map[i]}: {p:.2%}")
