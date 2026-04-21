import os
import joblib
import streamlit as st

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "emotion_classifier.joblib")
    return joblib.load(model_path)

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
