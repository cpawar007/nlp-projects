import streamlit as st
import joblib
import os

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "emotion_classifier.joblib")
    return joblib.load(model_path)

model = load_model()

label_map = {
    0: "Sadness",
    1: "Anger",
    2: "Love",
    3: "Surprise",
    4: "Fear",
    5: "Joy"
}

st.set_page_config(page_title="Emotion Classifier", layout="centered")

st.title("Emotion Classifier")
st.write("Enter text to detect the dominant emotion.")

text_input = st.text_area("Text")

if st.button("Analyze"):
    if text_input.strip():
        pred = model.predict([text_input])[0]
        prob = model.predict_proba([text_input])[0].max()

        emotion = label_map[pred]

        st.divider()

        st.subheader("Result")
        st.markdown(f"### {emotion}")
        st.progress(int(prob * 100))
        st.caption(f"Confidence: {prob:.2%}")
    else:
        st.warning("Please enter some text.")
