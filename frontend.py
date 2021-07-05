import streamlit as st
import requests

# interact with FastAPI endpoint
backend = "http://127.0.0.1:8000/predict-sentiment"


def process(text, server_url):

    r = requests.post(
        server_url, json={"textinput":text}, headers={"Content-Type": "application/json"}, timeout=8000)

    return r

# construct UI layout
st.title("Sentiment Analysis (7 Emotions)")

st.write(
    """Sentiment analysis based on the 1-P-3-ISEAR dataset.
         The front end is handled by streamlit and the backend with a FastAPI service.
         Visit 'http://127.0.0.1:8000/docs' for FastAPI documentation."""
)  # description and instructions

# input_text = st.text_input('How are you feeling?')
input_text = st.text_area('How are you feeling?', height=None, max_chars=None, key=None)

if st.button("Get Emotion"):

    if input_text:
        sentiment = process(input_text, backend)
        st.write(sentiment)

    else:
        # handle case with no image
        st.write("Need a sentence to analyze")