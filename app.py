import streamlit as st
import pickle

st.set_page_config(page_title="🎬 IMDB Sentiment Analyzer", page_icon="🎥", layout="centered")

# Custom CSS for clean UI
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
            padding: 20px;
        }
        .stTextArea textarea {
            font-size: 16px;
        }
        .stButton>button {
            font-size: 16px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🎬 IMDB Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a movie review below to predict the sentiment.</p>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    with open("sentiment_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Text input
review = st.text_area("Your Review", placeholder="Type or paste a movie review here...", height=200)

# Prediction button
if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        prediction = model.predict([review])[0]
        if prediction == "positive":
            st.success("🎉 Sentiment: Positive")
        else:
            st.error("😞 Sentiment: Negative")
