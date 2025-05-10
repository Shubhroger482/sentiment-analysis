import streamlit as st
import joblib

# Set up Streamlit page
st.set_page_config(page_title="🎬 IMDB Sentiment Analyzer", page_icon="🎥", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.joblib")

model = load_model()

# Custom UI
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

st.markdown("<h1 style='text-align: center;'>🎬 IMDB Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a movie review below to predict the sentiment.</p>", unsafe_allow_html=True)

# Input text
review = st.text_area("Your Review", placeholder="Type or paste a movie review here...", height=200)

# Analyze button
if st.button("Analyze Sentiment"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        prediction = model.predict([review])[0]
        if prediction == "positive":
            st.success("🎉 Sentiment: Positive")
        else:
            st.error("😞 Sentiment: Negative")
