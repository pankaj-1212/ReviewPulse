import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ReviewPulse - AI Sentiment Analyzer", page_icon="📊", layout="centered")

# --- DOWNLOAD NLP ASSETS QUIETLY ---
@st.cache_resource
def load_nlp_assets():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    return set(stopwords.words('english')), WordNetLemmatizer()

stop_words, lemmatizer = load_nlp_assets()

# --- LOAD AI MODELS ---
@st.cache_resource
def load_ml_model():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('sentiment_ml_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

try:
    vectorizer, model = load_ml_model()
except FileNotFoundError:
    st.error("🚨 Model files missing! Please ensure 'tfidf_vectorizer.pkl' and 'sentiment_ml_model.pkl' are in the same folder.")

# --- TEXT CLEANING FUNCTION ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>|https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# --- USER INTERFACE (UI) ---
st.title("📊 ReviewPulse: AI Sentiment Analyzer")
st.write("Evaluate customer feedback instantly using our advanced 3-Class Sentiment Engine.")

user_input = st.text_area("✍️ Enter Customer Review Text Here:", placeholder="Type a review (e.g., The phone is great but delivery was late...)")

if st.button("Analyze Sentiment", type="primary"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        # Preprocess and predict
        cleaned_review = clean_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_review])
        prediction = model.predict(vectorized_text)[0]
        
        # 3-Class Sentiment Formatting
        st.subheader("🔮 Analysis Result:")
        
        # Checking if explicitly positive keywords exist to assist the model
        positive_boosters = ['amazing', 'perfect', 'love', 'best', 'great', 'excellent', 'stunning']
        has_positive_word = any(word in cleaned_review for word in positive_boosters)
        
        if prediction == 2 or (prediction == 1 and has_positive_word):
            st.success("😊 **Positive Sentiment** — This is a **Happy Customer**!")
        elif prediction == 1:
            st.info("😐 **Neutral Sentiment** — The feedback is **Satisfied or Factual**.")
        else:
            st.error("😭 **Negative Sentiment** — This is an **Unhappy Customer**!")
