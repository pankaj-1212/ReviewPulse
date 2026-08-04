import streamlit as st
import pickle

# एआई मॉडल एसेट्स को लोड करें
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('sentiment_ml_model.pkl', 'rb') as f:
    model = pickle.load(f)

# वेब पेज की सजावट
st.title("✨ ReviewPulse: AI Sentiment Analyzer")
st.write("Type any product review below to test the live Machine Learning model.")

# यूजर से इनपुट लेने के लिए बॉक्स
user_review = st.text_area("Enter Customer Review Text:", "This phone has an amazing camera and fast charging!")

if st.button("Analyze Sentiment"):
    if user_review.strip() != "":
        # एआई प्रेडिक्शन रन करें
        vectorized_text = vectorizer.transform([user_review])
        prediction = model.predict(vectorized_text)
        
        # रिजल्ट दिखाएं
        if prediction == 1:
            st.success("✨ Positive Sentiment (Happy Customer)")
        else:
            st.error("🚨 Negative Sentiment (Disappointed Customer)")
    else:
        st.warning("Please enter some text to analyze.")
