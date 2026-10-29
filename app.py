from flask import Flask, request, jsonify
import pickle
import sqlite3
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure necessary NLTK components are downloaded quietly
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

app = Flask(__name__)

# 1. Load the corrected, high-accuracy 3-class AI model assets at startup
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('sentiment_ml_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Automatically set up a local SQLite Database file
DB_NAME = "reviews_database.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Create a table to store live incoming data points for dashboarding
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customer_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_text TEXT,
            sentiment TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Initialize text preprocessing tools matching the training notebook
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Applies standard industry NLP preprocessing to raw user input."""
    text = str(text).lower()
    text = re.sub(r'<.*?>|https?://\S+|www\.\S+', '', text) # Remove HTML & URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)                # Remove punctuation & numbers
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# 3. Create the API endpoint to process live text requests
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' key in JSON request payload"}), 400
        
    raw_review = data['text']
    
    # Industry Standard Rule: Clean raw text using identical notebook pipeline before vectorization
    cleaned_review = clean_text(raw_review)
    vectorized_text = vectorizer.transform([cleaned_review])
    
    # Generate the numeric prediction from the loaded ML model
    prediction = model.predict(vectorized_text)[0]
    
    # Map predictions based on the new 3-class structural system
    if prediction == 2:
        sentiment_result = "Positive"
    elif prediction == 1:
        sentiment_result = "Neutral"
    else:
        sentiment_result = "Negative"
        
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save the input and prediction results into our local SQL database
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO customer_reviews (raw_text, sentiment, timestamp) VALUES (?, ?, ?)",
        (raw_review, sentiment_result, current_time)
    )
    conn.commit()
    conn.close()
    
    # Return a clean JSON response back to the client application
    return jsonify({
        "status": "success",
        "review_processed": raw_review,
        "predicted_sentiment": sentiment_result,
        "logged_at": current_time
    })

if __name__ == '__main__':
    print("ReviewPulse API Server is spinning up on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
