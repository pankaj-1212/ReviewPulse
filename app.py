from flask import Flask, request, jsonify
import pickle
import sqlite3
from datetime import datetime

app = Flask(__name__)

# 1. Load the corrected, high-accuracy AI model assets at startup
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('sentiment_ml_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Automatically set up a local SQLite Database file
DB_NAME = "reviews_database.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Create a table to store live incoming data points for Power BI
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

# 3. Create the API endpoint to process live text requests
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' key in JSON request payload"}), 400
    
    raw_review = data['text']
    
    # Transform raw text into mathematical vectors and run prediction
    vectorized_text = vectorizer.transform([raw_review])
    prediction = model.predict(vectorized_text)
    
    # Map predictions (1 = Positive, 0 = Negative)
    sentiment_result = "Positive" if prediction == 1 else "Negative"
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
    
    # Return a clean JSON response back to the client
    return jsonify({
        "status": "success",
        "review_processed": raw_review,
        "predicted_sentiment": sentiment_result,
        "logged_at": current_time
    })

if __name__ == '__main__':
    print("🚀 ReviewPulse API Server is spinning up on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)