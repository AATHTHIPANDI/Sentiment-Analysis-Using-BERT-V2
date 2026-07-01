from flask import Flask, render_template, request, jsonify
from sentiment_analysis import load_model_and_tokenizer, analyze_sentiment
import torch
import os
import gc  # For garbage collection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Force CPU usage for serverless deployment
device = torch.device("cpu")
logging.info(f"Using device: {device}")

# Load model and tokenizer once at startup with optimizations
try:
    # Clear any existing cached memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    tokenizer, model = load_model_and_tokenizer("nlptown/bert-base-multilingual-uncased-sentiment", device)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise

def map_sentiment_to_category(score):
    """Map 1-5 sentiment score to category and color"""
    if score <= 2:  # Negative/Angry
        return {"category": "Negative", "color": "red", "emoji": "🔴"}
    elif score <= 3:  # Neutral
        return {"category": "Neutral", "color": "orange", "emoji": "🟠"}
    else:  # Positive/Happy
        return {"category": "Positive", "color": "green", "emoji": "🟢"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        score, probabilities = analyze_sentiment(text, tokenizer, model, device)
        sentiment_info = map_sentiment_to_category(score)
        
        return jsonify({
            "score": score,
            "probabilities": {str(i+1): float(p) for i, p in enumerate(probabilities)},
            "category": sentiment_info["category"],
            "color": sentiment_info["color"],
            "emoji": sentiment_info["emoji"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use debug=True for development only
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)