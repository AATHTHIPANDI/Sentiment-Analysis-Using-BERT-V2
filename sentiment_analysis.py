"""
Sentiment analysis module using BERT model for text classification.
Provides functions for sentiment analysis with scores from 1-5.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F


DEFAULT_MODEL = 'nlptown/bert-base-multilingual-uncased-sentiment'


def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL, device: torch.device = None):
    """
    Load the BERT model and tokenizer with memory optimizations.
    Args:
        model_name: HuggingFace model name/path
        device: torch device to use (will detect CUDA if None)
    Returns:
        tuple: (tokenizer, model)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load tokenizer with minimal memory usage
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            local_files_only=False
        )
        
        # Load model with memory optimizations
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,  # Use float32 instead of float64
            local_files_only=False
        )
        
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        # Clear CUDA cache if using GPU
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return tokenizer, model
    except Exception as e:
        raise RuntimeError(f"Failed to load model/tokenizer: {str(e)}")


def analyze_sentiment(text: str, tokenizer, model, device: torch.device = None) -> tuple:
    """
    Analyze sentiment of input text.
    Args:
        text: Input text to analyze
        tokenizer: Pre-loaded BERT tokenizer
        model: Pre-loaded BERT model
        device: torch device to use (will detect CUDA if None)
    Returns:
        tuple: (sentiment_score, probabilities)
        - sentiment_score: int 1-5 (1=very negative, 5=very positive)
        - probabilities: array of class probabilities
    """
    if not text:
        raise ValueError("Empty text provided")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tokenize and move to device
    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=512)
    tokens = tokens.to(device)
    
    # Get predictions
    with torch.no_grad():
        result = model(tokens)
        probabilities = F.softmax(result.logits, dim=-1)
        sentiment_score = int(torch.argmax(probabilities)) + 1
        
    return sentiment_score, probabilities.cpu().numpy()[0]


if __name__ == '__main__':
    # Example usage in console mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    
    tokenizer, model = load_model_and_tokenizer(device=device)
    
    while True:
        text = input("\nEnter text to analyze (or 'exit' to quit): ").strip()
        if text.lower() == 'exit':
            break
            
        try:
            score, probs = analyze_sentiment(text, tokenizer, model, device)
            print(f"\nSentiment Score (1-5): {score}")
            print("Class Probabilities:")
            for i, p in enumerate(probs, 1):
                print(f"  Score {i}: {p:.4f}")
        except Exception as e:
            print(f"Error: {str(e)}")