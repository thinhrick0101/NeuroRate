import torch
import pandas as pd
from tqdm import tqdm
from config import Config
from inference import BERTInference

def batch_predict(texts, model_path=None, batch_size=16):
    """
    Run batch prediction on a list of texts
    
    Args:
        texts: List of review texts to predict ratings for
        model_path: Path to model weights (optional)
        batch_size: Number of texts to process at once
        
    Returns:
        List of prediction results
    """
    config = Config()
    if not model_path:
        model_path = f"{config.model_save_dir}/bert_star_rating_best.pt"
    
    inference_model = BERTInference(model_path=model_path, config=config)
    
    results = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_results = [inference_model.predict(text) for text in batch_texts]
        results.extend(batch_results)
    
    return results

# Example usage
if __name__ == "__main__":
    # Example: Load reviews from a CSV file
    try:
        df = pd.read_csv("reviews.csv")
        texts = df["review_text"].tolist()
        
        print(f"Running inference on {len(texts)} reviews...")
        predictions = batch_predict(texts)
        
        # Add predictions to dataframe
        df["predicted_rating"] = [result["star_rating"] for result in predictions]
        
        # Save results
        df.to_csv("reviews_with_predictions.csv", index=False)
        print("Predictions completed and saved.")
        
    except Exception as e:
        print(f"Error: {e}")
        
        # Example with hardcoded texts if no CSV available
        print("Running example predictions...")
        test_texts = [
            "The car consumes too much gas, I would recommend other cars"
        ]
        predictions = batch_predict(test_texts)
        
        for text, pred in zip(test_texts, predictions):
            print(f"Text: {text[:50]}...")
            print(f"Predicted rating: {pred['star_rating']} stars")