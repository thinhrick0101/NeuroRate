import os
import sys
import torch
import argparse

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Print PyTorch version and CUDA availability for debugging
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

from config import Config
from tokenizer import WordPieceTokenizer
from model import BERT
from dataset import create_dataloaders
from training import train_model
from inference import BERTInference


def main():
    parser = argparse.ArgumentParser(
        description="BERT for Amazon Reviews Star Rating Prediction"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "inference", "explore_dataset"],
        help="Mode to run the script (train, inference, or explore_dataset)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the pre-trained model for inference",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to predict star rating for (inference mode)",
    )
    
    # Parse arguments, providing empty list for Jupyter environment
    args = parser.parse_args([])

    # Initialize config
    config = Config()

    # Create directories
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.tokenizer_dir, exist_ok=True)

    if args.mode == "explore_dataset":
        # Simple mode to explore dataset structure
        from datasets import load_dataset

        dataset = load_dataset(
            config.dataset_name, config.dataset_subset, trust_remote_code=True
        )
        print("Dataset structure:")
        print(dataset)
        print("\nAvailable splits:", list(dataset.keys()))

        # Display sample data
        if "full" in dataset:
            split = "full"
        else:
            split = list(dataset.keys())[0]

        print(f"\nSample from split '{split}':")
        sample = dataset[split][0]
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                # Truncate long text values
                value = value[:100] + "..."
            print(f"{key}: {value}")

    elif args.mode == "train":
        print("Creating dataloaders...")
        train_dataloader, val_dataloader, tokenizer = create_dataloaders(config)

        print("Initializing model...")
        model = BERT(config)

        print(f"Starting training for {config.num_epochs} epochs...")
        model = train_model(model, train_dataloader, val_dataloader, config)

        print(f"Training complete. Model saved to {config.model_save_dir}")

    elif args.mode == "inference":
        if args.model_path is None:
            args.model_path = os.path.join(
                config.model_save_dir, "bert_star_rating_best.pt"
            )

        if args.text is None:
            args.text = "This product is absolutely amazing! I love it."

        print(f"Loading model from {args.model_path}...")
        inference = BERTInference(args.model_path, config)

        print(f"Predicting star rating for: {args.text}")
        result = inference.predict(args.text)

        print(f"Predicted star rating: {result['star_rating']}")
        print("Probability distribution:")
        for star, prob in result["probabilities"].items():
            print(f"{star} stars: {prob:.4f}")


if __name__ == "__main__":
    main()