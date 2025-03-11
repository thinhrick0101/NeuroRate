import torch
import os
from model import Encoder
from model_config import Config

# Load configuration
config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path, device):
    """
    Load trained model with proper vocabulary size
    """
    print(f"Loading model from {checkpoint_path}...")

    # Load checkpoint with weights_only=True for security
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    # Get the state dictionary to extract the embedding size
    state_dict = checkpoint["model_state_dict"]

    # Extract vocabulary size from the embedding weight matrix
    vocab_size = state_dict["sentence_embedding.embedding.weight"].shape[0]
    print(f"Detected vocabulary size: {vocab_size}")

    # Initialize model with the correct vocabulary size
    model = Encoder(
        d_model=config.hidden_size,
        num_heads=config.num_heads,
        drop_prob=config.hidden_dropout_prob,
        num_layers=config.Layer,
        max_sequence_length=config.max_position_embeddings,
        use_bert_tokenization=True,
        num_classes=5,  # For Amazon ratings (1-5)
        custom_vocab_size=vocab_size,  # Pass the detected vocab size
    ).to(device)

    # Load the state dictionary
    model.load_state_dict(state_dict)
    model.eval()

    return model


# Load the trained model
checkpoint_path = "checkpoints/best_model.pt"
if os.path.exists(checkpoint_path):
    model = load_model(checkpoint_path, device)
else:
    print(f"Warning: No model found at {checkpoint_path}. Using untrained model.")
    # Initialize a default model if no checkpoint is available
    model = Encoder(
        d_model=config.hidden_size,
        num_heads=config.num_heads,
        drop_prob=config.hidden_dropout_prob,
        num_layers=config.Layer,
        max_sequence_length=config.max_position_embeddings,
        use_bert_tokenization=True,
        num_classes=5,
    ).to(device)


# Perform inference
def predict(text):
    with torch.no_grad():
        # Create proper attention mask for a single input
        # Shape: [1, 1, 1, max_length]
        attention_mask = torch.ones(
            (1, 1, 1, config.max_position_embeddings),
            device=device,
        )

        # Forward pass
        outputs = model([text], attention_mask, start_token=True, end_token=True)

        # Get prediction (add 1 since we subtracted 1 during training)
        predicted_class = torch.argmax(outputs, dim=1).item()
        rating = predicted_class + 1  # Convert back to 1-5 rating scale

    return rating


# Example usage
if __name__ == "__main__":
    sample_text = "This product exceeded my expectations. The quality is excellent!"
    result = predict(sample_text)
    print(f"Predicted rating: {result}/5")

    negative_text = "Terrible product, broke after one use. Would not recommend."
    result = predict(negative_text)
    print(f"Predicted rating: {result}/5")