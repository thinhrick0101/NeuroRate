import torch
from model import Encoder
from model_config import Config

# Load configuration
config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model - include num_classes
model = Encoder(
    d_model=config.hidden_size,
    num_heads=config.num_heads,
    drop_prob=config.hidden_dropout_prob,
    num_layers=config.Layer,
    max_sequence_length=config.max_position_embeddings,
    use_bert_tokenization=True,
    num_classes=2,  # Set based on your classification task
).to(device)

# Load trained model weights
checkpoint = torch.load("checkpoints/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


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

        # Get prediction
        predictions = torch.argmax(outputs, dim=1)

    return predictions.item()


# Example usage
result = predict("Sample text to classify")
print(f"Prediction: {result}")
