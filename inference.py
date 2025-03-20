import torch
import torch.nn.functional as F
from config import Config
from model import BERT
from tokenizer import WordPieceTokenizer


class BERTInference:
    def __init__(self, model_path, config=None):
        if config is None:
            config = Config()

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer_handler = WordPieceTokenizer(config)
        self.tokenizer = self.tokenizer_handler.load()

        # Load model
        self.model = BERT(config)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        # Use the tokenizer directly without the tokenize method
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_sequence_length,
            return_tensors="pt",
        )

        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

            # Convert to 1-5 rating
            star_rating = predicted_class + 1

        # Get probability distribution
        probs = probabilities.cpu().numpy()[0]

        return {
            "star_rating": star_rating,
            "probabilities": {
                i + 1: float(probs[i]) for i in range(self.config.num_classes)
            },
            "text": text,
        }