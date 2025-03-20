import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from datasets import load_dataset
import re
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("fine_tuning.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Text preprocessing function
def clean_text(text):
    """Clean and normalize text data"""
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Convert to lowercase
    text = text.lower()

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    return text.strip()


class RatingDataset(Dataset):
    """Dataset for text with star ratings"""

    def __init__(self, texts, ratings, tokenizer, max_length=512):
        self.texts = texts
        self.ratings = ratings  # Ratings are 1-5
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        rating = self.ratings[idx]

        # Convert ratings to 0-4 range for classification
        rating = rating - 1

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Remove the batch dimension
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(rating, dtype=torch.long),
        }


def train(model, dataloader, optimizer, scheduler, device):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        # Get predictions
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Backward and optimize
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy, all_preds, all_labels


def evaluate(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            # Get predictions
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    report = classification_report(all_labels, all_preds, output_dict=True)
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)

    return avg_loss, accuracy, report, all_preds, all_labels


def plot_confusion_matrix(labels, predictions, epoch, save_dir="checkpoints/lora"):
    """Plot and save confusion matrix"""
    # Convert back to 1-5 ratings for clarity
    actual_labels = [label + 1 for label in labels]
    actual_predictions = [pred + 1 for pred in predictions]

    # Create confusion matrix
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(actual_labels, actual_predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(1, 6),
        yticklabels=range(1, 6),
    )
    plt.xlabel("Predicted Rating")
    plt.ylabel("True Rating")
    plt.title(f"Confusion Matrix - Epoch {epoch+1}")

    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the figure
    save_path = os.path.join(save_dir, f"confusion_matrix_epoch_{epoch+1}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Confusion matrix saved to {save_path}")


def save_model(model, tokenizer, output_dir):
    """Save the LoRA model and tokenizer"""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")


def load_model(model_path, device):
    """Load a saved LoRA model"""
    # Load the PEFT configuration
    peft_config = PeftConfig.from_pretrained(model_path)

    # Load the base model
    model = BertForSequenceClassification.from_pretrained(
        peft_config.base_model_name_or_path, num_labels=5
    )

    # Load the PEFT model
    model = PeftModel.from_pretrained(model, model_path)
    model.to(device)

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_path)

    return model, tokenizer


def main():
    # Configuration
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,  # rank of the update matrices
        lora_alpha=32,  # scaling factor
        lora_dropout=0.1,  # dropout probability
        target_modules=["query", "key", "value"],  # which modules to apply LoRA to
    )

    # Model configuration
    model_name = "bert-base-uncased"
    num_labels = 5  # 5 star ratings

    # Training configuration
    batch_size = 16
    num_epochs = 10
    learning_rate = 5e-5
    warmup_steps = 500
    max_seq_length = 256
    checkpoint_dir = "checkpoints/lora"

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # Add LoRA adapters to the model
    model = get_peft_model(model, lora_config)
    model.to(device)

    # Print trainable parameters
    model.print_trainable_parameters()

    # Load dataset - using Amazon reviews as an example
    logger.info("Loading dataset...")
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_All_Beauty",
        trust_remote_code=True,
    )

    # Clean and prepare data
    raw_data = dataset["full"].select_columns(["text", "rating"])
    raw_data = raw_data.select(range(20000))  # Limit dataset size

    texts = [clean_text(text) for text in raw_data["text"]]
    ratings = list(raw_data["rating"])

    # Remove empty texts
    filtered_data = [
        (text, rating) for text, rating in zip(texts, ratings) if text.strip()
    ]
    texts, ratings = zip(*filtered_data)

    # Split into train/validation
    from sklearn.model_selection import train_test_split

    train_texts, val_texts, train_ratings, val_ratings = train_test_split(
        texts, ratings, test_size=0.2, random_state=42, stratify=ratings
    )

    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")

    # Create datasets
    train_dataset = RatingDataset(train_texts, train_ratings, tokenizer, max_seq_length)
    val_dataset = RatingDataset(val_texts, val_ratings, tokenizer, max_seq_length)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    logger.info("Starting training...")
    best_accuracy = 0

    for epoch in range(num_epochs):
        # Train
        train_loss, train_accuracy, train_preds, train_labels = train(
            model, train_dataloader, optimizer, scheduler, device
        )

        # Evaluate
        val_loss, val_accuracy, val_report, val_preds, val_labels = evaluate(
            model, val_dataloader, device
        )

        # Log results
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(
            f"Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f}"
        )
        logger.info(f"Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")

        # Plot confusion matrix
        plot_confusion_matrix(val_labels, val_preds, epoch, checkpoint_dir)

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_model(model, tokenizer, f"{checkpoint_dir}/best_model")
            logger.info(f"New best model saved with accuracy: {val_accuracy:.4f}")

    # Save final model
    save_model(model, tokenizer, f"{checkpoint_dir}/final_model")

    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_accuracy:.4f}")

    return best_accuracy


if __name__ == "__main__":
    main()
