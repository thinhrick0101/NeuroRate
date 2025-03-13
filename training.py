import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
import json
import logging
from tqdm import tqdm
import numpy as np
import random
import re
from model import Encoder
from model_config import Config
from transformers import BertTokenizer, BertTokenizerFast
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

from datasets import load_dataset

# ...existing imports...
import torch.nn.functional as F  # if not already imported


# NEW CLASS: FocalLoss
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses training on hard examples.
    """

    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    """Dataset for text samples"""

    def __init__(self, texts, labels=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {"text": self.texts[idx]}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


def collate_fn(
    batch, tokenizer=None, max_length=512, device="cuda", config: Config = None
):
    texts = [item["text"] for item in batch]
    # If we're doing classification, also collect labels
    has_labels = "label" in batch[0]

    # Create a proper attention mask for transformer models
    # Shape: [batch_size, 1, 1, max_length] - broadcastable to attention scores
    attention_mask = torch.ones((len(texts), 1, 1, max_length), device=device)

    # Convert to tensors and create dict
    batch_dict = {
        "texts": texts,
        "attention_mask": attention_mask,
    }

    if has_labels:
        # Adjust labels to be zero-indexed (Amazon ratings are 1-5, we need 0-4) [0,num_class-1]
        labels = torch.tensor(
            [item["label"] - 1 for item in batch], device=device
        ).long()
        batch_dict["labels"] = labels

    return batch_dict


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    gradient_accumulation_steps=4,
    scheduler=None,
):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    optimizer.zero_grad()  # Zero gradients once at the beginning

    progress_bar = tqdm(dataloader, desc="Training")
    for i, batch in enumerate(progress_bar):
        # Get inputs
        texts = batch["texts"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels")

        # Forward pass
        outputs = model(texts, attention_mask, start_token=True, end_token=True)

        # If we have labels, calculate loss
        if labels is not None:
            loss = (
                criterion(outputs, labels) / gradient_accumulation_steps
            )  # Normalize loss
            # Store predictions and labels for metrics
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        else:
            # For unsupervised learning, you might use a different loss function
            loss = criterion(outputs) / gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Only update weights periodically to save memory
        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(dataloader):
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # Important! Call scheduler.step() after optimizer.step()
            optimizer.zero_grad()

        # Update metrics
        total_loss += (
            loss.item() * gradient_accumulation_steps
        )  # De-normalize for tracking
        progress_bar.set_postfix({"loss": f"{total_loss/(progress_bar.n+1):.4f}"})

    # Calculate accuracy if we have labels
    metrics = {}
    if all_labels and all_preds:
        metrics["accuracy"] = accuracy_score(all_labels, all_preds)
        logger.info(f"Training accuracy: {metrics['accuracy']:.4f}")

    return total_loss / len(dataloader), metrics


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get inputs
            texts = batch["texts"]
            attention_mask = batch["attention_mask"]
            labels = batch.get("labels")

            # Forward pass
            outputs = model(texts, attention_mask, start_token=True, end_token=True)

            # Calculate loss
            if labels is not None:
                loss = criterion(outputs, labels)
                # Store predictions and labels for metrics
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            else:
                loss = criterion(outputs)

            # Update metrics
            total_loss += loss.item()

    # Calculate additional metrics
    metrics = {}
    if all_labels and all_preds:
        metrics["accuracy"] = accuracy_score(all_labels, all_preds)
        metrics["predictions"] = all_preds
        metrics["labels"] = all_labels

        logger.info(f"Validation accuracy: {metrics['accuracy']:.4f}")

        # Generate classification report for more details
        report = classification_report(all_labels, all_preds, zero_division=0)
        logger.info(f"Classification Report:\n{report}")

    return total_loss / len(dataloader), metrics


def save_checkpoint(model, optimizer, scheduler, epoch, loss, history, filepath):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "loss": loss,
            "history": history,
        },
        filepath,
    )
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, scheduler, filepath):
    if not os.path.exists(filepath):
        logger.info(f"No checkpoint found at {filepath}, starting from scratch")
        return (
            model,
            optimizer,
            scheduler,
            0,
            float("inf"),
            {"train_loss": [], "val_loss": []},
        )

    logger.info(f"Loading checkpoint from {filepath}")
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (
        scheduler
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"]
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    history = checkpoint.get("history", {"train_loss": [], "val_loss": []})

    logger.info(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    return model, optimizer, scheduler, epoch, loss, history


def create_padding_mask(seq, padding_idx=0):
    """Create attention padding mask"""
    # seq shape: [batch_size, seq_len]
    # output shape: [batch_size, 1, 1, seq_len] for broadcasting
    return (seq != padding_idx).unsqueeze(1).unsqueeze(2).to(seq.device)


def plot_training_history(history, save_path="training_history.png"):
    """Plot and save training history"""
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Training history plot saved to {save_path}")


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

    # Remove non-alphanumeric characters (but keep spaces)
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


def create_balanced_sampler(labels):
    """Create a weighted sampler to balance classes in training"""
    # Count class frequencies
    class_counts = Counter(labels)
    logger.info(f"Class distribution: {class_counts}")

    # Calculate weights for each sample
    weights = [1.0 / class_counts[label] for label in labels]

    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(weights, len(labels))
    return sampler


# NEW FUNCTION: create_class_weight
def create_class_weight(labels, num_classes=5):
    """Compute class weights from label frequencies."""
    from collections import Counter

    count = Counter(labels)
    total = len(labels)
    class_weights = {}
    for cls in range(num_classes):
        if cls in count:
            class_weights[cls] = total / (count[cls] * num_classes)
        else:
            class_weights[cls] = 1.0
    max_weight = max(class_weights.values())
    for cls in class_weights:
        class_weights[cls] = min(5.0, class_weights[cls] / max_weight)
    logger.info(f"Class weights: {class_weights}")
    return class_weights


def log_dataset_statistics(texts, labels, split_name="Dataset"):
    """Log statistics about the dataset"""
    if not texts or not labels:
        return

    # Text length statistics
    text_lengths = [len(text.split()) for text in texts]
    avg_length = sum(text_lengths) / len(text_lengths)
    max_length = max(text_lengths)
    min_length = min(text_lengths)

    # Label distribution
    label_counts = Counter(labels)

    # Log statistics
    logger.info(f"\n{split_name} Statistics:")
    logger.info(f"Number of samples: {len(texts)}")
    logger.info(f"Average text length: {avg_length:.2f} words")
    logger.info(f"Min/Max text length: {min_length}/{max_length} words")
    logger.info(f"Label distribution: {label_counts}")

    # Create and save distribution plot
    if split_name != "Dataset":
        plt.figure(figsize=(8, 5))
        plt.bar(label_counts.keys(), label_counts.values())
        plt.title(f"{split_name} Rating Distribution")
        plt.xlabel("Rating")
        plt.ylabel("Count")
        plt.savefig(f"checkpoints/{split_name.lower()}_distribution.png")
        plt.close()


def plot_confusion_matrix(
    labels, predictions, epoch, save_path="checkpoints/confusion_matrix.png"
):
    """Plot and save confusion matrix"""
    # Convert predictions to actual ratings (add 1 since we subtracted 1 for 0-indexing)
    actual_labels = [label + 1 for label in labels]
    actual_predictions = [pred + 1 for pred in predictions]

    # Create confusion matrix
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

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"checkpoints/confusion_matrix_epoch_{epoch+1}.png")
    plt.close()
    logger.info(
        f"Confusion matrix saved to checkpoints/confusion_matrix_epoch_{epoch+1}.png"
    )


def train_and_save_tokenizer(
    texts,
    vocab_size=30000,
    min_frequency=2,
    save_path="checkpoints/tokenizer_vocab.txt",
):
    """
    Train a BERT tokenizer from scratch and save its vocabulary

    Args:
        texts: List of texts to train the tokenizer on
        vocab_size: Maximum vocabulary size
        min_frequency: Minimum frequency for a token to be included
        save_path: Path to save the vocabulary file

    Returns:
        Trained BERT Tokenizer
    """
    from transformers import BertTokenizerFast, BertTokenizer

    logger.info(
        f"Training BERT tokenizer with vocab_size={vocab_size}, min_frequency={min_frequency}"
    )

    # Create the folder for saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Get directory and filename from save_path
    save_dir = os.path.dirname(save_path)

    # Initialize a base tokenizer (we'll use this to train a new one)
    base_tokenizer = BertTokenizerFast.from_pretrained(
        "bert-base-uncased", do_lower_case=True
    )

    # Define the special tokens we want to include
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

    # Train a new tokenizer on the corpus without passing special_tokens directly
    # The base tokenizer already has these special tokens defined
    new_tokenizer = base_tokenizer.train_new_from_iterator(
        texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )

    # Make sure our special tokens are in the vocabulary
    # This step ensures the special tokens have the expected token IDs
    new_tokenizer.add_special_tokens(
        {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
        }
    )

    # Save the tokenizer
    new_tokenizer.save_pretrained(save_dir)
    logger.info(f"Tokenizer saved to {save_dir}")

    return new_tokenizer


def main():
    # Set random seed
    set_seed(42)

    # Configuration
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Memory saving settings
    gradient_accumulation_steps = 4
    batch_size = 4  # Reduced from 8

    # Training settings - moved earlier to avoid NameError
    num_epochs = 8
    checkpoint_path = "checkpoints/best_model.pt"
    history_path = "checkpoints/training_history.png"
    tokenizer_path = "checkpoints/vocab.txt"

    # Empty cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Data loading
    # Replace with your actual data loading logic
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_All_Beauty",
        trust_remote_code=True,
    )

    ds_raw = dataset["full"].select_columns(["text", "rating"])
    ds_raw = ds_raw.select(range(250000))

    # Extract texts and ratings
    texts = [clean_text(text) for text in ds_raw["text"]]
    ratings = list(ds_raw["rating"])

    # Remove empty texts and corresponding ratings
    filtered_data = [
        (text, rating) for text, rating in zip(texts, ratings) if text.strip()
    ]

    # Check if any data was filtered out
    if len(filtered_data) < len(texts):
        logger.info(f"Removed {len(texts) - len(filtered_data)} empty text samples")

    texts, ratings = zip(*filtered_data) if filtered_data else ([], [])

    # Log overall dataset statistics
    log_dataset_statistics(texts, ratings, "Overall Dataset")

    # Use stratified sampling to maintain rating distribution
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, ratings, test_size=0.2, random_state=42, stratify=ratings
    )

    # When processing Amazon ratings, ensure they're in the valid range
    for i, rating in enumerate(train_labels):
        if rating < 1 or rating > 5:
            logger.warning(f"Found invalid rating {rating}, setting to 3")
            train_labels[i] = 3

    for i, rating in enumerate(val_labels):
        if rating < 1 or rating > 5:
            logger.warning(f"Found invalid rating {rating}, setting to 3")
            val_labels[i] = 3

    # Log statistics for train and validation sets
    log_dataset_statistics(train_texts, train_labels, "Training Set")
    log_dataset_statistics(val_texts, val_labels, "Validation Set")

    train_dataset = TextDataset(
        train_texts, train_labels, max_length=config.max_position_embeddings
    )
    val_dataset = TextDataset(
        val_texts, val_labels, max_length=config.max_position_embeddings
    )

    # Create balanced sampler for training data
    train_sampler = create_balanced_sampler(train_labels)

    # Create data loaders with smaller batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,  # Use balanced sampler instead of shuffle=True
        collate_fn=lambda batch: collate_fn(
            batch,
            max_length=config.max_position_embeddings,
            device=device,
            config=config,
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(
            batch,
            max_length=config.max_position_embeddings,
            device=device,
            config=config,
        ),
    )

    # Calculate number of classes from labels - Amazon ratings are 1-5
    num_classes = 5
    logger.info(f"Using {num_classes} classes for classification")

    # Train WordPiece tokenizer on training texts (before model initialization)
    custom_tokenizer = None
    if not os.path.exists(tokenizer_path):
        logger.info("Training tokenizer from scratch...")
        custom_tokenizer = train_and_save_tokenizer(
            train_texts,
            vocab_size=config.vocab_size,
            min_frequency=2,
            save_path=tokenizer_path,
        )
    else:
        logger.info(f"Loading pre-trained tokenizer from {tokenizer_path}")
        tokenizer_dir = os.path.dirname(tokenizer_path)
        custom_tokenizer = BertTokenizerFast.from_pretrained(tokenizer_dir)

    # Log tokenizer vocabulary statistics
    if custom_tokenizer:
        logger.info(f"Tokenizer vocabulary size: {len(custom_tokenizer.vocab)}")
        # Sample a few texts to test tokenization
        sample_texts = train_texts[:3]
        for text in sample_texts:
            tokens = custom_tokenizer.tokenize(text[:100])  # First 100 chars only
            logger.info(f"Sample tokenization: {tokens}")

    # Calculate class weights for balanced loss
    class_weights = create_class_weight(
        [r - 1 for r in train_labels], num_classes=num_classes
    )
    weight_tensor = torch.FloatTensor(
        [class_weights[i] for i in range(num_classes)]
    ).to(device)

    # Model initialization remains unchanged
    model = Encoder(
        d_model=config.hidden_size,
        num_heads=config.num_heads,
        drop_prob=config.hidden_dropout_prob,
        num_layers=config.Layer,
        max_sequence_length=config.max_position_embeddings,
        use_bert_tokenization=True,
        corpus=None,
        vocab_file=tokenizer_path,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
    )

    # >>> CHANGE: Use a lower gamma in FocalLoss for more stable training
    criterion = FocalLoss(gamma=1.0, weight=weight_tensor)
    # <<< End change

    # ...existing scheduler initialization...

    # (Optional) To combat overfitting further, you may try:
    #   - Increasing dropout in embedding and classifier layers.
    #   - Introducing data augmentation on the minority classes.
    #   - Adjusting early stopping patience.

    # Add learning rate scheduler
    # Instead of ReduceLROnPlateau, use OneCycleLR for better convergence
    scheduler = OneCycleLR(
        optimizer,
        max_lr=2e-4,  # Increase max_lr
        total_steps=len(train_loader) * num_epochs // gradient_accumulation_steps,
        pct_start=0.2,
        anneal_strategy="linear",
        div_factor=10.0,  # Smaller div_factor
        final_div_factor=50.0,  # Smaller final_div_factor
    )

    # Let's keep ReduceLROnPlateau as a backup if you prefer it
    # scheduler = ReduceLROnPlateau(
    #    optimizer, mode="min", factor=0.5, patience=3, verbose=True
    # )

    # Setup for early stopping - now using both loss and accuracy
    early_stopping_patience = 5
    early_stopping_counter = 0
    best_val_accuracy = 0.0

    # Try to resume from checkpoint
    model, optimizer, scheduler, start_epoch, best_val_loss, history = load_checkpoint(
        model, optimizer, scheduler, checkpoint_path
    )

    early_stopping_counter = 0
    early_stopping_patience = 5
    best_val_loss = float("inf")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Train with gradient accumulation
        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            gradient_accumulation_steps=gradient_accumulation_steps,
            scheduler=scheduler,
        )
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.4f}")

        # Evaluate
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Validation loss: {val_loss:.4f}")

        # Plot confusion matrix using validation predictions
        if "predictions" in val_metrics and "labels" in val_metrics:
            plot_confusion_matrix(
                val_metrics["labels"], val_metrics["predictions"], epoch
            )

        # Update learning rate scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
            current_lr = scheduler.optimizer.param_groups[0]["lr"]
            logger.info(f"Current learning rate: {current_lr}")
        else:
            # If using OneCycleLR, no need to call step here as it's called after each batch
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Current learning rate: {current_lr}")

        # Update history with metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if "accuracy" in train_metrics:
            history.setdefault("train_accuracy", []).append(train_metrics["accuracy"])
        if "accuracy" in val_metrics:
            history.setdefault("val_accuracy", []).append(val_metrics["accuracy"])

        # Save periodic checkpoint (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                history,
                f"checkpoints/model_epoch_{epoch+1}.pt",
            )

        # Save best model (based on both loss and accuracy)
        improved = False

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                history,
                "checkpoints/best_loss_model.pt",
            )
            logger.info(f"New best validation loss: {val_loss:.4f}")

        if "accuracy" in val_metrics and val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            improved = True
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                history,
                "checkpoints/best_accuracy_model.pt",
            )
            logger.info(f"New best validation accuracy: {val_metrics['accuracy']:.4f}")

        # If either loss or accuracy improved, save as overall best model
        if improved:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, history, checkpoint_path
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            logger.info(
                f"Validation metrics did not improve. Counter: {early_stopping_counter}/{early_stopping_patience}"
            )

            if early_stopping_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Enhanced plot for training history
    plot_enhanced_training_history(history, history_path)

    logger.info("Training completed!")
    return best_val_loss


def plot_enhanced_training_history(history, save_path="training_history.png"):
    """Plot and save enhanced training history with multiple metrics"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Loss subplot
    axes[0].plot(history["train_loss"], label="Training Loss")
    axes[0].plot(history["val_loss"], label="Validation Loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy subplot (if available)
    if "train_accuracy" in history and "val_accuracy" in history:
        axes[1].plot(history["train_accuracy"], label="Training Accuracy")
        axes[1].plot(history["val_accuracy"], label="Validation Accuracy")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training and Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True)

    plt.xlabel("Epochs")
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Enhanced training history plot saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    main()
