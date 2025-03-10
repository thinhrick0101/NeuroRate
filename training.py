import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
import logging
from tqdm import tqdm
import numpy as np
import random
from model import Encoder
from model_config import Config
from bert_tokenizer import WordPieceTokenizer
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        labels = torch.tensor([item["label"] for item in batch], device=device)
        batch_dict["labels"] = labels

    return batch_dict


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Get inputs
        texts = batch["texts"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels")

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(texts, attention_mask, start_token=True, end_token=True)

        # If we have labels, calculate loss
        if labels is not None:
            loss = criterion(outputs, labels)
        else:
            # For unsupervised learning, you might use a different loss function
            # This is just a placeholder
            loss = criterion(outputs)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{total_loss/(progress_bar.n+1):.4f}"})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

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
            else:
                loss = criterion(outputs)

            # Update metrics
            total_loss += loss.item()

    return total_loss / len(dataloader)


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


def main():
    # Set random seed
    set_seed(42)

    # Configuration
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading
    # Replace with your actual data loading logic
    train_texts = ["This is a sample text", "Another sample"]
    train_labels = [1, 0]  # For classification tasks

    val_texts = ["A validation sample", "Second validation sample"]
    val_labels = [0, 1]

    # Create datasets
    train_dataset = TextDataset(
        train_texts, train_labels, max_length=config.max_position_embeddings
    )
    val_dataset = TextDataset(
        val_texts, val_labels, max_length=config.max_position_embeddings
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(
            batch,
            max_length=config.max_position_embeddings,
            device=device,
            config=config,
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(
            batch,
            max_length=config.max_position_embeddings,
            device=device,
            config=config,
        ),
    )

    # Initialize tokenizer
    tokenizer = WordPieceTokenizer(vocab_size=config.vocab_size)

    # Build vocabulary from training corpus if needed
    # tokenizer.build_vocab(train_texts)

    # Calculate number of classes from labels
    num_classes = len(set(train_labels))
    logger.info(f"Detected {num_classes} classes in dataset")

    # Model initialization
    model = Encoder(
        d_model=config.hidden_size,
        num_heads=config.num_heads,
        drop_prob=config.hidden_dropout_prob,
        num_layers=config.Layer,
        max_sequence_length=config.max_position_embeddings,
        use_bert_tokenization=True,
        corpus=train_texts,  # Build vocabulary from training corpus
        num_classes=num_classes,  # Pass number of classes
    ).to(device)

    # Optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    # Add learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # Setup for early stopping
    early_stopping_patience = 5
    early_stopping_counter = 0

    # Training settings
    num_epochs = 10
    checkpoint_path = "checkpoints/best_model.pt"
    history_path = "checkpoints/training_history.png"

    # Try to resume from checkpoint
    model, optimizer, scheduler, start_epoch, best_val_loss, history = load_checkpoint(
        model, optimizer, scheduler, checkpoint_path
    )

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.4f}")

        # Evaluate
        val_loss = evaluate(model, val_loader, criterion, device)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Validation loss: {val_loss:.4f}")

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

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

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, history, checkpoint_path
            )
            logger.info(f"New best validation loss: {val_loss:.4f}")
        else:
            early_stopping_counter += 1
            logger.info(
                f"Validation loss did not improve. Counter: {early_stopping_counter}/{early_stopping_patience}"
            )

            if early_stopping_counter >= early_stopping_patience:
                logger.info("Early stopping triggered!")
                break

    # Plot and save training history
    plot_training_history(history, history_path)

    logger.info("Training completed!")
    return best_val_loss


if __name__ == "__main__":
    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    main()
