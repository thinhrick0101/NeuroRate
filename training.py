import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from config import Config
from model import BERT

# Import mixed precision training if available
try:
    from torch.cuda.amp import autocast, GradScaler

    AMP_AVAILABLE = torch.cuda.is_available()
except ImportError:
    AMP_AVAILABLE = False


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles=0.5,
    min_lr_ratio=0.0,
    last_epoch=-1,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * num_cycles * 2.0 * progress))

        # Adjust to respect minimum learning rate
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    min_lr_ratio=0.0,
    power=1.0,
    last_epoch=-1,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        remaining = (1.0 - progress) ** power

        # Scale from 1.0 to min_lr_ratio
        return min_lr_ratio + (1.0 - min_lr_ratio) * remaining

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_scheduler(optimizer, config, num_training_steps):
    """Returns a learning rate scheduler based on config settings."""
    # Calculate warmup steps either from fixed value or ratio
    if config.use_warmup_ratio:
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    else:
        num_warmup_steps = config.warmup_steps

    scheduler_type = config.lr_scheduler_type.lower()

    if scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
            num_training_steps,
            min_lr_ratio=config.min_lr_ratio,
        )
    elif scheduler_type == "polynomial":
        return get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
            num_training_steps,
            min_lr_ratio=config.min_lr_ratio,
        )
    elif scheduler_type == "constant":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    else:
        raise ValueError(f"Scheduler type '{config.lr_scheduler_type}' not supported.")


def train_model(model, train_dataloader, val_dataloader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        # Print GPU memory info
        print(
            f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
        )
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        # Set memory optimization
        torch.cuda.empty_cache()
        gc.collect()

    # Initialize mixed precision training if available
    use_amp = AMP_AVAILABLE and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision training")

    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Configure optimizer with parameter groups for different learning rates
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Learning rate scheduler
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    total_steps = (
        len(train_dataloader) // config.gradient_accumulation_steps
    ) * config.num_epochs

    scheduler = get_scheduler(optimizer, config, total_steps)

    # Training metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1s = []
    val_f1s = []
    learning_rates = []
    best_val_accuracy = 0
    best_val_f1 = 0
    no_improvement_count = 0

    # Create directories for saving models
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        optimizer.zero_grad()  # Zero gradients at the beginning of epoch
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current learning rate: {current_lr:.6f}")

        # Training loop
        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass with mixed precision if available
                if use_amp:
                    with autocast():
                        outputs = model(
                            input_ids=input_ids, attention_mask=attention_mask
                        )
                        loss = (
                            criterion(outputs, labels)
                            / config.gradient_accumulation_steps
                        )
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = (
                        criterion(outputs, labels) / config.gradient_accumulation_steps
                    )

                epoch_loss += loss.item() * config.gradient_accumulation_steps

                # Backward pass with mixed precision if available
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Get predictions
                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().detach().numpy())
                all_labels.extend(labels.cpu().detach().numpy())

                # Update progress bar
                progress_bar.set_postfix(
                    {"loss": loss.item() * config.gradient_accumulation_steps}
                )

                # Gradient accumulation: only update every config.gradient_accumulation_steps
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    if use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.max_grad_norm
                        )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.max_grad_norm
                        )
                        optimizer.step()

                    scheduler.step()
                    learning_rates.append(scheduler.get_last_lr()[0])
                    optimizer.zero_grad()

                    # Free up memory
                    del input_ids, attention_mask, labels, outputs, loss, preds
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        # Calculate epoch metrics
        epoch_loss = epoch_loss / len(train_dataloader)
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average="weighted")

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        train_f1s.append(epoch_f1)

        print(
            f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, F1: {epoch_f1:.4f}"
        )

        # Validation
        val_loss, val_accuracy, val_f1 = evaluate_model(
            model, val_dataloader, criterion, device, config, use_amp
        )
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_f1s.append(val_f1)

        print(
            f"Epoch {epoch+1}/{config.num_epochs} - Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}"
        )

        # Save the best model based on validation accuracy
        is_best = False
        if val_accuracy > best_val_accuracy + config.early_stopping_threshold:
            best_val_accuracy = val_accuracy
            best_val_f1 = max(best_val_f1, val_f1)
            no_improvement_count = 0
            is_best = True
            torch.save(
                model.state_dict(),
                os.path.join(config.model_save_dir, f"bert_star_rating_best.pt"),
            )
            print(
                f"New best model saved with validation accuracy: {best_val_accuracy:.4f}, F1: {val_f1:.4f}"
            )
        else:
            no_improvement_count += 1
            print(
                f"No improvement for {no_improvement_count} epochs. Best validation accuracy: {best_val_accuracy:.4f}, F1: {best_val_f1:.4f}"
            )
            if (
                config.early_stopping
                and no_improvement_count >= config.early_stopping_patience
            ):
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Save the model checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accuracies": train_accuracies,
                "val_accuracies": val_accuracies,
                "train_f1s": train_f1s,
                "val_f1s": val_f1s,
                "best_val_accuracy": best_val_accuracy,
                "best_val_f1": best_val_f1,
            },
            os.path.join(config.model_save_dir, f"bert_star_rating_checkpoint.pt"),
        )

        # Generate and save confusion matrix after each epoch
        if is_best:
            try:
                generate_confusion_matrix(
                    model, val_dataloader, device, config, epoch, use_amp
                )
            except Exception as e:
                print(f"Error generating confusion matrix: {e}")

        # Clean up memory
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # Plot training and validation metrics
    try:
        plot_metrics(
            train_losses,
            val_losses,
            train_accuracies,
            val_accuracies,
            learning_rates,
            train_f1s,
            val_f1s,
            config,
        )
    except Exception as e:
        print(f"Error plotting metrics: {e}")

    return model


def evaluate_model(model, dataloader, criterion, device, config, use_amp=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    batch_count = 0  # Keep track of processed batches

    with torch.no_grad():
        for batch in dataloader:
            try:
                # Process in smaller chunks to avoid OOM
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass with mixed precision if available
                if use_amp:
                    with autocast():
                        outputs = model(
                            input_ids=input_ids, attention_mask=attention_mask
                        )
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, labels)

                total_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                batch_count += 1

                # Clean up GPU memory
                del input_ids, attention_mask, labels, outputs, loss, preds
                if device.type == "cuda" and batch_count % 10 == 0:  # Every 10 batches
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in evaluation batch: {e}")
                continue

    avg_loss = total_loss / max(batch_count, 1)  # Avoid division by zero

    if len(all_labels) > 0 and len(all_preds) > 0:
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
    else:
        print("Warning: No predictions or labels available for evaluation")
        accuracy = 0.0
        f1 = 0.0

    return avg_loss, accuracy, f1


def generate_confusion_matrix(model, dataloader, device, config, epoch, use_amp=False):
    """Generate and save a confusion matrix to visualize predictions."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating confusion matrix"):
            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                if use_amp:
                    with autocast():
                        outputs = model(
                            input_ids=input_ids, attention_mask=attention_mask
                        )
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                del input_ids, attention_mask, labels, outputs, preds
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in confusion matrix batch: {e}")
                continue

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"{i+1} stars" for i in range(config.num_classes)],
        yticklabels=[f"{i+1} stars" for i in range(config.num_classes)],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Epoch {epoch+1})")
    plt.tight_layout()
    plt.savefig(os.path.join(config.log_dir, f"confusion_matrix_epoch_{epoch+1}.png"))
    plt.close()


def plot_metrics(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
    learning_rates=None,
    train_f1s=None,
    val_f1s=None,
    config=None,
):
    # Plot losses and accuracies
    plt.figure(figsize=(18, 6))

    # Plot losses
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    # Plot accuracies
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")

    # Plot F1 scores if available
    if train_f1s and val_f1s:
        plt.subplot(1, 3, 3)
        plt.plot(train_f1s, label="Training F1")
        plt.plot(val_f1s, label="Validation F1")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.title("Training and Validation F1 Score")

    plt.tight_layout()
    plt.savefig(os.path.join(config.log_dir, "training_metrics.png"))
    plt.close()

    # Add learning rate plot if available
    if learning_rates is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(learning_rates)
        plt.title("Learning Rate Schedule")
        plt.xlabel("Training Steps")
        plt.ylabel("Learning Rate")
        plt.savefig(os.path.join(config.log_dir, "learning_rate.png"))
        plt.close()