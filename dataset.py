import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from config import Config
from tokenizer import WordPieceTokenizer
import torch.multiprocessing as mp

from process import process_dataset


def worker_init_fn(worker_id):
    """Function to initialize each worker"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class AmazonReviewDataset(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_sequence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        review_text = item["text"]  # Changed from "reviewText" to "text"

        # Rating is 0-indexed in the model but 1-5 in the dataset
        rating = item["rating"] - 1  # Changed from "overall" to "rating"
        
        # (1,2) - (3) - (4,5)
        # rating 1,2 = 0
        # rating 3 = 1
        # rating 4,5 = 2
        
        if (rating == 1 or rating == 0):
            rating = 0
        elif (rating == 2):
            rating = 1
        elif (rating == 3 or rating == 4):
            rating = 2

        try:
            # Directly use the tokenizer (not tokenizer.tokenizer)
            tokens = self.tokenizer(
                review_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            return {
                "input_ids": tokens["input_ids"].squeeze(),
                "attention_mask": tokens["attention_mask"].squeeze(),
                "labels": torch.tensor(rating, dtype=torch.long),
            }

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Fallback with an empty tensor of appropriate size
            return {
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
                "labels": torch.tensor(rating, dtype=torch.long),
            }


def create_dataloaders(config):
    # Load the dataset with trust_remote_code=True
    print(f"Loading dataset: {config.dataset_name}")
    dataset = load_dataset("nanaut/MLVU144", trust_remote_code=True)

    # # Print dataset structure
    print("Dataset structure:", dataset)
    print("Available splits:", list(dataset.keys()))

    # # Select columns and limit number of examples
    dataset = dataset["train"].select_columns(["text", "rating"])
    
    ds_raw = process_dataset(dataset, config)    
    
    print(f"Selected {len(ds_raw)} examples from 'full' split")

    # Train a tokenizer or load a pre-trained one
    tokenizer_handler = WordPieceTokenizer(config)
    try:
        tokenizer = tokenizer_handler.load()
        print("Loaded pre-trained tokenizer")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Training a new tokenizer...")
        # Use all available reviews for tokenizer training
        all_texts = [item["text"] for item in ds_raw]
        print(f"Training tokenizer on {len(all_texts)} reviews")
        tokenizer = tokenizer_handler.train(all_texts)
        print("Tokenizer training completed")

    # Create a dataset from the selected data
    full_dataset = AmazonReviewDataset(ds_raw, tokenizer, config)

    # Split into train and validation sets
    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    print(f"Splitting into {train_size} training and {val_size} validation samples")
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Create data loaders with reduced number of workers and smaller batch size to save memory
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,  # Reduced from 4 to 2
        worker_init_fn=worker_init_fn,
        multiprocessing_context=mp.get_context("fork"),
        pin_memory=False,  # Don't use pinned memory to save GPU memory
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,  # Reduced from 4 to 2
        worker_init_fn=worker_init_fn,
        multiprocessing_context=mp.get_context("fork"),
        pin_memory=False,  # Don't use pinned memory to save GPU memory
    )

    return train_dataloader, val_dataloader, tokenizer