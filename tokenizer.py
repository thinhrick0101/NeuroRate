import os
import collections
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from config import Config


class WordPieceTokenizer:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None

    def train(self, texts=None):
        """Train a WordPiece tokenizer on the dataset"""
        print(
            f"Starting tokenizer training with vocabulary size: {self.config.vocab_size}"
        )

        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.decoder = decoders.WordPiece()

        # If texts are not provided, try to load them from the dataset
        if texts is None:
            print(
                "No texts provided for tokenizer training, attempting to load from dataset"
            )
            try:
                dataset = load_dataset(
                    self.config.dataset_name,
                    self.config.dataset_subset,
                    trust_remote_code=True,
                )
                print(f"Dataset loaded. Available splits: {list(dataset.keys())}")

                # Use the 'full' split and select the text column
                ds_raw = dataset["full"].select_columns(["text"])
                # Limit the number of examples for tokenizer training
                ds_raw = ds_raw.select(
                    range(min(self.config.max_examples, len(dataset["full"])))
                )
                texts = [item["text"] for item in ds_raw]
                print(
                    f"Using {len(texts)} texts from 'full' split for tokenizer training"
                )
            except Exception as e:
                print(f"Failed to load texts from dataset: {e}")
                raise ValueError("No texts provided and could not load from dataset")

        # Define special tokens
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        trainer = trainers.WordPieceTrainer(
            vocab_size=self.config.vocab_size, special_tokens=special_tokens
        )

        # Train the tokenizer
        print(f"Training tokenizer on {len(texts)} texts...")
        tokenizer.train_from_iterator(texts, trainer)
        print("Tokenizer training completed")

        # Add post-processor
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B [SEP]",
            special_tokens=[
                ("[CLS]", tokenizer.token_to_id("[CLS]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ],
        )

        # Save the tokenizer
        os.makedirs(self.config.tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(self.config.tokenizer_dir, "wordpiece.json")
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")

        # Create a Hugging Face compatible tokenizer
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )

        return self.tokenizer

    def load(self):
        """Load a pre-trained tokenizer"""
        tokenizer_path = os.path.join(self.config.tokenizer_dir, "wordpiece.json")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")

        print(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        return self.tokenizer

    def tokenize(
        self, text, padding=True, truncation=True, max_length=None, return_tensors=None
    ):
        """Tokenize a text using the WordPiece tokenizer"""
        # Remove this method as it's causing issues
        # Instead, use the HuggingFace tokenizer directly
        print(
            "Warning: Using deprecated tokenize method. Use the tokenizer directly instead."
        )

        if self.tokenizer is None:
            try:
                self.load()
            except Exception as e:
                print(f"Failed to load tokenizer: {e}")
                raise ValueError(
                    "Tokenizer not found. Please train the tokenizer first."
                )

        if max_length is None:
            max_length = self.config.max_sequence_length

        # Direct use of HuggingFace tokenizer
        return self.tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )