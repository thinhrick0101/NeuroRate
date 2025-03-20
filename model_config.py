import torch


class Config:
    Layer: int = 4  # Increase from 4 to 6 layers for more capacity
    num_heads: int = 4  # Increase from 4 to 8 heads for better attention
    hidden_size: int = 128  # Increase from 256 to 384 for more representation power
    hidden_dropout_prob: float = 0.2  # Reduce from 0.3 for better convergence
    vocab_size: int = 30522  # Keep the same as BERT
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
