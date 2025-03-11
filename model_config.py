import torch


class Config:
    Layer: int = 6  # Increase from 4 to 6 layers for more capacity
    num_heads: int = 8  # Increase from 4 to 8 heads for better attention
    hidden_size: int = 384  # Increase from 256 to 384 for more representation power
    intermediate_size: int = 1536  # Set to 4x hidden_size (best practice)
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.2  # Reduce from 0.3 for better convergence
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    vocab_size: int = 30522
    pad_token_id: int = 0
    cls_token_id: int = 101
    sep_token_id: int = 102
