import torch
class Config():
    Layer: int = 12
    num_heads: int = 12
    hidden_size: int = 768
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    vocab_size: int = 30522
    pad_token_id: int = 0
    cls_token_id: int = 101
    sep_token_id: int = 102