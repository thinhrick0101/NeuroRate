import torch
import torch.nn as nn
import math
from bert_tokenizer import WordPieceTokenizer


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"

    def __init__(
        self,
        max_sequence_length,
        d_model,
        language_to_index,
        START_TOKEN,
        END_TOKEN,
        PADDING_TOKEN,
    ):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PosEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.2)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN

    def batch_tokenize(self, batch, start_token, end_token):
        """Tokenize a batch of sentences at character level"""
        tokenized = []
        for sentence in batch:
            # Convert sentence to list of character tokens
            sentence_tokens = [
                self.language_to_index.get(
                    token, self.language_to_index.get(self.PADDING_TOKEN)
                )
                for token in list(sentence)
            ]

            # Add special tokens if requested
            if start_token:
                sentence_tokens.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_tokens.append(self.language_to_index[self.END_TOKEN])

            # Pad or truncate to max sequence length
            if len(sentence_tokens) > self.max_sequence_length:
                sentence_tokens = sentence_tokens[: self.max_sequence_length]
            else:
                padding = [self.language_to_index[self.PADDING_TOKEN]] * (
                    self.max_sequence_length - len(sentence_tokens)
                )
                sentence_tokens.extend(padding)

            tokenized.append(torch.tensor(sentence_tokens))

        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())

    def forward(self, x, start_token, end_token):
        """
        Process input text into embeddings with positional encoding

        Args:
            x: Batch of input sentences
            start_token: Whether to add start token
            end_token: Whether to add end token

        Returns:
            Embedded representation with positional encoding
        """
        # Tokenize the input
        x = self.batch_tokenize(x, start_token, end_token)

        # Apply embedding
        x = self.embedding(x)

        # Create query/key tensors for positional encoding
        batch_size, seq_len = x.shape[:2]
        q = k = x.view(batch_size, seq_len, 1, -1)  # Add head dimension

        # Apply positional encoding
        q_rotated, k_rotated = self.position_encoder(q, k)

        # Reshape back and use the positional encoded queries
        x = q_rotated.squeeze(2)  # Remove head dimension

        # Apply dropout
        x = self.dropout(x)

        return x


class BERTSentenceEmbedding(nn.Module):
    """BERT-style sentence embedding with custom tokenizer"""

    def __init__(self, max_sequence_length, d_model, vocab_file=None, corpus=None):
        super().__init__()

        # Initialize the tokenizer
        self.tokenizer = WordPieceTokenizer(vocab_file)
        if corpus and not vocab_file:
            # Build vocabulary if corpus provided and no vocab file
            self.tokenizer.build_vocab(corpus)

        # Set constants for compatibility
        self.max_sequence_length = max_sequence_length
        self.START_TOKEN = self.tokenizer.cls_token
        self.END_TOKEN = self.tokenizer.sep_token
        self.PADDING_TOKEN = self.tokenizer.pad_token

        # Initialize embeddings
        self.vocab_size = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(self.vocab_size, d_model)

        # Position encoding and dropout
        self.position_encoder = PosEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.2)

        # Map for vocabulary lookup
        self.language_to_index = self.tokenizer.vocab

    def batch_tokenize(self, batch, start_token, end_token):
        """Tokenize a batch of sentences"""
        tokenized = []
        for sentence in batch:
            # Use tokenizer to encode text
            token_ids = self.tokenizer.encode(
                sentence,
                max_length=self.max_sequence_length,
                add_special_tokens=start_token or end_token,
            )
            tokenized.append(torch.tensor(token_ids))

        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())

    def forward(self, x, start_token, end_token):
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)

        # Create query/key tensors for position encoding
        batch_size, seq_len = x.shape[:2]
        q = k = x.view(batch_size, seq_len, 1, -1)  # Add head dimension

        # Apply positional encoding
        q_rotated, k_rotated = self.position_encoder(q, k)

        # Reshape back and use the positional encoded queries
        x = q_rotated.squeeze(2)  # Remove head dimension

        x = self.dropout(x)
        return x


class PosEncoding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) Module.

    Args:
        head_dim (int): The dimension of each attention head.
        seq_len (int): Maximum sequence length.
    """

    def __init__(self, head_dim, seq_len):
        super(PosEncoding, self).__init__()
        self.head_dim = head_dim
        self.seq_len = seq_len

        # Register buffers for sin and cos embeddings
        self.register_buffer("sin_emb", None)
        self.register_buffer("cos_emb", None)
        self.init_rope_matrix()

    def init_rope_matrix(self):
        # Create frequency tensor based on head dimension
        theta = 10000.0 ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim)

        # Create position tensor
        pos = torch.arange(self.seq_len).float().unsqueeze(1)  # [seq_len, 1]

        # Compute angle for each position and frequency
        angles = pos * theta  # Shape: [seq_len, head_dim/2]

        # Compute sin and cos for interleaved format
        sin_emb = torch.sin(angles).repeat_interleave(2, dim=1)  # [seq_len, head_dim]
        cos_emb = torch.cos(angles).repeat_interleave(2, dim=1)  # [seq_len, head_dim]

        self.sin_emb = sin_emb
        self.cos_emb = cos_emb

    def forward(self, q, k):
        """
        Apply RoPE to queries and keys.

        Args:
            q, k: Query and Key tensors, shape [batch, seq_len, num_heads, head_dim]

        Returns:
            Rotated q, k
        """
        batch_size, seq_len, num_heads, head_dim = q.shape
        assert (
            head_dim == self.head_dim
        ), f"Head dim mismatch! Expected {self.head_dim}, got {head_dim}"
        assert (
            seq_len <= self.seq_len
        ), f"Sequence length exceeds maximum! Expected <= {self.seq_len}, got {seq_len}"

        # Ensure correct shape for sin and cos embeddings
        sin_emb = (
            self.sin_emb[:seq_len].unsqueeze(0).unsqueeze(2)
        )  # [1, seq_len, 1, head_dim]
        cos_emb = (
            self.cos_emb[:seq_len].unsqueeze(0).unsqueeze(2)
        )  # [1, seq_len, 1, head_dim]

        # Even indices: x, Odd indices: y
        q_even, q_odd = q[..., ::2], q[..., 1::2]  # Split into even-odd pairs
        k_even, k_odd = k[..., ::2], k[..., 1::2]

        # RoPE transformation
        q_rotated = torch.cat(
            [
                q_even * cos_emb - q_odd * sin_emb,  # Real part
                q_odd * cos_emb + q_even * sin_emb,  # Imaginary part
            ],
            dim=-1,
        )

        k_rotated = torch.cat(
            [
                k_even * cos_emb - k_odd * sin_emb,  # Real part
                k_odd * cos_emb + k_even * sin_emb,  # Imaginary part
            ],
            dim=-1,
        )

        return q_rotated, k_rotated


class ReLU(nn.Module):
    def __init__(self):
        """Initialize the ReLU activation function."""
        super(ReLU, self).__init__()

    def forward(self, x):
        """Apply ReLU element-wise: max(0, x)."""
        return torch.maximum(x, torch.tensor(0.0, device=x.device))


class FFN(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(
            d_model, d_model * 4
        )  # This ratio (4) is an empirical best practice in deep learning.
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.relu = ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.2):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0  # Ensure d_model is divisible by num_heads
        self.head_dim = d_model // num_heads
        # self.dropout = nn.Dropout(dropout)

        # Linear transformations for Q, K, V
        self.qkv = nn.Linear(d_model, d_model * 3)

        # Output linear transformation
        self.linear_out = nn.Linear(d_model, d_model)

        # Scaled dot-product attention
        self.scale = 1 / math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        assert C % self.num_heads == 0, "d_model must be divisible by num_heads"
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(
            1, 2
        )  # [B, num_heads, T, head_dim]
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(
            1, 2
        )  # [B, num_heads, T, head_dim]
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(
            1, 2
        )  # [B, num_heads, T, head_dim]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        x = torch.matmul(attention, v)

        # Concatenate and linear transformation
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        x = self.linear_out(x)

        return x


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    RMSNorm is a simplified version of LayerNorm that:
    1. Doesn't require mean centering (no mean subtraction)
    2. Only uses RMS (Root Mean Square) statistics
    3. Has better numerical stability and computational efficiency

    Paper: "Root Mean Square Layer Normalization" (https://arxiv.org/abs/1910.07467)

    Args:
        dim (int): The feature dimension to normalize over
        eps (float): A small constant for numerical stability
    """

    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        # Learnable scale parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Calculate the RMS value per sample across features
        # Square all elements, take mean, then sqrt (Root Mean Square)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Apply RMSNorm to input tensor

        Args:
            x: Input tensor of shape [..., dim]

        Returns:
            Normalized tensor of the same shape
        """
        # Normalize and scale with learned parameters
        normalized = self._norm(x)
        # Apply learnable parameters (element-wise multiplication)
        return normalized * self.weight


class EncodingLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.2):
        super(EncodingLayer, self).__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask, pre_norm=True):
        if pre_norm:
            # Pre-norm version (more stable)
            residual = x
            x = self.norm1(x)
            x = self.attn(x, x, x, mask)
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.norm2(x)
            x = self.ffn(x)
            x = self.dropout2(x)
            x = residual + x
        else:
            # Post-norm version (original)
            residual = x
            x = self.attn(x, x, x, mask)
            x = self.dropout1(x)
            x = self.norm1(x + residual)

            residual = x
            x = self.ffn(x)
            x = self.dropout2(x)
            x = self.norm2(x + residual)
        return x


class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        drop_prob,
        num_layers,
        max_sequence_length,
        language_to_index=None,
        START_TOKEN=None,
        END_TOKEN=None,
        PADDING_TOKEN=None,
        use_bert_tokenization=False,
        corpus=None,
        vocab_file=None,
    ):
        super().__init__()

        if use_bert_tokenization:
            self.sentence_embedding = BERTSentenceEmbedding(
                max_sequence_length, d_model, vocab_file, corpus
            )
            # Get special tokens from the tokenizer
            START_TOKEN = self.sentence_embedding.START_TOKEN
            END_TOKEN = self.sentence_embedding.END_TOKEN
            PADDING_TOKEN = self.sentence_embedding.PADDING_TOKEN
        else:
            self.sentence_embedding = SentenceEmbedding(
                max_sequence_length,
                d_model,
                language_to_index,
                START_TOKEN,
                END_TOKEN,
                PADDING_TOKEN,
            )

        self.layers = SequentialEncoder(
            *[EncodingLayer(d_model, num_heads, drop_prob) for _ in range(num_layers)]
        )

    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x
