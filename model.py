import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias


class EmbeddingLayer(nn.Module):
    def __init__(self, config):
        super(EmbeddingLayer, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_sequence_length, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        self.layer_norm = LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} not divisible by number of attention heads {config.num_attention_heads}"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.output(context_layer)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layernorm1 = LayerNorm(config.hidden_size)
        self.layernorm2 = LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout(attention_output)
        attention_output = self.layernorm1(hidden_states + attention_output)

        intermediate_output = self.intermediate(attention_output)
        intermediate_output = F.gelu(intermediate_output)

        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.layernorm2(attention_output + layer_output)

        return layer_output


class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.embeddings = EmbeddingLayer(config)
        self.encoder = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )

        # Classifier for star rating prediction (1-5)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is not None:
            # Create attention mask for self-attention
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        embedding_output = self.embeddings(input_ids, token_type_ids)

        hidden_states = embedding_output
        for layer in self.encoder:
            hidden_states = layer(hidden_states, extended_attention_mask)

        # Use [CLS] token representation for classification
        pooled_output = hidden_states[:, 0]
        pooled_output = self.pooler(pooled_output)
        pooled_output = torch.tanh(pooled_output)
        pooled_output = self.dropout(pooled_output)

        # Predict star rating (1-5)
        logits = self.classifier(pooled_output)

        return logits