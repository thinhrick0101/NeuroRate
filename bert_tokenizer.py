import torch
import re
import collections
from typing import List, Dict, Tuple, Set, Optional

class WordPieceTokenizer:
    """
    A custom implementation of BERT-style WordPiece tokenization.
    """
    def __init__(self, vocab_file=None, vocab_size=30000, min_frequency=2,
                 special_tokens=None):
        # Special tokens
        self.special_tokens = special_tokens or {
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
            "unk_token": "[UNK]"
        }
        
        # Set attributes for compatibility with existing code
        self.cls_token = self.special_tokens["cls_token"]
        self.sep_token = self.special_tokens["sep_token"]
        self.pad_token = self.special_tokens["pad_token"]
        self.mask_token = self.special_tokens["mask_token"]
        self.unk_token = self.special_tokens["unk_token"]
        
        # Vocab params
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Token to ID mapping
        self.vocab = {}
        if vocab_file:
            self.load_vocab(vocab_file)
        else:
            # Initialize with special tokens
            for i, token in enumerate(self.special_tokens.values()):
                self.vocab[token] = i
                
        # ID to token mapping (reverse vocab)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
    def load_vocab(self, vocab_file: str):
        """Load vocabulary from file."""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                token = line.strip()
                self.vocab[token] = i
                
    def save_vocab(self, vocab_file: str):
        """Save vocabulary to file."""
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for token in self.vocab.keys():
                f.write(token + '\n')
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts using WordPiece algorithm."""
        # Preprocess all texts
        all_tokens = []
        for text in texts:
            all_tokens.extend(self._preprocess_text(text))
            
        # Count token frequencies
        token_counts = collections.Counter(all_tokens)
        
        # Start with whole words above frequency threshold
        vocab = {token for token, count in token_counts.items() 
                if count >= self.min_frequency}
        
        # Add special tokens
        for token in self.special_tokens.values():
            vocab.add(token)
            
        # Implement WordPiece algorithm
        # This is a simplified version - production implementations are more complex
        while len(vocab) < self.vocab_size:
            best_pair = None
            max_freq = 0
            
            # Find best pair to merge
            for token in list(token_counts.keys()):
                if token in vocab or len(token) < 3:
                    continue
                    
                for i in range(1, len(token)):
                    prefix = token[:i]
                    suffix = token[i:]
                    if prefix in vocab and suffix in vocab:
                        pair_freq = token_counts[token]
                        if pair_freq > max_freq:
                            max_freq = pair_freq
                            best_pair = (prefix, suffix, token)
            
            if best_pair is None or max_freq < self.min_frequency:
                break
                
            # Add merged token to vocab
            prefix, suffix, merged = best_pair
            vocab.add(merged)
            
            if len(vocab) >= self.vocab_size:
                break
                
        # Create final vocabulary
        self.vocab = {token: idx for idx, token in enumerate(list(self.special_tokens.values()) + 
                                                            sorted(vocab - set(self.special_tokens.values())))}
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for tokenization."""
        # Convert to lowercase
        text = text.lower()
        
        # Basic preprocessing similar to BERT
        text = re.sub(r"[^\w\s]", " ", text)  # Replace punctuation with space
        text = re.sub(r"\s+", " ", text)      # Replace multiple spaces with single space
        
        # Split by whitespace
        return text.strip().split()
    
    def encode(self, text: str, max_length: int = None, 
              add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        
        # Add special tokens
        if add_special_tokens:
            tokens = [self.cls_token] + tokens + [self.sep_token]
            
        # Convert tokens to IDs
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # Handle unknown tokens
                token_ids.append(self.vocab[self.unk_token])
        
        # Pad or truncate to max_length
        if max_length is not None:
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            else:
                token_ids.extend([self.vocab[self.pad_token]] * (max_length - len(token_ids)))
                
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = [self.ids_to_tokens.get(idx, self.unk_token) for idx in token_ids]
        
        # Remove special tokens
        tokens = [token for token in tokens if token not in self.special_tokens.values()]
        
        # Merge WordPiece tokens (simplified)
        text = " ".join(tokens)
        text = re.sub(r"\s##", "", text)  # Remove WordPiece markers
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using WordPiece algorithm."""
        words = self._preprocess_text(text)
        tokens = []
        
        for word in words:
            if word in self.vocab:
                tokens.append(word)
                continue
                
            # Try to split into subwords
            is_tokenized = False
            for end in range(len(word), 0, -1):
                prefix = word[:end]
                if prefix in self.vocab:
                    tokens.append(prefix)
                    suffix = word[end:]
                    if suffix:
                        # Mark continuation of word with ##
                        tokens.append("##" + suffix)
                    is_tokenized = True
                    break
                    
            if not is_tokenized:
                tokens.append(self.unk_token)
                
        return tokens