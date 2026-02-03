import numpy as np
from typing import List, Dict


class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0

        # Special tokens with fixed IDs
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        # Initialize with special tokens immediately
        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        """Initialize special tokens with fixed IDs."""
        special_tokens = [
            (self.pad_token, 0),
            (self.unk_token, 1),
            (self.bos_token, 2),
            (self.eos_token, 3)
        ]
        
        for token, token_id in special_tokens:
            self.word_to_id[token] = token_id
            self.id_to_word[token_id] = token
        
        self.vocab_size = 4

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # Start with next available ID after special tokens
        next_id = 4
        
        for text in texts:
            if not text.strip():
                continue
                
            # Split by whitespace and clean up
            words = text.split()
            for word in words:
                # Clean up the word (remove extra whitespace)
                word = word.strip()
                if not word:
                    continue
                    
                # Add to vocabulary if not already present and not a special token
                if word not in self.word_to_id:
                    self.word_to_id[word] = next_id
                    self.id_to_word[next_id] = word
                    next_id += 1
        
        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        if not text.strip():
            return []
            
        result = []
        words = text.split()
        
        for word in words:
            word = word.strip()
            if not word:
                continue
                
            # Use UNK token (ID=1) for unknown words
            token_id = self.word_to_id.get(word, 1)  # UNK token ID is 1
            result.append(token_id)
            
        return result

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        Skip special tokens except to handle unknown words.
        """
        words = []
        
        for token_id in ids:
            # Skip special tokens (except we need to handle UNK somehow)
            if token_id in [0, 2, 3]:  # PAD, BOS, EOS
                continue
            elif token_id == 1:  # UNK
                words.append("<UNK>")
            else:
                word = self.id_to_word.get(token_id, "<UNK>")
                words.append(word)
        
        return " ".join(words)

    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary mapping."""
        return self.word_to_id.copy()