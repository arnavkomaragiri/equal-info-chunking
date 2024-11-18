import torch

import numpy as np

from .base import BaseChunkingStrategy
from .constants import LOG_2
from typeguard import typechecked
from typing import Optional
from torchtyping import TensorType, patch_typeguard

patch_typeguard()

class RandomChunkingStrategy(BaseChunkingStrategy):
    def __init__(self, *args, seed: Optional[int] = None, weight: Optional[float] = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(seed=seed)
        self.max_rand_size = weight * self.max_chunk_size
        self.token_sizes = []
    
    @typechecked
    def get_chunk_size(self, tokens: TensorType["seq"], logits: TensorType["seq", "dim"], start_ptr: int, end_ptr: int) -> float:
        if len(self.token_sizes) < tokens.shape[0]:
            old_sizes = self.token_sizes
            self.token_sizes = [None for _ in tokens]
            for i, s in enumerate(old_sizes):
                self.token_sizes[i] = s
        base_ptr = start_ptr
        while base_ptr < len(self.token_sizes) and self.token_sizes[base_ptr] != None:
            base_ptr += 1
        if base_ptr != len(self.token_sizes):
            rand_token_sizes = self.rng.uniform(0, self.max_rand_size, end_ptr - base_ptr)
            for i, s in enumerate(rand_token_sizes):
                self.token_sizes[i + base_ptr] = float(s)
        return sum(self.token_sizes[start_ptr:end_ptr])