import gzip
import torch

from .base import BaseChunkingStrategy
from transformers import PreTrainedTokenizer
from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard
from typing import Union

patch_typeguard()

class GZIPChunkingStrategy(BaseChunkingStrategy):
    def __init__(self, tokenizer: PreTrainedTokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    def get_gzipped_size(self, input_string: str) -> int:
        data = input_string.encode('utf-8')
        return len(gzip.compress(data))
    
    @typechecked
    def get_chunk_size(self, tokens: TensorType["seq"], logits: TensorType["seq", "dim"], start_ptr: int, end_ptr: int) -> float:
        long_string, short_string = self.tokenizer.batch_decode([tokens[:end_ptr], tokens[:start_ptr]])
        return self.get_gzipped_size(long_string) - self.get_gzipped_size(short_string)