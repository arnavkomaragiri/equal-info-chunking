import gzip
import torch

from .base import BaseChunkingStrategy
from transformers import PreTrainedTokenizer
from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard
from typing import Union

patch_typeguard()

def get_gzipped_size(input_string: Union[str, bytes]) -> int:
    """
    Calculate the size in bytes of a string after gzip compression.

    Args:
        input_string: The input string or bytes object to be compressed.
                     If a string is provided, it will be encoded as UTF-8.

    Returns:
        int: Size of the compressed data in bytes

    Raises:
        TypeError: If input is neither string nor bytes
        ValueError: If compression fails
    """
    try:
        # Convert string to bytes if necessary
        if isinstance(input_string, str):
            data = input_string.encode('utf-8')
        elif isinstance(input_string, bytes):
            data = input_string
        else:
            raise TypeError("Input must be string or bytes")

        # Compress the data
        compressed = gzip.compress(data)
        return len(compressed)

    except Exception as e:
        raise ValueError(f"Failed to compress data: {str(e)}")

class GZIPChunkingStrategy(BaseChunkingStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @typechecked
    def get_chunk_size(self, tokens: TensorType["seq"], logits: TensorType["seq", "dim"], start_ptr: int, end_ptr: int, tokenizer: PreTrainedTokenizer) -> float:
        long_string, short_string = tokenizer.batch_decode([tokens[:end_ptr+1], tokens[:start_ptr]])
        return get_gzipped_size(long_string) - get_gzipped_size(short_string)