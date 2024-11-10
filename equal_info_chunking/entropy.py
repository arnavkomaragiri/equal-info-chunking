import torch

from .base import BaseChunkingStrategy
from .constants import LOG_2
from transformers import PreTrainedTokenizer
from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard

patch_typeguard()

class EntropyChunkingStrategy(BaseChunkingStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @typechecked
    def get_chunk_size(self, tokens: TensorType["seq"], logits: TensorType["seq", "dim"], start_ptr: int, end_ptr: int, tokenizer: PreTrainedTokenizer) -> float:
        # get target logits
        target_logits = logits[start_ptr:end_ptr]
        # get log probs and probs
        log_probs = torch.log_softmax(target_logits, dim=-1)
        probs = torch.exp(log_probs)
        # do entropy product computation
        prods = probs * log_probs
        # mask out zero entries in entropy tensor
        prods[probs == 0] = 0
        prods[log_probs == 0] = 0
        # compute final entropy
        return -torch.sum(prods).item() / LOG_2
