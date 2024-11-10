# torch import
import torch

# transformers import
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation.utils import ModelOutput

# type checking
from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard
from typing import Iterable, Callable, Optional

# patch typeguard to use torchtyping
patch_typeguard()

# base chunking strategy with batch processing
class BaseChunkingStrategy:
    def __init__(self, max_chunk_size: float):
        self.max_chunk_size = max_chunk_size
        
    @typechecked
    def get_chunk_size(self, tokens: TensorType["seq"], logits: TensorType["seq", "dim"], start_ptr: int, end_ptr: int) -> bool:
        raise NotImplementedError("chunk sizing function not implemented")

    def on_sequence_end(self):
        raise NotImplementedError("sequence end event function not implemented")

    def get_gen_config(self) -> GenerationConfig:
        return GenerationConfig(
            use_cache=True,
            output_logits=True,
            return_dict_in_generate=True
        ) 
    
    def iterate(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, generate: bool = True, yield_trailing_chunk: bool = False) -> Iterable:
        inputs = tokenizer(prompt, return_tensors="pt")
        if generate:
            generation_config = self.get_gen_config()
            outputs: ModelOutput = model.generate(
                **inputs,
                generation_config=generation_config
            )
            tokens = outputs.sequences
            logits = torch.stack(*outputs.logits).permute(1, 0, 2)
        else:
            with torch.no_grad():
                outputs: torch.Tensor = model(**inputs, labels=inputs["input_ids"])
            tokens = inputs["input_ids"]
            start_logit = torch.full((outputs.logits.shape[0], 1, outputs.logits.shape[-1]), -torch.inf)
            batch_idx = torch.arange(outputs.logits.shape[0])
            start_logit[batch_idx, 0, tokens[batch_idx, 0]] = 1 
            logits = torch.cat((start_logit, outputs.logits[:, :-1, :]), dim=1)

        assert tokens.shape[0] == 1, f"expected single batch dimension tokens, received dimension {tokens.shape[0]}"
        assert logits.shape[0] == 1, f"expected single batch dimension logits, received dimension {logits.shape[0]}"

        # TODO: figure out how to get batch processing working
        tokens, logits = tokens[0], logits[0]

        if tokens.shape[0] <= 1:
            yield tokens
        else: 
            base, chunk_size = 0, 0
            for i in range(1, logits.shape[0]):
                chunk_size = self.get_chunk_size(tokens, logits, base, i)
                if chunk_size > self.max_chunk_size:
                    yield tokens[base:i]
                    base = i
            if yield_trailing_chunk and base != logits.shape[0] - 1:
                yield tokens[base:]

        try:
            self.on_sequence_end()
        except NotImplementedError:
            # if we don't have a sequence end method implemented don't worry about it
            pass

        
        

