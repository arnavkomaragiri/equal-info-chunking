# torch import
import torch

# transformers import
from transformers import AutoModelForCasualLM, AutoTokenizer, GenerationConfig
from transformers.generation.utils import ModelOutput, GenerateDecoderOnlyOutput

# type checking
from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard
from typing import Iterable, Callable, Optional

# patch typeguard to use torchtyping
patch_typeguard()

# base chunking strategy with batch processing
class BaseChunkingStrategy:
    @typechecked
    def on_token_generate(tokens: TensorType["seq"], logits: TensorType["seq", "dim"]) -> bool:
        raise NotImplementedError("token generation event function not implemented")

    def on_sequence_end():
        raise NotImplementedError("sequence end event function not implemented")

    def get_gen_config() -> GenerationConfig:
        return GenerationConfig(
            use_cache=True,
            output_logits=True,
            return_dict_in_generate=True
        ) 
    
    def iterate(self, model: AutoModelForCasualLM, tokenizer: AutoTokenizer, prompt: str, generate: bool = True):
        inputs = tokenizer(prompt, return_tensors="pt")
        if generate:
            generation_config = self.get_gen_config()
            outputs: GenerateDecoderOnlyOutput = model.generate(
                **inputs,
                generation_config=generation_config
            )
            assert outputs.logits.shape[0] == 1, f"expected single batch dimension output, received dimension {outputs.logits.shape[0]}"
            tokens, logits = outputs.sequences[0], outputs.logits[0]
        else:
            with torch.no_grad():
                outputs: torch.Tensor = model(**inputs, labels=inputs["input_ids"])
            assert outputs.logits.shape[0] == 1, f"expected single batch dimension output, received dimension {outputs.logits.shape[0]}"
            tokens, logits = inputs["input_ids"][0], outputs.logits[0]
        
        base = 0
        for i in range(1, logits.shape[0]):
            tok, log = tokens[i].reshape(1, -1), logits[i - 1, :].reshape(1, -1)
            chunk = self.on_token_generate(tok, log)
            if chunk:
                yield tokens[base:i]
                base = i

        try:
            self.on_sequence_end()
        except NotImplementedError:
            # if we don't have a sequence end method implemented don't worry about it
            pass

        
        

