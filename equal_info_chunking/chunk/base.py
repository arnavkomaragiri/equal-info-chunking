# torch import
import torch

# transformers import
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer, BatchEncoding
from transformers.generation.utils import ModelOutput

from copy import deepcopy

# type checking
from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard
from typing import List, Dict, Tuple, Union, Iterable, ForwardRef

# patch typeguard to use torchtyping
patch_typeguard()

# set chat messages type alias for convenience
ChatMessages = List[Dict[str, str]]

# set forward reference for base chunking for return type hints
BaseChunkingRef = ForwardRef("BaseChunkingStrategy")
# base chunking strategy with batch processing
class BaseChunkingStrategy:
    def __init__(self, max_chunk_size: float):
        self.max_chunk_size = max_chunk_size

    @typechecked
    def parallel_infer(
        self, model: PreTrainedModel, inputs: BatchEncoding, **kwargs
    ) -> Tuple[TensorType["batch", "seq"], TensorType["batch", "seq", "dim"]]:
        with torch.no_grad():
            outputs: torch.Tensor = model(**inputs, labels=inputs["input_ids"])
        tokens = inputs["input_ids"]
        start_logit = torch.full((outputs.logits.shape[0], 1, outputs.logits.shape[-1]), -torch.inf)
        batch_idx = torch.arange(outputs.logits.shape[0])
        start_logit[batch_idx, 0, tokens[batch_idx, 0]] = 1 
        logits = torch.cat((start_logit, outputs.logits[:, :-1, :]), dim=1)
        return tokens, logits
    
    @typechecked
    def generate_infer(
        self, model: PreTrainedModel, inputs: BatchEncoding, **kwargs
    ) -> Tuple[TensorType["batch", "seq"], TensorType["batch", "seq", "dim"]]:
        generation_config = self.get_gen_config()
        outputs: ModelOutput = model.generate(
            **inputs,
            generation_config=generation_config,
            **kwargs
        )
        tokens = outputs.sequences[:, inputs['input_ids'].shape[1]:]
        logits = torch.stack(outputs.logits).permute(1, 0, 2)
        return tokens, logits

    @typechecked
    def calibrate(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            docs: List[Union[str, ChatMessages]],
            weight: float = 1
        ) -> BaseChunkingRef:
        max_chunk_size = 0
        for d in docs:
            if isinstance(d, list):
                d = tokenizer.apply_chat_template(
                    d,
                    tokenize=False,
                    add_generation_prompt=True
                )
            inputs = tokenizer(d, return_tensors="pt")
            tokens, logits = self.parallel_infer(model, inputs)
            max_chunk_size = max(max_chunk_size, max([self.get_chunk_size(tokens[0], logits[0], i, i + 1) for i in range(tokens.shape[1])]))
            
        new_strategy = deepcopy(self)
        new_strategy.max_chunk_size = weight * max_chunk_size
        return new_strategy
        
    @typechecked
    def get_chunk_size(
        self, 
        tokens: TensorType["seq"], logits: TensorType["seq", "dim"], 
        start_ptr: int, end_ptr: int,
    ) -> bool:
        raise NotImplementedError("chunk sizing function not implemented")

    def on_sequence_end(self):
        raise NotImplementedError("sequence end event function not implemented")

    def get_gen_config(self) -> GenerationConfig:
        return GenerationConfig(
            use_cache=True,
            output_logits=True,
            return_dict_in_generate=True
        ) 
    
    @typechecked
    def iterate(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            prompt: Union[str, ChatMessages],
            generate: bool = True,
            yield_trailing_chunk: bool = False,
            **kwargs
        ) -> Iterable:
        if isinstance(prompt, list):
            prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )
        inputs = tokenizer(prompt, return_tensors="pt")
        if generate:
            tokens, logits = self.generate_infer(model, inputs, **kwargs)
        else:
            tokens, logits = self.parallel_infer(model, inputs, **kwargs)

        assert tokens.shape[0] == 1, f"expected single batch dimension tokens, received dimension {tokens.shape[0]}"
        assert logits.shape[0] == 1, f"expected single batch dimension logits, received dimension {logits.shape[0]}"

        # don't support batch processing since the pad token handling gets overly complex
        tokens, logits = tokens[0], logits[0]

        if tokens.shape[0] <= 1:
            yield tokens
        else:
            base, chunk_size = 0, 0
            for i in range(1, logits.shape[0]):
                # get chunk size including current token
                chunk_size = self.get_chunk_size(tokens, logits, base, i + 1)
                # if chunk is larger than max chunk size, yield the chunk without the current token
                if chunk_size > self.max_chunk_size:
                    yield tokens[base:i]
                    base = i

            # if we have any trailing chunks and we want them, yield them
            if yield_trailing_chunk and base != logits.shape[0] - 1:
                yield tokens[base:]

        try:
            self.on_sequence_end()
        except NotImplementedError:
            # if we don't have a sequence end method implemented don't worry about it
            pass
