# torch import
import torch
# heap operations
import heapq

# transformers import
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizer, BatchEncoding
from transformers.modeling_outputs import BaseModelOutput
from transformers.generation.utils import ModelOutput

from copy import deepcopy

# type checking
from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard
from typing import List, Dict, Tuple, Union, Iterable, ForwardRef

# constants
from .constants import INF

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
            outputs: BaseModelOutput = model(**inputs, labels=inputs["input_ids"])
        tokens = inputs["input_ids"].cpu()
        raw_logits = outputs.logits.cpu()
        start_logit = torch.full((raw_logits.shape[0], 1, raw_logits.shape[-1]), -torch.inf)
        batch_idx = torch.arange(raw_logits.shape[0])
        start_logit[batch_idx, 0, tokens[batch_idx, 0]] = 1 
        logits = torch.cat((start_logit, raw_logits[:, :-1, :]), dim=1)
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
            inputs = inputs.to(model.device)
            tokens, logits = self.parallel_infer(model, inputs)
            token_sizes = [self.get_chunk_size(tokens[0], logits[0], i, i + 1) for i in range(tokens.shape[1])]
            token_sizes = [s for s in token_sizes if s < INF]
            doc_max = max(token_sizes)
            max_chunk_size = max(max_chunk_size, doc_max)
            
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

    def on_sequence_start(self):
        raise NotImplementedError("sequence start event function not implemented")

    def on_sequence_end(self):
        raise NotImplementedError("sequence end event function not implemented")

    def get_gen_config(self) -> GenerationConfig:
        return GenerationConfig(
            use_cache=True,
            output_logits=True,
            return_dict_in_generate=True
        ) 

    @typechecked
    def stream_chunks(self, tokens: TensorType["seq"], logits: TensorType["seq", "dim"], yield_trailing_chunk: bool = False):
        try:
            self.on_sequence_start()
        except NotImplementedError:
            # if we don't have a sequence start method implemented don't worry about it
            pass

        base, chunk_size = 0, 0
        for i in range(1, logits.shape[0]):
            # get chunk size including current token
            chunk_size = self.get_chunk_size(tokens, logits, base, i + 1)
            # if chunk is larger than max chunk size, yield the chunk without the current token
            if chunk_size > self.max_chunk_size:
                yield tokens[base:i], chunk_size
                base = i

        # if we have any trailing chunks and we want them, yield them
        if yield_trailing_chunk and base != logits.shape[0] - 1:
            chunk_size = self.get_chunk_size(tokens, logits, base, logits.shape[0])
            yield tokens[base:], chunk_size

        try:
            self.on_sequence_end()
        except NotImplementedError:
            # if we don't have a sequence end method implemented don't worry about it
            pass

    @typechecked
    def tune_chunks(self, chunks: List[Tuple[TensorType, float]]) -> List[Tuple[TensorType, float]]:
        if len(chunks) <= 1:
            return chunks
        chunk_tensor, chunk_sizes = [list(x) for x in zip(*chunks)]

        for i in range(1, len(chunk_sizes)):
            prev_size, curr_size = chunk_sizes[i - 1], chunk_sizes[i]
            if prev_size < curr_size and chunk_tensor[i].shape[0] != 0:
                chunk_tensor[i - 1] = torch.concat((chunk_tensor[i - 1], chunk_tensor[i][:1]), dim=0)
                chunk_tensor[i] = chunk_tensor[i][1:]
            elif prev_size > curr_size and chunk_tensor[i - 1].shape[0] != 0:
                chunk_tensor[i] = torch.concat((chunk_tensor[i - 1][-1:], chunk_tensor[i]))
                chunk_tensor[i - 1] = chunk_tensor[i - 1][:-1]
        return [(c, -1) for c in chunk_tensor]
    
    @typechecked
    def iterate(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            prompt: Union[str, ChatMessages],
            generate: bool = True,
            tune_chunks: bool = False,
            yield_trailing_chunk: bool = False,
            **kwargs
        ) -> Iterable:
        if isinstance(prompt, list):
            prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        if generate:
            tokens, logits = self.generate_infer(model, inputs, **kwargs)
        else:
            tokens, logits = self.parallel_infer(model, inputs, **kwargs)
        tokens, logits = tokens.cpu(), logits.cpu()

        assert tokens.shape[0] == 1, f"expected single batch dimension tokens, received dimension {tokens.shape[0]}"
        assert logits.shape[0] == 1, f"expected single batch dimension logits, received dimension {logits.shape[0]}"

        # don't support batch processing since the pad token handling gets overly complex
        tokens, logits = tokens[0], logits[0]

        chunks = None
        if tune_chunks:
            chunks = list(self.stream_chunks(tokens, logits, yield_trailing_chunk=yield_trailing_chunk))
            chunks = self.tune_chunks(chunks)
        else:
            chunks = self.stream_chunks(tokens, logits, yield_trailing_chunk=yield_trailing_chunk)

        for c, _ in chunks:
            yield c

