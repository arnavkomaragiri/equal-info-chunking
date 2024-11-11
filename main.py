import torch
import argparse

import matplotlib.pyplot as plt

from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer
from equal_info_chunking.base import BaseChunkingStrategy, ChatMessages
from equal_info_chunking.entropy import EntropyChunkingStrategy
from equal_info_chunking.gzip import GZIPChunkingStrategy
from heuristic.math_extract_steps import split_solution
from typing import List, Union

def compute_strategy_agreement(
        model: PreTrainedModel, tokenizer: PreTrainedTokenizer, strategy: BaseChunkingStrategy, prompt: Union[str, ChatMessages],
        generate: bool = False, max_new_tokens: int = 100, verbose: bool = False
    ) -> float:
    tokens, equal_inf_chunks = [], []
    iterator = strategy.iterate(
        model, tokenizer, prompt, generate=generate, yield_trailing_chunk=True, max_new_tokens=max_new_tokens
    )
    for chunk in iterator:
        tokens += [chunk]
        chunk_str = tokenizer.batch_decode([chunk])[0]
        equal_inf_chunks += [chunk_str]
        if verbose:
            print('=' * 50)
            print(chunk_str)
    if verbose:
        print('=' * 50)

    # print(len("".join(equal_inf_chunks)))

    collected_output = tokenizer.batch_decode([torch.concat(tokens, dim=0)])[0]
    # print(len(collected_output))
    heuristic_chunks: List[str] = split_solution(collected_output)

    heur_visited = [0 for _ in heuristic_chunks] 
    info_visited = [0 for _ in equal_inf_chunks]
    for i, info_chunk in enumerate(equal_inf_chunks):
        for j, heur_chunk in enumerate(heuristic_chunks):
            if info_chunk in heur_chunk or heur_chunk in info_chunk:
                if verbose:
                    print('=' * 50)
                    print(info_chunk)
                    print('-' * 50)
                    print(heur_chunk)
                    print('=' * 50)
                info_visited[i] = 1
                heur_visited[j] = 1
            # else:
            #     if verbose:
            #         print('=' * 50)
            #         print("FOUND CHUNK MISMATCH")
            #         print(info_chunk)
            #         print('-' * 50)
            #         print(heur_chunk)
            #         print('=' * 50)

    # heuristic_chunk_idxs = []
    # equal_inf_chunk_idxs = []

    # base = 0
    # for chunk in heuristic_chunks:
    #     heuristic_chunk_idxs += [(base, base + len(chunk))]
    #     assert chunk == collected_output[base:base+len(chunk)], f"mismatched chunk:\n{chunk}\n{'-' * 50}\n{collected_output[base:base+len(chunk)]}"
    #     base += len(chunk) + 1
    # print(heuristic_chunk_idxs)

    # base = 0
    # for chunk in chunks:
    #     equal_inf_chunk_idxs += [(base, base + len(chunk))]
    #     assert chunk == collected_output[base:base+len(chunk)], f"mismatched chunk:\n{chunk}\n{'-' * 50}\n{collected_output[base:base+len(chunk)]}"
    #     base += len(chunk) + 1
    # print(equal_inf_chunk_idxs)

    # contains = lambda a, b: (a[0] <= b[0]) and (a[1] >= b[1])

    # heur_visited = [0 for _ in heuristic_chunk_idxs] 
    # info_visited = [0 for _ in equal_inf_chunk_idxs]
    # for i, info_chunk in enumerate(equal_inf_chunk_idxs):
    #     for j, heur_chunk in enumerate(heuristic_chunk_idxs):
    #         if contains(info_chunk, heur_chunk) or contains(heur_chunk, info_chunk):
    #             if verbose:
    #                 print('=' * 50)
    #                 print(info_chunk)
    #                 print('-' * 50)
    #                 print(heur_chunk)
    #                 print('=' * 50)
    #             info_visited[i] = 1
    #             heur_visited[j] = 1
    #         else:
    #             if verbose:
    #                 print('=' * 50)
    #                 print("FOUND CHUNK MISMATCH")
    #                 print(info_chunk)
    #                 print(chunks[i])
    #                 print('-' * 50)
    #                 print(heur_chunk)
    #                 print(heuristic_chunks[j])
    #                 print('=' * 50)
        
    # print(heur_visited)
    # print(info_visited)
    agreement = (sum(heur_visited) + sum(info_visited)) / (len(heur_visited) + len(info_visited))
    return agreement

def plot_size_vs_agreement(
        model: PreTrainedModel, tokenizer: PreTrainedTokenizer, strategy: BaseChunkingStrategy, prompt: Union[str, ChatMessages],
        generate: bool = False, max_new_tokens: int = 100, step: float = 0.1, label: str = ""
    ):
    max_chunk_size = float(strategy.max_chunk_size)
    sizes, purities = [], []
    num_trials = int(1 // step)
    for n in trange(1, num_trials + 1):
        scale = n * step
        strategy.max_chunk_size = max_chunk_size * scale
        agreement = compute_strategy_agreement(model, tokenizer, strategy, prompt, generate=generate, max_new_tokens=max_new_tokens)
        sizes += [strategy.max_chunk_size]
        purities += [agreement]
    plt.plot(sizes, purities, label=label)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prompt', type=str, required=True)
    parser.add_argument('-g', '--generate', action='store_true')
    parser.add_argument('-m', '--model', type=str, default='gpt2')
    parser.add_argument('-s', '--chunk-size', type=float, default=10)
    parser.add_argument('-q', '--quantize', action='store_true')
    parser.add_argument('-n', '--max-new-tokens', type=int, default=100)
    parser.add_argument('-w', '--weight', type=float, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    kwargs = {}
    if args.quantize:
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        kwargs['quantization_config'] = config

    model = AutoModelForCausalLM.from_pretrained(args.model, low_cpu_mem_usage=True, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)

    prompt = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": args.prompt}
    ]

    entropy_strategy = EntropyChunkingStrategy(args.chunk_size).calibrate(model, tokenizer, [prompt], weight=args.weight)
    gzip_strategy = GZIPChunkingStrategy(tokenizer, args.chunk_size).calibrate(model, tokenizer, [prompt], weight=args.weight)

    print('Entropy Chunking: ' + ('=' * 50))
    agreement = compute_strategy_agreement(
        model, tokenizer, entropy_strategy, prompt, 
        generate=args.generate, max_new_tokens=args.max_new_tokens, verbose=True
    )
    print(f"Entropy Chunking Agreement: {100 * agreement:.5f}%")

    print()
    print('GZIP Chunking: ' + ('=' * 50))
    agreement = compute_strategy_agreement(
        model, tokenizer, gzip_strategy, prompt, 
        generate=args.generate, max_new_tokens=args.max_new_tokens, verbose=True
    )
    print(f"GZIP Chunking Agreement: {100 * agreement:.5f}%")

    plot_size_vs_agreement(model, tokenizer, entropy_strategy, prompt, generate=args.generate, max_new_tokens=args.max_new_tokens, label="Entropy Chunking")
    plot_size_vs_agreement(model, tokenizer, gzip_strategy, prompt, generate=args.generate, max_new_tokens=args.max_new_tokens, label="GZIP Chunking")
    plt.xlabel("Max Chunk Size")
    plt.ylabel("Chunking Agreement")
    plt.legend()
    plt.show()
