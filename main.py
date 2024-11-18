import re
import torch
import argparse

import matplotlib.pyplot as plt

from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer
from equal_info_chunking import BaseChunkingStrategy, EntropyChunkingStrategy, NLLChunkingStrategy, GZIPChunkingStrategy, RandomChunkingStrategy, ChatMessages
from equal_info_chunking.metrics import compute_strategy_agreement
from equal_info_chunking.heuristic import split_solution
from typing import List, Union


def plot_size_vs_agreement(
        model: PreTrainedModel, tokenizer: PreTrainedTokenizer, strategy: BaseChunkingStrategy, prompt: Union[str, ChatMessages],
        generate: bool = False, max_new_tokens: int = 100, tune_chunks: bool = False, step: float = 0.1, label: str = ""
    ):
    max_chunk_size = float(strategy.max_chunk_size)
    sizes, agreements = [], []
    num_trials = int(1 // step)
    for n in trange(1, num_trials + 1):
        scale = n * step
        strategy.max_chunk_size = max_chunk_size * scale

        iterator = strategy.iterate(model, tokenizer, prompt, generate=generate, yield_trailing_chunk=True, max_new_tokens=max_new_tokens, tune_chunks=tune_chunks)
        cand_chunks = []
        for chunk in iterator:
            cand_chunks += tokenizer.batch_decode([chunk])

        cand_str = "".join(cand_chunks)
        heur_chunks = split_solution(cand_str)

        agreement = compute_strategy_agreement(cand_chunks, heur_chunks, input_str=cand_str)
        sizes += [strategy.max_chunk_size]
        agreements += [agreement]
    plt.plot(sizes, agreements, label=label)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prompt', type=str, required=True)
    parser.add_argument('-g', '--generate', action='store_true')
    parser.add_argument('-m', '--model', type=str, default='gpt2')
    parser.add_argument('-s', '--chunk-size', type=float, default=10)
    parser.add_argument('-q', '--quantize', action='store_true')
    parser.add_argument('-n', '--max-new-tokens', type=int, default=100)
    parser.add_argument('-w', '--weight', type=float, default=1)
    parser.add_argument('-t', '--tune-chunks', action='store_true')
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

    entr_strategy = EntropyChunkingStrategy(args.chunk_size).calibrate(model, tokenizer, [prompt], weight=args.weight)
    nlgl_strategy = NLLChunkingStrategy(args.chunk_size).calibrate(model, tokenizer, [prompt], weight=args.weight)
    gzip_strategy = GZIPChunkingStrategy(tokenizer, args.chunk_size).calibrate(model, tokenizer, [prompt], weight=args.weight)
    rand_strategy = RandomChunkingStrategy(entr_strategy.max_chunk_size, weight=0.1)

    iterator = entr_strategy.iterate(model, tokenizer, prompt, generate=args.generate, yield_trailing_chunk=True, max_new_tokens=args.max_new_tokens, tune_chunks=args.tune_chunks)
    entr_chunks = []
    for chunk in iterator:
        print('=' * 100)
        print(tokenizer.batch_decode([chunk])[0])
        entr_chunks += tokenizer.batch_decode([chunk])

    iterator = rand_strategy.iterate(model, tokenizer, prompt, generate=args.generate, yield_trailing_chunk=True, max_new_tokens=args.max_new_tokens, tune_chunks=args.tune_chunks)
    rand_chunks = []
    for chunk in iterator:
        print('=' * 100)
        print(tokenizer.batch_decode([chunk])[0])
        rand_chunks += tokenizer.batch_decode([chunk])

    # iterator = nlgl_strategy.iterate(model, tokenizer, prompt, generate=args.generate, yield_trailing_chunk=True, max_new_tokens=args.max_new_tokens, tune_chunks=args.tune_chunks)
    nlgl_chunks = []
    # for chunk in iterator:
    #     print('=' * 100)
    #     print(tokenizer.batch_decode([chunk])[0])
    #     nlgl_chunks += tokenizer.batch_decode([chunk])
    
    iterator = gzip_strategy.iterate(model, tokenizer, prompt, generate=args.generate, yield_trailing_chunk=True, max_new_tokens=args.max_new_tokens, tune_chunks=args.tune_chunks)
    gzip_chunks = []
    for chunk in iterator:
        print('=' * 100)
        print(tokenizer.batch_decode([chunk])[0])
        gzip_chunks += tokenizer.batch_decode([chunk])

    entr_str, nlgl_str, gzip_str = "".join(entr_chunks), "".join(nlgl_chunks), "".join(gzip_chunks)
    rand_str = "".join(rand_chunks)

    print('Entropy Chunking: ' + ('=' * 50))
    entr_heur_chunks = split_solution(entr_str)
    entr_agreement = compute_strategy_agreement(entr_chunks, entr_heur_chunks, input_str=entr_str)
    print(f"Entropy Chunking Agreement: {100 * entr_agreement:.5f}%")

    print('Random Chunking: ' + ('=' * 50))
    rand_heur_chunks = split_solution(rand_str)
    rand_agreement = compute_strategy_agreement(rand_chunks, rand_heur_chunks, input_str=rand_str)
    print(f"Random Chunking Agreement: {100 * rand_agreement:.5f}%")

    # print()
    # print('NLL Chunking: ' + ('=' * 50))
    # nlgl_heur_chunks = split_solution(nlgl_str)
    # nlgl_agreement = compute_strategy_agreement(nlgl_chunks, nlgl_heur_chunks, input_str=nlgl_str)
    # print(f"NLL Chunking Agreement: {100 * nlgl_agreement:.5f}%")

    print()
    print('GZIP Chunking: ' + ('=' * 50))
    gzip_heur_chunks = split_solution(gzip_str)
    gzip_agreement = compute_strategy_agreement(gzip_chunks, gzip_heur_chunks, input_str=gzip_str)
    print(f"GZIP Chunking Agreement: {100 * gzip_agreement:.5f}%")

    plot_size_vs_agreement(model, tokenizer, entr_strategy, prompt, generate=args.generate, max_new_tokens=args.max_new_tokens, tune_chunks=args.tune_chunks, label="Entropy Chunking")
    plot_size_vs_agreement(model, tokenizer, rand_strategy, prompt, generate=args.generate, max_new_tokens=args.max_new_tokens, tune_chunks=args.tune_chunks, label="Random Chunking")
    plot_size_vs_agreement(model, tokenizer, gzip_strategy, prompt, generate=args.generate, max_new_tokens=args.max_new_tokens, tune_chunks=args.tune_chunks, label="GZIP Chunking")
    plt.xlabel("Max Chunk Size")
    plt.ylabel("Chunking Agreement")
    plt.legend()
    plt.show()