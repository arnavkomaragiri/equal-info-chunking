import torch
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from equal_info_chunking.entropy import EntropyChunkingStrategy
from equal_info_chunking.gzip import GZIPChunkingStrategy

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prompt', type=str, required=True)
    parser.add_argument('-g', '--generate', action='store_true')
    parser.add_argument('-m', '--model', type=str, default='gpt2')
    parser.add_argument('-s', '--chunk-size', type=float, default=10)
    parser.add_argument('-q', '--quantize', action='store_true')
    parser.add_argument('-n', '--max-new-tokens', type=int, default=100)
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

    entropy_strategy = EntropyChunkingStrategy(args.chunk_size)
    gzip_strategy = GZIPChunkingStrategy(tokenizer, args.chunk_size)

    prompt = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": args.prompt}
    ] 

    print('Entropy Chunking: ' + ('=' * 50))
    iterator = entropy_strategy.iterate(
        model, tokenizer, prompt, generate=args.generate, yield_trailing_chunk=True, max_new_tokens=args.max_new_tokens
    )
    for chunk in iterator:
        print('=' * 50)
        print(tokenizer.batch_decode([chunk])[0])
        print('=' * 50)

    print()
    print('GZIP Chunking: ' + ('=' * 50))
    iterator = gzip_strategy.iterate(
        model, tokenizer, prompt, generate=args.generate, yield_trailing_chunk=True, max_new_tokens=args.max_new_tokens
    )
    for chunk in iterator:
        print('=' * 50)
        print(tokenizer.batch_decode([chunk])[0])
        print('=' * 50)
