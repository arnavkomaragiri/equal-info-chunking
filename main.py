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
    
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)

    entropy_strategy = EntropyChunkingStrategy(args.chunk_size)
    gzip_strategy = GZIPChunkingStrategy(args.chunk_size)

    print('Entropy Chunking: ' + ('=' * 50))
    for chunk in entropy_strategy.iterate(model, tokenizer, args.prompt, generate=args.generate, yield_trailing_chunk=True):
        print('=' * 50)
        print(tokenizer.batch_decode([chunk])[0])
        print('=' * 50)

    print()
    print('GZIP Chunking: ' + ('=' * 50))
    for chunk in gzip_strategy.iterate(model, tokenizer, args.prompt, generate=args.generate, yield_trailing_chunk=True):
        print('=' * 50)
        print(tokenizer.batch_decode([chunk])[0])
        print('=' * 50)
