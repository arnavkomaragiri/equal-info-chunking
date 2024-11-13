import re
from typing import List, Optional

def compute_strategy_agreement(cand_chunks: List[str], test_chunks: List[str], verbose: bool = False, input_str: Optional[str] = None) -> float:
    test_chunk_idxs = []
    cand_chunk_idxs = []

    cand_chunks = [re.sub("\\s+", "", c) for c in cand_chunks if len(c.strip()) != 0]
    test_chunks = [re.sub("\\s+", "", c) for c in test_chunks if len(c.strip()) != 0]

    if input_str is not None:
        input_str = re.sub("\\s+", "", input_str)

    base = 0
    for chunk in test_chunks:
        test_chunk_idxs += [(base, base + len(chunk) - 1)]
        if input_str is not None:
            assert chunk == input_str[base:base+len(chunk)], f"mismatched chunk:\n{chunk}\n{'-' * 50}\n{input_str[base:base+len(chunk)]}"
        base += len(chunk)

    base = 0
    for chunk in cand_chunks:
        cand_chunk_idxs += [(base, base + len(chunk) - 1)]
        if input_str is not None:
            assert chunk == input_str[base:base+len(chunk)], f"mismatched chunk:\n{chunk}\n{'-' * 50}\n{input_str[base:base+len(chunk)]}"
        base += len(chunk)

    contains = lambda a, b: (a[0] <= b[0]) and (a[1] >= b[1])

    test_visited = [0 for _ in test_chunk_idxs] 
    cand_visited = [0 for _ in cand_chunk_idxs]
    for i, cand_chunk in enumerate(cand_chunk_idxs):
        for j, test_chunk in enumerate(test_chunk_idxs):
            if contains(cand_chunk, test_chunk) or contains(test_chunk, cand_chunk):
                if verbose:
                    print('=' * 50)
                    print(test_chunk)
                    print('-' * 50)
                    print(cand_chunk)
                    print('=' * 50)
                cand_visited[i] = 1
                test_visited[j] = 1
        
    agreement = (sum(test_visited) + sum(cand_visited)) / (len(test_visited) + len(cand_visited))
    return agreement

if __name__ == "__main__":
    input_str = "This is a test of the chunking agreement system. There are two sentences in this string."

    cand_chunks = input_str.split('.')
    test_chunks = cand_chunks[0].split(' ') + [cand_chunks[1]]

    agreement = compute_strategy_agreement(cand_chunks, test_chunks, verbose=True, input_str=input_str)
    assert agreement == 1.0, f"agreement function error detected, perfect chunking has non-perfect agreement score"
