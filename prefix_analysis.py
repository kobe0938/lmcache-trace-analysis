#!/usr/bin/env python3
"""
Minimal analysis of non-prefix token counts per request.

For request i (1-indexed):
  y[i] = y[i-1] + (len(tokens[i]) - max_shared_prefix(tokens[i], any previous))

We compute max_shared_prefix using a trie of all previous requests for O(L) time per request.
Outputs CSV with columns: request_index, cumulative_non_prefix_count
"""

import csv
import ast
import sys
from typing import Dict, Any, List


def parse_tokens(token_str: str) -> List[int]:
    try:
        return list(ast.literal_eval(token_str))
    except Exception:
        return []


class Trie:
    def __init__(self) -> None:
        self.root: Dict[int, Any] = {}

    def longest_prefix_len(self, tokens: List[int]) -> int:
        node = self.root
        length = 0
        for t in tokens:
            nxt = node.get(t)
            if nxt is None:
                break
            length += 1
            node = nxt
        return length

    def insert(self, tokens: List[int]) -> None:
        node = self.root
        for t in tokens:
            if t not in node:
                node[t] = {}
            node = node[t]


def analyze_prefix_sharing(csv_path: str) -> List[int]:
    trie = Trie()
    cumulative_counts: List[int] = []
    cumulative = 0

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            tokens = parse_tokens(row["tokens"]) if "tokens" in row else []

            if idx == 1:
                non_prefix = len(tokens)
            else:
                common = trie.longest_prefix_len(tokens)
                non_prefix = len(tokens) - common

            cumulative += non_prefix
            cumulative_counts.append(cumulative)

            trie.insert(tokens)

            if idx % 1000 == 0:
                print(f"Processed {idx} requests; cumulative={cumulative}")

    return cumulative_counts


def calculate_hit_rate(csv_path: str) -> float:
    """Calculate hit rate: hit tokens / all tokens"""
    trie = Trie()
    total_tokens = 0
    hit_tokens = 0

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            tokens = parse_tokens(row["tokens"]) if "tokens" in row else []
            total_tokens += len(tokens)

            if idx == 1:
                # First request has no prefix hits
                pass
            else:
                common = trie.longest_prefix_len(tokens)
                hit_tokens += common

            trie.insert(tokens)

    return hit_tokens / total_tokens if total_tokens > 0 else 0.0


def main() -> None:
    csv_path = "/home/ubuntu/yuhan/conversation_qps_6_tokenized.csv"
    out_path = "/home/ubuntu/yuhan/lmcache-trace-analysis/prefix_analysis_results.csv"

    print("Starting prefix sharing analysis (stdlib-only)...")
    counts = analyze_prefix_sharing(csv_path)

    print("Calculating hit rate...")
    hit_rate = calculate_hit_rate(csv_path)

    print("Writing results...")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["request_index", "cumulative_non_prefix_count"])
        for i, c in enumerate(counts, start=1):
            writer.writerow([i, c])

    print(f"Done. Total requests: {len(counts)}")
    print(f"Final cumulative count: {counts[-1] if counts else 0}")
    print(f"Hit rate: {hit_rate:.4f} ({hit_rate*100:.2f}%)")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
