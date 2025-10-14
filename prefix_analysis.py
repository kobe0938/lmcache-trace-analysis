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
from typing import Dict, Any, List, Tuple
from collections import OrderedDict


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

    def remove(self, tokens: List[int]) -> None:
        """Remove a token sequence from the trie"""
        if not tokens:
            return
        
        # Find path and collect nodes
        path = []
        node = self.root
        for t in tokens:
            if t not in node:
                return  # Path doesn't exist
            path.append((node, t))
            node = node[t]
        
        # Remove from leaf up, stopping if node has other children
        for i in range(len(path) - 1, -1, -1):
            parent, token = path[i]
            if i == len(path) - 1:
                # Remove leaf if it has no children
                if not parent[token]:
                    del parent[token]
                else:
                    break
            else:
                # Remove intermediate node if it has no children
                if not parent[token]:
                    del parent[token]
                else:
                    break


class LRUTokenPool:
    """Token pool with LRU eviction based on token count limit"""
    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self.current_tokens = 0
        # OrderedDict to track access order: request_id -> tokens
        # Most recently used at the end, least recently used at the front
        self.requests: OrderedDict[int, List[int]] = OrderedDict()
    
    def longest_prefix_len(self, tokens: List[int]) -> Tuple[int, int]:
        """
        Find longest prefix match and return (length, matching_request_id).
        Also marks the matched request as recently used (LRU).
        """
        best_len = 0
        best_id = -1
        
        # Check all requests to find best prefix match
        for req_id, req_tokens in self.requests.items():
            # Calculate common prefix length
            common_len = 0
            for i in range(min(len(tokens), len(req_tokens))):
                if tokens[i] == req_tokens[i]:
                    common_len += 1
                else:
                    break
            
            if common_len > best_len:
                best_len = common_len
                best_id = req_id
        
        # Mark the matched request as recently used
        if best_id != -1:
            self.requests.move_to_end(best_id)
        
        return best_len, best_id
    
    def add_request(self, request_id: int, tokens: List[int]) -> None:
        """Add a request, evicting old ones if necessary"""
        # Evict until we have space
        while self.current_tokens + len(tokens) > self.max_tokens and self.requests:
            # Evict least recently used (first item in OrderedDict)
            old_id, old_tokens = self.requests.popitem(last=False)
            self.current_tokens -= len(old_tokens)
        
        # Add new request (most recent, so at the end)
        self.requests[request_id] = tokens
        self.current_tokens += len(tokens)


def analyze_prefix_sharing(csv_path: str, pool_size: int = None) -> List[int]:
    """
    Analyze prefix sharing with optional LRU pool size limit.
    
    Args:
        csv_path: Path to CSV file with tokenized requests
        pool_size: Maximum tokens in pool (None = unlimited)
    """
    if pool_size is None:
        # Original unlimited behavior
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
    else:
        # LRU pool behavior
        pool = LRUTokenPool(pool_size)
        cumulative_counts: List[int] = []
        cumulative = 0

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                tokens = parse_tokens(row["tokens"]) if "tokens" in row else []

                if idx == 1:
                    non_prefix = len(tokens)
                else:
                    common, _ = pool.longest_prefix_len(tokens)
                    non_prefix = len(tokens) - common

                cumulative += non_prefix
                cumulative_counts.append(cumulative)

                pool.add_request(idx, tokens)

                if idx % 1000 == 0:
                    print(f"Processed {idx} requests; cumulative={cumulative}, pool_tokens={pool.current_tokens}")

        return cumulative_counts


def calculate_hit_rate(csv_path: str, pool_size: int = None) -> float:
    """
    Calculate hit rate: hit tokens / all tokens
    
    Args:
        csv_path: Path to CSV file with tokenized requests
        pool_size: Maximum tokens in pool (None = unlimited)
    """
    if pool_size is None:
        # Original unlimited behavior
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
    else:
        # LRU pool behavior
        pool = LRUTokenPool(pool_size)
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
                    common, _ = pool.longest_prefix_len(tokens)
                    hit_tokens += common

                pool.add_request(idx, tokens)

        return hit_tokens / total_tokens if total_tokens > 0 else 0.0


def main() -> None:
    csv_path = "/home/ubuntu/yuhan/conversation_qps_6_tokenized.csv"
    out_path = "/home/ubuntu/yuhan/lmcache-trace-analysis/temp.csv"
    
    # Pool size limit in tokens (None = unlimited, or set to a number like 1000000)
    # 4100000 -> 500GB; 410000 -> 50GB
    pool_size = 410000
    
    pool_desc = f"pool_size={pool_size}" if pool_size else "unlimited"
    print(f"Starting prefix sharing analysis ({pool_desc})...")
    counts = analyze_prefix_sharing(csv_path, pool_size)

    print("Calculating hit rate...")
    hit_rate = calculate_hit_rate(csv_path, pool_size)

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
