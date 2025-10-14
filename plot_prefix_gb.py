#!/usr/bin/env python3
"""
Visualize cumulative non-prefix token counts as GB over requests.

Reads: prefix_analysis_results.csv with columns: request_index,cumulative_non_prefix_count
Converts tokens to GB using: 0.1221 GB per 1000 tokens.
Outputs: prefix_analysis_plot_gb.png
"""

import csv
import os
import matplotlib.pyplot as plt

TOKENS_TO_GB_PER_1K = 0.1221  # GB per 1000 tokens
CSV_PATH = "/home/ubuntu/yuhan/lmcache-trace-analysis/prefix_analysis_results_original.csv"
PNG_PATH = "/home/ubuntu/yuhan/lmcache-trace-analysis/prefix_analysis_plot_gb_original.png"


def load_counts(path):
    xs, ys_tokens = [], []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(int(row["request_index"]))
            ys_tokens.append(int(row["cumulative_non_prefix_count"]))
    return xs, ys_tokens


def tokens_to_gb(tokens):
    return tokens * (TOKENS_TO_GB_PER_1K / 1000.0)


def main():
    if not os.path.exists(CSV_PATH):
        print(f"Input not found: {CSV_PATH}")
        return

    xs, ys_tokens = load_counts(CSV_PATH)
    ys_gb = [tokens_to_gb(t) for t in ys_tokens]

    plt.figure(figsize=(12, 6))
    plt.plot(xs, ys_gb, "b-", linewidth=1)
    plt.xlabel("Request Number")
    plt.ylabel("Cumulative Non-Prefix Data (GB)")
    plt.title("Cumulative Non-Prefix Tokens (GB) vs Request Number")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {PNG_PATH}")


if __name__ == "__main__":
    main()


