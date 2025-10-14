#!/usr/bin/env python3
"""
Visualize token lengths across requests.
X-axis: request number
Y-axis: token length
"""

import csv
import ast
import matplotlib.pyplot as plt


def parse_tokens(token_str: str) -> list:
    try:
        return list(ast.literal_eval(token_str))
    except Exception:
        return []


def main():
    csv_path = "/home/ubuntu/yuhan/conversation_qps_6_tokenized.csv"
    
    request_numbers = []
    token_lengths = []
    
    print("Reading CSV and calculating token lengths...")
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            tokens = parse_tokens(row["tokens"]) if "tokens" in row else []
            request_numbers.append(idx)
            token_lengths.append(len(tokens))
            
            if idx % 1000 == 0:
                print(f"Processed {idx} requests...")
    
    print(f"Total requests: {len(request_numbers)}")
    print(f"Creating plot...")
    
    # Calculate moving average for smoother trend
    window_size = 100
    moving_avg = []
    for i in range(len(token_lengths)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(token_lengths), i + window_size // 2)
        moving_avg.append(sum(token_lengths[start_idx:end_idx]) / (end_idx - start_idx))
    
    # Create the plot with both raw data and trend
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot: Raw data with transparency
    ax1.scatter(request_numbers, token_lengths, s=1, alpha=0.3, color='lightblue')
    ax1.plot(request_numbers, moving_avg, color='red', linewidth=2, label=f'{window_size}-request moving average')
    ax1.set_xlabel("Request Number")
    ax1.set_ylabel("Token Length")
    ax1.set_title("Token Length Distribution Across Requests (Raw + Moving Average)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Just the smooth trend
    ax2.plot(request_numbers, moving_avg, color='blue', linewidth=2)
    ax2.fill_between(request_numbers, moving_avg, alpha=0.3)
    ax2.set_xlabel("Request Number")
    ax2.set_ylabel("Token Length (Moving Avg)")
    ax2.set_title(f"Token Length Trend ({window_size}-request Moving Average)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = "/home/ubuntu/yuhan/lmcache-trace-analysis/token_lengths_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"  Min token length: {min(token_lengths)}")
    print(f"  Max token length: {max(token_lengths)}")
    print(f"  Average token length: {sum(token_lengths) / len(token_lengths):.2f}")


if __name__ == "__main__":
    main()

