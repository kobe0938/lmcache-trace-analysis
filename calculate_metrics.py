#!/usr/bin/env python3
"""
Script to calculate TTFT (Time to First Token) and ITL (Inter-Token Latency) 
from vLLM metrics log files.

Usage: python calculate_metrics.py <metrics_log_file>
"""

import sys
import re
import argparse


def parse_metrics_file(file_path):
    """
    Parse the metrics log file and extract relevant metrics.
    
    Args:
        file_path (str): Path to the metrics log file
        
    Returns:
        dict: Dictionary containing extracted metrics
    """
    metrics = {}
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Extract TTFT metrics
        ttft_sum_match = re.search(
            r'vllm:time_to_first_token_seconds_sum\{[^}]*\}\s+([\d.e+-]+)', 
            content
        )
        ttft_count_match = re.search(
            r'vllm:time_to_first_token_seconds_count\{[^}]*\}\s+([\d.e+-]+)', 
            content
        )
        
        # Extract ITL metrics
        itl_sum_match = re.search(
            r'vllm:time_per_output_token_seconds_sum\{[^}]*\}\s+([\d.e+-]+)', 
            content
        )
        itl_count_match = re.search(
            r'vllm:time_per_output_token_seconds_count\{[^}]*\}\s+([\d.e+-]+)', 
            content
        )
        
        if ttft_sum_match and ttft_count_match:
            metrics['ttft_sum'] = float(ttft_sum_match.group(1))
            metrics['ttft_count'] = float(ttft_count_match.group(1))
        else:
            print("Warning: Could not find TTFT metrics in the file")
            
        if itl_sum_match and itl_count_match:
            metrics['itl_sum'] = float(itl_sum_match.group(1))
            metrics['itl_count'] = float(itl_count_match.group(1))
        else:
            print("Warning: Could not find ITL metrics in the file")
            
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
        
    return metrics


def calculate_and_display_metrics(metrics):
    """
    Calculate TTFT and ITL from the extracted metrics and display results.
    
    Args:
        metrics (dict): Dictionary containing extracted metrics
    """
    print("=" * 60)
    print("VLLM METRICS CALCULATION")
    print("=" * 60)
    
    # Calculate and display TTFT
    if 'ttft_sum' in metrics and 'ttft_count' in metrics:
        ttft = metrics['ttft_sum'] / metrics['ttft_count']
        print(f"\nTTFT (Time to First Token):")
        print(f"  Sum:   {metrics['ttft_sum']:.6f} seconds")
        print(f"  Count: {metrics['ttft_count']:.0f}")
        print(f"  TTFT:  {ttft:.6f} seconds")
        print(f"  Formula: {metrics['ttft_sum']:.6f} / {metrics['ttft_count']:.0f} = {ttft:.6f}")
    else:
        print("\nTTFT: Could not calculate (missing metrics)")
    
    # Calculate and display ITL
    if 'itl_sum' in metrics and 'itl_count' in metrics:
        itl = metrics['itl_sum'] / metrics['itl_count']
        print(f"\nITL (Inter-Token Latency):")
        print(f"  Sum:   {metrics['itl_sum']:.6f} seconds")
        print(f"  Count: {metrics['itl_count']:.0f}")
        print(f"  ITL:   {itl:.6f} seconds")
        print(f"  Formula: {metrics['itl_sum']:.6f} / {metrics['itl_count']:.0f} = {itl:.6f}")
    else:
        print("\nITL: Could not calculate (missing metrics)")
    
    print("\n" + "=" * 60)


def main():
    """Main function to handle command line arguments and orchestrate the calculation."""
    parser = argparse.ArgumentParser(
        description="Calculate TTFT and ITL from vLLM metrics log files"
    )
    parser.add_argument(
        "metrics_file", 
        help="Path to the metrics log file"
    )
    
    args = parser.parse_args()
    
    # Parse the metrics file
    metrics = parse_metrics_file(args.metrics_file)
    
    # Calculate and display results
    calculate_and_display_metrics(metrics)


if __name__ == "__main__":
    main()
# 4
# python /home/ubuntu/yuhan/lmcache-trace-analysis/calculate_metrics.py vllm_metrics_output.log
# python /home/ubuntu/yuhan/lmcache-trace-analysis/calculate_metrics.py lmcache_metrics_output.log