#!/usr/bin/env python3
"""
Script to extract token statistics from LMCache server logs.
Extracts lines containing "Total tokens X, LMCache hit tokens: Y, need to load: Z"
and saves them to a CSV file.
"""

import re
import csv
import argparse
from pathlib import Path


def extract_token_stats(log_file_path, output_csv_path):
    """Extract token statistics from log file and save to CSV."""
    
    # Regular expression to match the token statistics pattern
    pattern = r'Total tokens (\d+), LMCache hit tokens: (\d+), need to load: (-?\d+)'
    
    extracted_data = []
    
    print(f"Reading log file: {log_file_path}")
    
    with open(log_file_path, 'r', encoding='utf-8') as log_file:
        for line_num, line in enumerate(log_file, 1):
            match = re.search(pattern, line)
            if match:
                total_tokens = int(match.group(1))
                hit_tokens = int(match.group(2))
                need_to_load = int(match.group(3))
                
                extracted_data.append({
                    'index': len(extracted_data) + 1,
                    'total_tokens': total_tokens,
                    'lmcache_hit_tokens': hit_tokens,
                    'need_to_load': need_to_load
                })
    
    print(f"Found {len(extracted_data)} token statistics entries")
    
    # Write to CSV
    if extracted_data:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['index', 'total_tokens', 'lmcache_hit_tokens', 'need_to_load']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write data
            writer.writerows(extracted_data)
        
        print(f"Data saved to: {output_csv_path}")
        print(f"Total entries extracted: {len(extracted_data)}")
        
        # Calculate and print statistics
        total_tokens_sum = sum(entry['total_tokens'] for entry in extracted_data)
        hit_tokens_sum = sum(entry['lmcache_hit_tokens'] for entry in extracted_data)
        need_to_load_sum = sum(entry['need_to_load'] for entry in extracted_data)
        need_to_load_positive_sum = sum(entry['need_to_load'] for entry in extracted_data if entry['need_to_load'] > 0)
        
        print(f"\nStatistics:")
        print(f"Sum of total_tokens: {total_tokens_sum}")
        print(f"Sum of lmcache_hit_tokens: {hit_tokens_sum}")
        print(f"Sum of need_to_load: {need_to_load_sum}")
        print(f"Sum of need_to_load (positive only): {need_to_load_positive_sum}")
        
        if total_tokens_sum > 0:
            lmcache_hit_rate = need_to_load_positive_sum / total_tokens_sum
            vllm_hit_rate = (hit_tokens_sum - need_to_load_sum) / total_tokens_sum
            print(f"LMCache hit rate: {lmcache_hit_rate:.4f}")
            print(f"VLLM hit rate: {vllm_hit_rate:.4f}")
    else:
        print("No token statistics found in the log file.")


def main():
    parser = argparse.ArgumentParser(description='Extract token statistics from LMCache server logs')
    parser.add_argument('--log-file', '-l', required=True, 
                       help='Path to the log file')
    parser.add_argument('--output', '-o', default='token_stats.csv',
                       help='Output CSV file path (default: token_stats.csv)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file '{args.log_file}' does not exist.")
        return 1
    
    try:
        extract_token_stats(args.log_file, args.output)
        return 0
    except Exception as e:
        print(f"Error processing log file: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
