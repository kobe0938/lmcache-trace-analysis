#!/usr/bin/env python3
"""
Script to extract average prompt throughput from LMCache server logs.
Extracts lines containing "Avg prompt throughput: X tokens/s"
and calculates the simple average.
"""

import re
import csv
import argparse
from pathlib import Path


def extract_prompt_throughput(log_file_path, output_csv_path):
    """Extract prompt throughput statistics from log file and save to CSV."""
    
    # Regular expression to match the prompt throughput pattern
    pattern = r'Avg prompt throughput: ([\d.]+) tokens/s'
    
    extracted_data = []
    
    print(f"Reading log file: {log_file_path}")
    
    with open(log_file_path, 'r', encoding='utf-8') as log_file:
        for line_num, line in enumerate(log_file, 1):
            match = re.search(pattern, line)
            if match:
                throughput = float(match.group(1))
                
                extracted_data.append({
                    'index': len(extracted_data) + 1,
                    'prompt_throughput': throughput
                })
    
    print(f"Found {len(extracted_data)} prompt throughput entries")
    
    # Write to CSV
    if extracted_data:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['index', 'prompt_throughput']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write data
            writer.writerows(extracted_data)
        
        print(f"Data saved to: {output_csv_path}")
        print(f"Total entries extracted: {len(extracted_data)}")
        
        # Calculate average prompt throughput
        total_throughput = sum(entry['prompt_throughput'] for entry in extracted_data)
        avg_prompt_throughput = total_throughput / len(extracted_data)
        
        print(f"\nPrompt Throughput Statistics:")
        print(f"Average prompt throughput: {avg_prompt_throughput:.1f} tokens/s")
        print(f"Total sum: {total_throughput:.1f} tokens/s")
        print(f"Number of entries: {len(extracted_data)}")
        
    else:
        print("No prompt throughput statistics found in the log file.")


def main():
    parser = argparse.ArgumentParser(description='Extract prompt throughput statistics from LMCache server logs')
    parser.add_argument('--log-file', '-l', required=True, 
                       help='Path to the log file')
    parser.add_argument('--output', '-o', default='prompt_throughput_stats.csv',
                       help='Output CSV file path (default: prompt_throughput_stats.csv)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file '{args.log_file}' does not exist.")
        return 1
    
    try:
        extract_prompt_throughput(args.log_file, args.output)
        return 0
    except Exception as e:
        print(f"Error processing log file: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

# python3 lmcache-trace-analysis/extract_vllm_prompt_throughput.py --log-file lmcache_server.log --output vllm_prompt_throughput_stats.csv
