#!/usr/bin/env python3
"""
Script to extract throughput statistics from LMCache server logs.
Extracts lines containing "size: X gb, cost Y ms, throughput: Z GB/s"
and calculates weighted average throughput.
"""

import re
import csv
import argparse
from pathlib import Path


def extract_throughput_stats(log_file_path, output_csv_path):
    """Extract throughput statistics from log file and save to CSV."""
    
    # Regular expression to match the throughput pattern
    pattern = r'size: ([\d.]+) gb.*?throughput: ([\d.]+) GB/s'
    
    extracted_data = []
    
    print(f"Reading log file: {log_file_path}")
    
    with open(log_file_path, 'r', encoding='utf-8') as log_file:
        for line_num, line in enumerate(log_file, 1):
            match = re.search(pattern, line)
            if match:
                size_gb = float(match.group(1))
                throughput_gbps = float(match.group(2))
                
                extracted_data.append({
                    'index': len(extracted_data) + 1,
                    'size_gb': size_gb,
                    'throughput_gbps': throughput_gbps
                })
    
    print(f"Found {len(extracted_data)} throughput entries")
    
    # Write to CSV
    if extracted_data:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['index', 'size_gb', 'throughput_gbps']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write data
            writer.writerows(extracted_data)
        
        print(f"Data saved to: {output_csv_path}")
        print(f"Total entries extracted: {len(extracted_data)}")
        
        # Calculate weighted average throughput
        total_size = sum(entry['size_gb'] for entry in extracted_data)
        weighted_throughput_sum = sum(entry['size_gb'] * entry['throughput_gbps'] for entry in extracted_data)
        
        print(f"\nThroughput Statistics:")
        print(f"Total size: {total_size:.4f} GB")
        print(f"Number of entries: {len(extracted_data)}")
        
        if total_size > 0:
            weighted_avg_throughput = weighted_throughput_sum / total_size
            print(f"Weighted average throughput: {weighted_avg_throughput:.4f} GB/s")
        else:
            print("No valid size data found for throughput calculation")
            
        # Also calculate simple average for comparison
        avg_throughput = sum(entry['throughput_gbps'] for entry in extracted_data) / len(extracted_data)
        print(f"Simple average throughput: {avg_throughput:.4f} GB/s")
        
    else:
        print("No throughput statistics found in the log file.")


def main():
    parser = argparse.ArgumentParser(description='Extract throughput statistics from LMCache server logs')
    parser.add_argument('--log-file', '-l', required=True, 
                       help='Path to the log file')
    parser.add_argument('--output', '-o', default='throughput_stats.csv',
                       help='Output CSV file path (default: throughput_stats.csv)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file '{args.log_file}' does not exist.")
        return 1
    
    try:
        extract_throughput_stats(args.log_file, args.output)
        return 0
    except Exception as e:
        print(f"Error processing log file: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

# 3
# python /home/ubuntu/yuhan/lmcache-trace-analysis/extract_lmcache_throughput_stats.py --log-file lmcache_server_qps_5.log --output throughput_stats.csv