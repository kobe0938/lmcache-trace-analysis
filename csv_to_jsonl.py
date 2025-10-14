#!/usr/bin/env python3
"""
Convert /home/ubuntu/yuhan/conversation_qps_6_tokenized.csv to JSONL format:

Each JSON line has fields:
  - timestamp (ordered, synthetic)
  - input_length (len of tokens)
  - output_length (always 1)
  - hash_ids (exact tokens as provided)

The original CSV is left untouched. Output path:
  /home/ubuntu/yuhan/conversation_qps_6_tokenized_converted.jsonl
"""

import ast
import csv
import json
from datetime import datetime, timedelta


INPUT_CSV = "/home/ubuntu/yuhan/conversation_qps_6_tokenized.csv"
OUTPUT_JSONL = "/home/ubuntu/yuhan/conversation_qps_6_tokenized_converted.jsonl"


def parse_tokens(token_str):
    try:
        return list(ast.literal_eval(token_str))
    except Exception:
        return []


def main():
    # Base timestamp similar to the example's date; increment per row to keep order
    base_ts = datetime(2024, 12, 23, 5, 26, 23)
    delta = timedelta(microseconds=333333)  # ~0.333s per request

    with open(INPUT_CSV, "r", newline="") as fin, open(OUTPUT_JSONL, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        for idx, row in enumerate(reader):
            tokens = parse_tokens(row.get("tokens", "[]"))
            input_length = len(tokens)
            output_length = 1
            # Use exact tokens as hash_ids per request
            hash_ids = tokens

            # Format timestamp with 9 fractional digits to resemble example
            ts = base_ts + delta * idx
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S.%f") + "000"  # append 3 zeros for ns-style

            obj = {
                "timestamp": ts_str,
                "input_length": input_length,
                "output_length": output_length,
                "hash_ids": hash_ids,
            }
            fout.write(json.dumps(obj) + "\n")

    print(f"Wrote: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()


