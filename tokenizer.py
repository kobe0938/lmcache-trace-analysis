#!/usr/bin/env python3
import csv
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Sao10K/L3-8B-Lunaris-v1")
    
    # Read CSV with error handling
    df = pd.read_csv("/home/ubuntu/yuhan/conversation_id_10000_from_241227_qps_3_model_filtered.csv", 
                     on_bad_lines='skip', 
                     engine='python')
    
    # Process each row
    results = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Extract prompt from JSON body
            body = json.loads(row['body'])
            prompt = body.get('prompt', '')
            
            # Tokenize
            tokens = tokenizer.encode(prompt)
            
            # Store result
            results.append({
                'row_index': index,
                'tokens': tokens
            })
        except:
            results.append({
                'row_index': index,
                'tokens': []
            })
    
    # Save results
    pd.DataFrame(results).to_csv("/home/ubuntu/yuhan/conversation_id_10000_from_241227_qps_3_model_filtered_original_tokenized.csv", index=False)
    print("Done!")

if __name__ == "__main__":
    main()
