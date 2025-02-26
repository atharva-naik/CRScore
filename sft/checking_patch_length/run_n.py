#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import torch
import pandas as pd
import numpy as np
import tqdm
import argparse
import sys


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Process code patches with context")
    parser.add_argument("--n", type=int, default=15, help="Number of context lines before and after patch")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to process")
    parser.add_argument("--input", type=str, default="/home/mkapadni/work/crscore_plus_plus/Comment_Generation/msg-valid.jsonl", 
                        help="Path to input JSONL file")
    parser.add_argument("--output", type=str, default=None, 
                        help="Path to output CSV file (default: n={n}_pred_validdf.csv)")
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-Coder-3B-Instruct", 
                        help="Model to use for inference")
    return parser.parse_args()


def get_patch_context(oldf, patch, n):
    """
    Display code patch with n lines of context before and after.
    
    Args:
        oldf (str): Original file content
        patch (str): Patch content in unified diff format
        n (int): Number of context lines before and after
        
    Returns:
        str: Integrated code with context
    """
    import re
    
    # Split file into lines
    oldf_lines = oldf.split('\n')
    
    # Extract line number from patch header
    match = re.search(r'@@ -(\d+)', patch)
    if not match:
        return "Could not determine patch location"
    
    # Get starting line number (1-based) and convert to 0-based index
    start_line = int(match.group(1))
    idx = start_line - 1
    
    # Calculate context ranges
    before_start = max(0, idx - n)
    after_start = idx
    after_end = min(len(oldf_lines), after_start + n)
    
    # Extract added lines from patch (remove '+' prefix)
    added_lines = []
    found_header = False
    for line in patch.split('\n'):
        if line.startswith('@@'):
            found_header = True
            continue
        if found_header:
            if line.startswith('+'):
                added_lines.append(line[1:])
    
    # Build result
    result = []
    
    # Lines before patch
    for i in range(before_start, idx):
        result.append(oldf_lines[i])
    
    # Patch content (integrated)
    result.extend(added_lines)
    
    # Lines after patch
    for i in range(after_start, after_end):
        result.append(oldf_lines[i])
    
    return "\n".join(result)


def main():
    """Main function to run the script"""
    # Get command line arguments
    args = parse_args()
    
    # Set output file name if not provided
    if args.output is None:
        args.output = f"n={args.n}_pred_validdf.csv"
    
    # Load data
    print(f"Loading data from {args.input}")
    valid_df = pd.read_json(args.input, lines=True)
    
    # Load model with Unsloth optimizations
    print(f"Loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = 3000,  # Max generation length
        dtype = None,
        load_in_4bit = False,     # 4-bit quantization for memory efficiency
        device_map = "auto",     # Automatic GPU allocation
    )
    
    # Enable Unsloth's fast inference mode
    FastLanguageModel.for_inference(model)
    
    # Process the specified number of samples
    print(f"Processing {args.samples} samples with {args.n} lines of context")
    results = []
    for index, row in tqdm.tqdm(valid_df.iloc[:args.samples].iterrows(), total=args.samples):
        try:
            # Get patch context with specified n
            patch_context = get_patch_context(row["oldf"], row["patch"], n=args.n)
            
            # Create chat template
            messages = [
                {"content": """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
                 Now look at the suggested code changes and provide a concise, and compact review of the suggested code change.""", "role": "system"},
                {"content": patch_context, "role": "user"}
            ]
            
            # Apply chat template formatting
            text = tokenizer.apply_chat_template(
                messages,
                tokenize = False,
                add_generation_prompt = True
            )
            
            # Generate response
            inputs = tokenizer([text], return_tensors="pt").to("cuda")
            outputs = model.generate(
                **inputs,
                max_new_tokens = 512,
                temperature = 0.1,
                top_p = 0.1,
                repetition_penalty = 1.05
            )
            
            # Decode output
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            assistant_response = prediction.split("<|assistant|>")[-1].strip()
            
            #give me the text after "assistant"
            # Print response
            assistant_response = assistant_response.split("assistant")[-1].strip()
            # Add to results
            results.append({
                "index": index,
                "patch": row["patch"],
                "patch_context": patch_context,
                "original_msg": row["msg"],
                "predicted_msg": assistant_response
            })
            
            # Print progress
            if (index + 1) % 10 == 0:
                print(f"Processed {index + 1} samples")
                
        except Exception as e:
            print(f"Error processing sample {index}: {e}")
            # Continue with next sample
            continue
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    
    print(f"Completed processing {len(results)} samples and saved to {args.output}")


if __name__ == "__main__":
    main()