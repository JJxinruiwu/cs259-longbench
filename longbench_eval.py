#!/usr/bin/env python3

import os
import re
import json
import argparse
from datetime import datetime
from datasets import load_dataset
import evaluate

def load_references():
    """
    Load the qmsum test split and return a dict mapping sample index -> reference summary.
    You must adapt this to the correct field name in the dataset.
    """
    ds = load_dataset("zai-org/LongBench", "qmsum", split="test")
    ref_map = {}
    for i, rec in enumerate(ds):
        ans = rec["answers"]
        if isinstance(ans, list) and len(ans) > 0:
            ref = ans[0]
        else:
            ref = ans
        ref_map[i] = ref.strip()
    return ref_map

def evaluate_folder_outputs(output_dir: str, ref_map: dict):
    """
    Traverse output files (qmsum_test_{id}.txt) in output_dir,
    collect predictions and references, compute rouge via evaluate.
    Return results list and overall metrics.
    """
    rouge = evaluate.load("rouge")
    pattern = re.compile(r"qmsum_test_(\d+)\.txt")
    predictions = []
    references = []
    sample_ids = []

    for fname in sorted(os.listdir(output_dir)):
        m = pattern.match(fname)
        if not m:
            print(f"Skipping unrecognized file name: {fname}")
            continue
        idx = int(m.group(1))
        outpath = os.path.join(output_dir, fname)
        with open(outpath, "r", encoding="utf-8") as f:
            pred = f.read().strip()

        if idx not in ref_map:
            print(f"Warning: no reference for sample {idx}, skipping")
            continue

        ref = ref_map[idx]
        sample_ids.append(idx)
        predictions.append(pred)
        references.append(ref)

    # Compute ROUGE (aggregated)
    result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    # result is a dict: e.g. { 'rouge1': ..., 'rouge2': ..., 'rougeL': ..., 'rougeLsum': ... } :contentReference[oaicite:0]{index=0}

    # Also compute per-sample if desired by setting `use_aggregator=False`
    per_sample = rouge.compute(predictions=predictions, references=references, use_stemmer=True, use_aggregator=False)

    # per_sample entries are lists of scores per sample e.g. per_sample['rougeL'] is list of f1 for each sample :contentReference[oaicite:1]{index=1}

    results = []
    for i, idx in enumerate(sample_ids):
        rl = per_sample["rougeL"][i]
        results.append((idx, rl))

    return results, result

def main():
    parser = argparse.ArgumentParser(description="Evaluate LongBench outputs and save results to JSON")
    parser.add_argument("--output_dir", type=str, default="qmsum_outputs",
                        help="Directory containing output files (default: qmsum_outputs)")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model name used for generation (default: meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max new tokens used for generation (default: 512)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature used for generation (default: 1.0)")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Output JSON file path (default: results_<timestamp>.json)")
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    model_name = args.model_name
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    
    print("Loading references …")
    ref_map = load_references()
    print(f"Loaded {len(ref_map)} references.")

    print("Evaluating predictions in", output_dir)
    per_sample_scores, aggregated = evaluate_folder_outputs(output_dir, ref_map)

    print("\n=== ROUGE-L per sample ===")
    for idx, rl in per_sample_scores:
        print(f"Sample {idx}: ROUGE-L = {rl:.4f}")

    print("\n=== Aggregated ROUGE ===")
    # Print only rougeL, or print all metrics
    print(f"ROUGE-L (F1): {aggregated['rougeL']:.4f}")
    print(f"ROUGE-1: {aggregated['rouge1']:.4f}, ROUGE-2: {aggregated['rouge2']:.4f}, ROUGE-Lsum: {aggregated['rougeLsum']:.4f}")
    
    # Create results dictionary with timestamp
    timestamp = datetime.now().isoformat()
    results = {
        "timestamp": timestamp,
        "model_name": model_name,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "rougeL": aggregated['rougeL'],
        "rouge1": aggregated['rouge1'],
        "rouge2": aggregated['rouge2'],
        "rougeLsum": aggregated['rougeLsum'],
        "per_sample_scores": [{"sample_id": idx, "rougeL": float(rl)} for idx, rl in per_sample_scores]
    }
    
    # Determine output file path
    if args.results_file:
        results_file = args.results_file
    else:
        # Create filename with timestamp
        timestamp_safe = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results_{timestamp_safe}.json"
    
    # Save to JSON file
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Results saved to {results_file} ===")

if __name__ == "__main__":
    main()
