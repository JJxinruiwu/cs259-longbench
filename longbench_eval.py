import os
import re
from datasets import load_dataset
import evaluate

def load_references():
    ds = load_dataset("zai-org/LongBench", "qmsum", split="test", trust_remote_code=True)
    ref_map = {}
    for i, rec in enumerate(ds):
        ans = rec["answers"]
        ref = ans[0] if isinstance(ans, list) and len(ans) > 0 else ans
        ref_map[i] = ref.strip()
    return ref_map

def evaluate_folder_outputs(output_dir: str, ref_map: dict):
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

    result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    per_sample = rouge.compute(predictions=predictions, references=references, use_stemmer=True, use_aggregator=False)

    results = []
    for i, idx in enumerate(sample_ids):
        rl = per_sample["rougeL"][i]
        results.append((idx, rl))

    return results, result

