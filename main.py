import os
import json
import argparse
from datetime import datetime
from longbench_test_hf import run_all, TORCH_AVAILABLE
from longbench_eval import load_references, evaluate_folder_outputs

try:
    import torch
except ImportError:
    torch = None

def main():
    parser = argparse.ArgumentParser(description="Run LongBench evaluation with HuggingFace or GGUF models")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate (default: 512)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for generation (default: 1.0)")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Model name/path: HuggingFace model name or path to GGUF file")
    parser.add_argument("--model_type", type=str, choices=["hf", "gguf"], default=None,
                        help="Model type: 'hf' for HuggingFace, 'gguf' for GGUF (auto-detected if not specified)")
    parser.add_argument("--prompt_dir", type=str, default="./prompt_files",
                        help="Directory containing prompt files (default: ./prompt_files)")
    parser.add_argument("--output_dir", type=str, default="./qmsum_outputs",
                        help="Directory to save output files (default: ./qmsum_outputs)")
    parser.add_argument("--disable_cache", action="store_true",
                        help="Disable KV cache (HuggingFace only, slower but fixes DynamicCache issues)")
    parser.add_argument("--n_gpu_layers", type=int, default=-1,
                        help="For GGUF: number of layers to offload to GPU (-1 for all, 0 for CPU only, default: -1)")
    parser.add_argument("--n_ctx", type=int, default=32768,
                        help="For GGUF: context window size (default: 32768)")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation after generation")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip generation and only run evaluation on existing outputs")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Output JSON file path for evaluation results (default: results_<timestamp>.json)")
    
    args = parser.parse_args()
    
    local_prompt_dir = args.prompt_dir
    output_dir = args.output_dir
    model_name = args.model_name
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    
    if args.model_type is None:
        if model_name.endswith('.gguf'):
            model_type = "gguf"
        else:
            model_type = "hf"
    else:
        model_type = args.model_type
    
    if model_type == "gguf":
        if not os.path.isabs(model_name) and not os.path.dirname(model_name):
            model_name = os.path.join("./gguf", model_name)
        elif not os.path.exists(model_name) and os.path.exists(os.path.join("./gguf", os.path.basename(model_name))):
            model_name = os.path.join("./gguf", os.path.basename(model_name))
    
    if model_type == "hf":
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for HuggingFace models. Install with: pip install torch")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Detected GPU: {gpu_name} ({gpu_memory:.2f} GB)")
    else:
        device = "cpu"
    if device == "cpu" and model_type == "hf":
        print("Warning: CUDA not available, using CPU (will be slower)")
    
    use_cache = not args.disable_cache
    
    print(f"Model type: {model_type.upper()}")
    if model_type == "gguf":
        print(f"Model path: {model_name}")
        print(f"GPU layers: {args.n_gpu_layers}")
        print(f"Context size: {args.n_ctx}")
    else:
        print(f"Using device: {device}")
        print(f"Model: {model_name}")
        print(f"Use cache: {use_cache}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    
    if not args.eval_only:
        latencies, total_time = run_all(
            local_prompt_dir=local_prompt_dir,
            output_dir=output_dir,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            use_cache=use_cache,
            model_type=model_type,
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.n_ctx
        )
        
        print("\n=== Benchmark Summary ===")
        for fname, lat in latencies:
            if lat >= 0:
                print(f"{fname}: {lat:.3f} s")
            else:
                print(f"{fname}: FAILED")
        print(f"Total time for {len(latencies)} samples: {total_time:.3f} s")
        successful = [lat for _, lat in latencies if lat >= 0]
        if successful:
            avg = sum(successful) / len(successful)
            print(f"Average latency: {avg:.3f} s ({len(successful)} successful)")
    else:
        print("\nSkipping generation (--eval_only mode)")
        if not os.path.exists(output_dir):
            print(f"Error: Output directory {output_dir} does not exist. Cannot evaluate.")
            return
    
    if not args.skip_eval or args.eval_only:
        print("\n" + "="*50)
        print("Starting evaluation...")
        print("="*50)
        
        try:
            print("Loading references...")
            ref_map = load_references()
            print(f"Loaded {len(ref_map)} references.")

            print(f"Evaluating predictions in {output_dir}")
            per_sample_scores, aggregated = evaluate_folder_outputs(output_dir, ref_map)

            print("\n=== ROUGE-L per sample ===")
            for idx, rl in per_sample_scores:
                print(f"Sample {idx}: ROUGE-L = {rl:.4f}")

            print("\n=== Aggregated ROUGE ===")
            print(f"ROUGE-L (F1): {aggregated['rougeL']:.4f}")
            print(f"ROUGE-1: {aggregated['rouge1']:.4f}, ROUGE-2: {aggregated['rouge2']:.4f}, ROUGE-Lsum: {aggregated['rougeLsum']:.4f}")
            
            timestamp = datetime.now().isoformat()
            results = {
                "timestamp": timestamp,
                "model_name": model_name if not args.eval_only else "N/A (eval_only)",
                "max_new_tokens": max_new_tokens if not args.eval_only else "N/A",
                "temperature": temperature if not args.eval_only else "N/A",
                "rougeL": aggregated['rougeL'],
                "rouge1": aggregated['rouge1'],
                "rouge2": aggregated['rouge2'],
                "rougeLsum": aggregated['rougeLsum'],
                "per_sample_scores": [{"sample_id": idx, "rougeL": float(rl)} for idx, rl in per_sample_scores]
            }
            
            if args.results_file:
                results_file = args.results_file
            else:
                timestamp_safe = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = f"results_{timestamp_safe}.json"
            
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\n=== Results saved to {results_file} ===")
        except ImportError as e:
            print(f"\n[WARNING] Evaluation skipped: {e}")
            print("Install required packages: pip install datasets evaluate")
        except Exception as e:
            print(f"\n[ERROR] Evaluation failed: {e}")

if __name__ == "__main__":
    main()

