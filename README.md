# LongBench Evaluation

Tool for running LongBench QMSum evaluations with HuggingFace or GGUF models, including automatic ROUGE score computation.

## Installation

```bash
pip install transformers torch datasets evaluate
```

For GGUF models:
```bash
pip install llama-cpp-python
# With CUDA: export CMAKE_ARGS="-DGGML_CUDA=on" && pip install llama-cpp-python
# With Metal: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
```

## Usage

### HuggingFace Models

```bash
python main.py --model_name "meta-llama/Llama-3.2-1B-Instruct"
python main.py --model_name "your-model" --max_new_tokens 256 --temperature 0.7
python main.py --model_name "phi-3.5-moe-instruct" --disable_cache  # Fix DynamicCache errors
```

### GGUF Models

Place GGUF files in `./gguf/` directory:

```bash
python main.py --model_name "Llama-3.2-1B-Instruct-Q4_0.gguf"
python main.py --model_name "model.gguf" --max_new_tokens 256 --temperature 0.7
python main.py --model_name "model.gguf" --n_gpu_layers -1 --n_ctx 8192
python main.py --model_name "model.gguf" --n_gpu_layers 0  # CPU only
```

### Options

```bash
python main.py --skip_eval  # Skip evaluation
python main.py --results_file "my_results.json"  # Custom results file
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `meta-llama/Llama-3.2-1B-Instruct` | Model name or GGUF file |
| `--model_type` | auto-detect | `hf` or `gguf` |
| `--max_new_tokens` | `512` | Max tokens to generate |
| `--temperature` | `1.0` | Generation temperature |
| `--prompt_dir` | `./prompt_files` | Prompt directory |
| `--output_dir` | `./qmsum_outputs` | Output directory |
| `--disable_cache` | `False` | Disable KV cache (HF only) |
| `--n_gpu_layers` | `-1` | GGUF: GPU layers (-1=all, 0=CPU) |
| `--n_ctx` | `332768` | GGUF: Context window size |
| `--skip_eval` | `False` | Skip evaluation |
| `--results_file` | auto | Custom results JSON path |

## Output

- Generated summaries: `qmsum_outputs/qmsum_test_*.txt`
- Results JSON: `results_YYYYMMDD_HHMMSS.json` (contains ROUGE scores)

## Troubleshooting

- **DynamicCache error**: Add `--disable_cache`
- **Missing dependencies**: Install required packages (see Installation)
- **CUDA not available**: Script auto-falls back to CPU
- **GGUF model not found**: Place files in `./gguf/` or use full path
