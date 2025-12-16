# LongBench Evaluation

Tool for running LongBench QMSum evaluations with HuggingFace or GGUF models, including automatic ROUGE score computation.

## Installation

### Basic Installation

For CUDA 12.6 (RTX 3090 and compatible GPUs):
```bash
# Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.6)
# Note: torchaudio not available for Python 3.13, but not needed for this project
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install transformers datasets evaluate
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
# Note: For PyTorch with CUDA, use the command above instead
```

### GGUF Models

For GGUF models with CUDA support (RTX 3090):
```bash
# With CUDA support for RTX 3090
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

For CPU-only or Metal (macOS):
```bash
# CPU only
pip install llama-cpp-python

# Metal (macOS)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
```

### GPU Requirements

- **RTX 3090**: 24GB VRAM, CUDA Compute Capability 8.6
- **CUDA Version**: 12.6 (driver supports CUDA 12.6)
- **PyTorch**: Use CUDA 12.1 build (compatible with CUDA 12.6)

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
