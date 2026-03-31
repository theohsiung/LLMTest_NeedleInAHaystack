# Needle In A Haystack - Internal Testing Guide

Internal guide for running long context retrieval tests on our models.

For detailed documentation, refer to the original repository: https://github.com/gkamradt/LLMTest_NeedleInAHaystack

## Quick Start (3 Steps)

### Step 0: Setup Environment

Create virtual environment and install dependencies:

```bash
uv venv .venv --python=3.12
source .venv/bin/activate
pip install needlehaystack
```

### Step 1: Start vLLM Endpoints

**Important**: You need to start **two** vLLM servers:
1. **OpenAI model (port 8066)** - Used as evaluator
2. **Test model (other port)** - The model being tested

Ensure `max-model-len` is longer than your test context length.

**Start OpenAI Evaluator Model (port 8066):**
```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model gpt-3.5-turbo \
  --host 0.0.0.0 \
  --port 8066 \
  --max-model-len 16384
```

**Start Test Model (port 8067):**
```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model Harrrrychen/gptoss-20b-orchestration_with_reasoning_pipeline_t2126 \
  --host 0.0.0.0 \
  --port 8067 \
  --max-model-len 50000 \
  --gpu-memory-utilization 0.95 \
  --enable-auto-tool-choice \
  --tool-call-parser=openai
```

### Step 2: Run Inference

Update `--base_url` to point to your **test model** endpoint (port 8067):

```bash
python3 -m needlehaystack.run \
  --model_name "Harrrrychen/gptoss-20b-orchestration_withlo_reasoning_v2_t1977_labelmask" \
  --base_url "http://localhost:8067/v1" \
  --context_lengths_min 1000 \
  --context_lengths_max 50000 \
  --context_lengths_num_intervals 15 \
  --document_depth_percent_min 0 \
  --document_depth_percent_max 100 \
  --document_depth_percent_intervals 10 \
  --document_depth_percent_interval_type "sigmoid"
```

**Note**: The evaluator model (port 8066) will be used automatically for scoring.

### Step 3: Visualize Results

```bash
python3 visualize_results.py \
  --model_name "Harrrrychen" \
  --output "harry_heatmap.png"
``` (e.g., GPU 0 for evaluator, GPU 1 for test model)
- Port 8066 is reserved for the OpenAI evaluator model
- Modify test model `--port` (e.g., 8067)
## Notes

- Adjust `CUDA_VISIBLE_DEVICES` based on available GPUs
- Modify `--port` and `--base_url` to match your setup
- Results will be saved in the `results/` directory
