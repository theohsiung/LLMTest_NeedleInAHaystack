# 使用本地模型測試指南 (Local Model Testing Guide)

## 概述

本專案已擴展支援本地模型測試。你可以使用任何提供 OpenAI 兼容 API 的本地模型服務，例如：
- vLLM
- Ollama
- LocalAI
- LM Studio
- Text Generation WebUI (with OpenAI extension)

## 快速開始

### 1. 啟動你的本地模型服務

確保你的本地模型已經啟動並提供 OpenAI 兼容的 API 接口。

#### 使用 vLLM 示例：
```bash
python -m vllm.entrypoints.openai.api_server \
    --model your-model-path \
    --host 0.0.0.0 \
    --port 8000
```

#### 使用 Ollama 示例：
```bash
# Ollama 通常在 http://localhost:11434 提供 OpenAI 兼容的 API
# 格式：http://localhost:11434/v1
ollama serve
```

### 2. 設置環境變量

```bash
# 設置本地模型的 API 端點
export NIAH_MODEL_BASE_URL="http://localhost:8000/v1"

# API key 是可選的（對於本地模型）
export NIAH_MODEL_API_KEY="dummy-key"

# 如果使用 OpenAI 作為評估器，設置評估器的 API key
export NIAH_EVALUATOR_API_KEY="your-openai-api-key"

# 如果使用本地模型作為評估器（推薦！），可以不設置 NIAH_EVALUATOR_API_KEY
# 系統會自動使用 NIAH_MODEL_BASE_URL
```

### 3. 運行測試

#### 使用本地模型進行測試和評估（推薦）：
```bash
# 使用同一個本地模型作為測試和評估器
needlehaystack.run_test \
    --provider local \
    --evaluator local \
    --model_name "your-model-name" \
    --evaluator_model_name "your-model-name" \
    --base_url "http://localhost:8000/v1" \
    --document_depth_percents "[50]" \
    --context_lengths "[2000]"
```

#### 基本測試命令（使用 OpenAI 評估器）：
```bash
needlehaystack.run_test \
    --provider local \
    --model_name "your-model-name" \
    --base_url "http://localhost:8000/v1" \
    --document_depth_percents "[50]" \
    --context_lengths "[2000]"
```

#### 多個深度和長度測試：
```bash
needlehaystack.run_test \
    --provider local \
    --evaluator local \
    --model_name "your-model-name" \
    --evaluator_model_name "your-model-name" \
    --base_url "http://localhost:8000/v1" \
    --document_depth_percents "[10, 30, 50, 70, 90]" \
    --context_lengths "[1000, 2000, 4000, 8000]"
```

#### 完整的 Haystack 測試：
```bash
needlehaystack.run_test \
    --provider local \
    --evaluator local \
    --model_name "your-model-name" \
    --evaluator_model_name "your-model-name" \
    --base_url "http://localhost:8000/v1" \
    --context_lengths_min 1000 \
    --context_lengths_max 16000 \
    --context_lengths_num_intervals 10 \
    --document_depth_percent_min 0 \
    --document_depth_percent_max 100 \
    --document_depth_percent_intervals 10
```

## 參數說明

### 必需參數：
- `--provider local` - 使用本地模型 provider
- `--model_name` - 你的模型名稱（需要與 API 服務器中配置的名稱匹配）

### 推薦參數：
- `--evaluator local` - 使用本地模型作為評估器（省錢又方便！）
- `--evaluator_model_name` - 評估器使用的模型名稱（通常和 model_name 一樣）

### 可選參數：
- `--base_url` - 本地 API 端點 URL（也可以通過 `NIAH_MODEL_BASE_URL` 環境變量設置）
- `--evaluator_base_url` - 評估器的 API 端點（如果不設置，會使用 `base_url`）
- `--document_depth_percents` - 在文檔中放置 needle 的深度百分比列表
- `--context_lengths` - 要測試的上下文長度列表
- `--save_results` - 是否保存結果（默認 True）
- `--save_contexts` - 是否保存上下文（默認 True，注意文件可能很大）

## 針對不同本地服務的配置

### vLLM
```bash
export NIAH_MODEL_BASE_URL="http://localhost:8000/v1"
needlehaystack.run_test \
    --provider local \
    --evaluator local \
    --model_name "your-model" \
    --evaluator_model_name "your-model" \
    --base_url "http://localhost:8000/v1"
```

### Ollama
```bash
export NIAH_MODEL_BASE_URL="http://localhost:11434/v1"
needlehaystack.run_test \
    --provider local \
    --evaluator local \
    --model_name "llama2" \
    --evaluator_model_name "llama2" \
    --base_url "http://localhost:11434/v1"
```

### LM Studio
```bash
# LM Studio 通常使用端口 1234
export NIAH_MODEL_BASE_URL="http://localhost:1234/v1"
needlehaystack.run_test \
    --provider local \
    --evaluator local \
    --model_name "local-model" \
    --evaluator_model_name "local-model" \
    --base_url "http://localhost:1234/v1"
```

## 故障排除

### 1. 連接錯誤
確保：
- 本地模型服務正在運行
- base_url 正確（包括 `/v1` 後綴）
- 防火牆允許連接

### 2. 模型名稱不匹配
確保 `--model_name` 與你的 API 服務器中配置的模型名稱完全匹配。

### 3. Tokenizer 問題
本地模型使用 `cl100k_base` (GPT-3.5/4 tokenizer) 作為近似。如果你的模型使用不同的 tokenizer，token 計數可能不完全準確。

### 4. 內存不足
如果測試較長的上下文：
- 減少 `--context_lengths_num_intervals`
- 減少 `--num_concurrent_requests`（默認為 1）

## 結果分析

測試完成後，結果會保存在 `results/` 目錄中。你可以使用提供的可視化腳本來分析結果：

```bash
# 查看 LLMNeedleInHaystackVisualization.ipynb
jupyter notebook viz/LLMNeedleInHaystackVisualization.ipynb
```

## 示例：完整測試流程

```bash
# 1. 啟動 vLLM 服務器
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/model \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 16384

# 2. 在另一個終端，設置環境變量
export NIAH_MODEL_BASE_URL="http://localhost:8000/v1"

# 3. 運行快速測試（使用本地模型作為評估器）
cd /home/os-i-kai.dong/orchestrator/benchmark/LLMTest_NeedleInAHaystack
needlehaystack.run_test \
    --provider local \
    --evaluator local \
    --model_name "your-model" \
    --evaluator_model_name "your-model" \
    --document_depth_percents "[50]" \
    --context_lengths "[2000]"

# 4. 如果快速測試通過，運行完整測試
needlehaystack.run_test \
    --provider local \
    --evaluator local \
    --model_name "your-model" \
    --evaluator_model_name "your-model" \
    --context_lengths_min 1000 \
    --context_lengths_max 16000 \
    --context_lengths_num_intervals 15 \
    --document_depth_percent_intervals 10
```

## 多 Needle 測試

```bash
needlehaystack.run_test \
    --provider local \
    --evaluator local \
    --model_name "your-model" \
    --evaluator_model_name "your-model" \
    --base_url "http://localhost:8000/v1" \
    --multi_needle True \
    --needles '["First secret fact.", "Second secret fact.", "Third secret fact."]' \
    --document_depth_percents "[40]" \
    --context_lengths "[4000]"
```

## 注意事項

1. **評估器**：
   - **推薦使用本地模型評估器**：設置 `--evaluator local`，這樣就不需要 OpenAI API key
   - 評估任務很簡單（比對答案與 GT），本地模型完全可以勝任
   - 如果使用 OpenAI 評估器，需要設置 `NIAH_EVALUATOR_API_KEY`
2. **Token 計數**：使用 GPT tokenizer 近似，可能與你的實際模型 tokenizer 有差異
3. **性能**：測試可能需要較長時間，特別是對於較大的上下文長度
4. **資源**：確保你的硬件有足夠的 GPU 內存來處理較長的上下文

## 技術細節

新增的模組：

### Provider
**`LocalModel` provider** 位於 [needlehaystack/providers/local.py](needlehaystack/providers/local.py)

實現了 `ModelProvider` 接口的所有必需方法：
- `evaluate_model()` - 調用本地模型 API
- `generate_prompt()` - 生成測試提示
- `encode_text_to_tokens()` - 將文本編碼為 tokens
- `decode_tokens()` - 將 tokens 解碼為文本

### Evaluator
**`LocalEvaluator`** 位於 [needlehaystack/evaluators/local.py](needlehaystack/evaluators/local.py)

使用本地模型進行評估：
- 評估器的任務很簡單：比對模型回答和 ground truth
- 使用與測試模型相同的評分標準（1-10 分）
- 可以使用和測試相同的模型或不同的本地模型
- 不需要額外的 API key 或外部服務

## 獲取幫助

如果遇到問題，請檢查：
1. 本地模型服務日誌
2. API 端點是否可訪問 (`curl http://localhost:8000/v1/models`)
3. 環境變量是否正確設置
