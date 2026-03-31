# 如何測試並生成熱力圖

## 第一步：運行完整測試

### 快速測試（推薦先測試）
測試幾個 context length 和 depth 組合，確保設置正確：

```bash
cd /home/os-i-kai.dong/orchestrator/benchmark/LLMTest_NeedleInAHaystack

python3 -m needlehaystack.run \
    --model_name "openai/gpt-oss-20b" \
    --base_url "http://localhost:8066/v1" \
    --document_depth_percents "[10, 30, 50, 70, 90]" \
    --context_lengths "[2000, 4000, 8000]"
```

### 完整測試（類似圖片中的結果）
測試多個 context length 和 depth，使用 sigmoid 分佈：

```bash
python3 -m needlehaystack.run \
    --model_name "openai/gpt-oss-20b" \
    --base_url "http://localhost:8066/v1" \
    --context_lengths_min 1000 \
    --context_lengths_max 16000 \
    --context_lengths_num_intervals 15 \
    --document_depth_percent_min 0 \
    --document_depth_percent_max 100 \
    --document_depth_percent_intervals 35 \
    --document_depth_percent_interval_type "sigmoid"
```

### 超長 Context 測試（類似 Claude 2.1 的 200K 測試）
如果你的模型支持長 context：

```bash
python3 -m needlehaystack.run \
    --model_name "your-long-context-model" \
    --base_url "http://localhost:YOUR_PORT/v1" \
    --context_lengths_min 1000 \
    --context_lengths_max 200000 \
    --context_lengths_num_intervals 35 \
    --document_depth_percent_min 0 \
    --document_depth_percent_max 100 \
    --document_depth_percent_intervals 35 \
    --document_depth_percent_interval_type "sigmoid"
```

## 第二步：結果會保存在哪裡

測試完成後，結果會保存在：
- **JSON 結果**: `results/openai/gpt-oss-20b_len_XXXX_depth_YYYY_results.json`
- **Context 文本**: `contexts/openai/gpt-oss-20b_len_XXXX_depth_YYYY_context.txt`

## 第三步：生成可視化圖表

### 方法 1：使用 Jupyter Notebook（推薦）

```bash
# 安裝 jupyter 如果還沒有
pip install jupyter matplotlib seaborn pandas numpy

# 啟動 Jupyter Notebook
cd /home/os-i-kai.dong/orchestrator/benchmark/LLMTest_NeedleInAHaystack
jupyter notebook viz/CreateVizFromLLMTesting.ipynb
```

在 notebook 中：
1. 修改 `model_name` 變量為你的模型名稱（如 `"openai/gpt-oss-20b"`）
2. 運行所有 cells
3. 會生成熱力圖

### 方法 2：使用命令行腳本

我可以幫你創建一個獨立的 Python 腳本來生成圖表。

## 參數說明

### Context Length 相關
- `--context_lengths_min`: 最小 context length（如 1000）
- `--context_lengths_max`: 最大 context length（如 16000 或 200000）
- `--context_lengths_num_intervals`: 測試多少個不同的 context length（如 15 或 35）
- `--context_lengths`: 直接指定要測試的長度列表（如 `"[1000, 2000, 4000, 8000]"`）

### Document Depth 相關
- `--document_depth_percent_min`: 最小深度百分比（通常是 0）
- `--document_depth_percent_max`: 最大深度百分比（通常是 100）
- `--document_depth_percent_intervals`: 測試多少個不同的深度（如 35）
- `--document_depth_percents`: 直接指定深度列表（如 `"[10, 30, 50, 70, 90]"`）
- `--document_depth_percent_interval_type`: 分佈類型
  - `"linear"`: 線性分佈（均勻間隔）
  - `"sigmoid"`: S型分佈（更密集地測試中間和邊緣）

### 其他有用參數
- `--num_concurrent_requests`: 並發請求數（默認 1，可以加速但注意資源）
- `--save_results`: 是否保存結果（默認 True）
- `--save_contexts`: 是否保存完整 context（默認 True，但會佔用大量空間）
- `--seconds_to_sleep_between_completions`: 請求之間的等待時間（避免超載）

## 實用技巧

### 1. 分階段測試
```bash
# 階段 1: 快速測試確認配置
python3 -m needlehaystack.run \
    --model_name "openai/gpt-oss-20b" \
    --base_url "http://localhost:8066/v1" \
    --context_lengths "[2000, 8000]" \
    --document_depth_percents "[50]"

# 階段 2: 中等規模測試
python3 -m needlehaystack.run \
    --model_name "openai/gpt-oss-20b" \
    --base_url "http://localhost:8066/v1" \
    --context_lengths_min 1000 \
    --context_lengths_max 16000 \
    --context_lengths_num_intervals 8 \
    --document_depth_percent_intervals 10

# 階段 3: 完整測試
python3 -m needlehaystack.run \
    --model_name "openai/gpt-oss-20b" \
    --base_url "http://localhost:8066/v1" \
    --context_lengths_num_intervals 35 \
    --document_depth_percent_intervals 35
```

### 2. 不保存 Context（節省空間）
```bash
python3 -m needlehaystack.run \
    --model_name "openai/gpt-oss-20b" \
    --base_url "http://localhost:8066/v1" \
    --context_lengths_num_intervals 35 \
    --document_depth_percent_intervals 35 \
    --save_contexts false
```

### 3. 測試多個模型
```bash
# 模型 1
python3 -m needlehaystack.run \
    --model_name "model-1" \
    --base_url "http://localhost:8001/v1" \
    --context_lengths_num_intervals 15 \
    --document_depth_percent_intervals 15

# 模型 2
python3 -m needlehaystack.run \
    --model_name "model-2" \
    --base_url "http://localhost:8002/v1" \
    --context_lengths_num_intervals 15 \
    --document_depth_percent_intervals 15

# 然後在 Jupyter notebook 中比較兩者的結果
```

## 預估測試時間

假設每次 API 調用需要 1-2 秒：
- 小規模測試（5 lengths × 5 depths）: ~50 次調用 = 1-2 分鐘
- 中等測試（15 lengths × 15 depths）: ~225 次調用 = 5-10 分鐘
- 大規模測試（35 lengths × 35 depths）: ~1225 次調用 = 30-60 分鐘

使用 `--num_concurrent_requests 5` 可以加速約 5 倍，但要確保服務器能處理。

## 故障排除

### 測試中斷了怎麼辦？
結果是實時保存的，已完成的測試不會丟失。只需重新運行相同的命令，程序會跳過已有結果的組合。

### 內存不足
- 減少 `context_lengths_max`
- 設置 `--save_contexts false`
- 減少 `--num_concurrent_requests`

### 生成圖表時找不到數據
確保結果文件在 `results/` 目錄下，文件名格式為 `{model_name}_len_{length}_depth_{depth}_results.json`
