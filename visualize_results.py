#!/usr/bin/env python3
"""
生成 Needle In A Haystack 測試結果的熱力圖
使用方式: python visualize_results.py --model_name "openai/gpt-oss-20b"
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import os
import glob
import argparse

def load_results(results_folder, model_name):
    """
    從結果文件夾加載測試結果
    
    Args:
        results_folder: 結果文件夾路徑
        model_name: 模型名稱，用於過濾文件
    
    Returns:
        DataFrame with columns: Document Depth, Context Length, Score
    """
    # 處理模型名稱中的斜線（如 openai/gpt-oss-20b）
    model_name_safe = model_name.replace('/', '-')
    
    # 尋找所有相關的 JSON 結果文件
    # 支持兩種路徑格式: results/model_name/... 或 results/model_name_...
    pattern1 = os.path.join(results_folder, model_name, "*.json")
    pattern2 = os.path.join(results_folder, model_name.replace('/', os.sep), "*.json")
    pattern3 = os.path.join(results_folder, f"{model_name_safe}*.json")
    
    json_files = []
    for pattern in [pattern1, pattern2, pattern3]:
        json_files.extend(glob.glob(pattern))
    
    if not json_files:
        print(f"警告: 在 {results_folder} 中找不到 {model_name} 的結果文件")
        print(f"嘗試的模式: {pattern1}, {pattern2}, {pattern3}")
        return None
    
    print(f"找到 {len(json_files)} 個結果文件")
    
    # 讀取所有結果
    data = []
    for file in json_files:
        try:
            with open(file, 'r') as f:
                json_data = json.load(f)
                document_depth = json_data.get("depth_percent", None)
                context_length = json_data.get("context_length", None)
                score = json_data.get("score", None)
                
                if document_depth is not None and context_length is not None and score is not None:
                    data.append({
                        "Document Depth": document_depth,
                        "Context Length": context_length,
                        "Score": score
                    })
        except Exception as e:
            print(f"讀取文件 {file} 時出錯: {e}")
    
    if not data:
        print("錯誤: 沒有有效的數據")
        return None
    
    df = pd.DataFrame(data)
    print(f"\n成功加載 {len(df)} 條測試結果")
    print(f"Context Length 範圍: {df['Context Length'].min()} - {df['Context Length'].max()}")
    print(f"Document Depth 範圍: {df['Document Depth'].min()}% - {df['Document Depth'].max()}%")
    print(f"平均分數: {df['Score'].mean():.2f}")
    
    return df

def create_heatmap(df, model_name, output_file='heatmap.png'):
    """
    創建熱力圖
    
    Args:
        df: 包含測試結果的 DataFrame
        model_name: 模型名稱
        output_file: 輸出文件名
    """
    # 創建數據透視表
    # 如果有多次測試，取平均值
    pivot_table = pd.pivot_table(
        df, 
        values='Score', 
        index=['Document Depth', 'Context Length'], 
        aggfunc='mean'
    ).reset_index()
    
    pivot_table = pivot_table.pivot(
        index="Document Depth", 
        columns="Context Length", 
        values="Score"
    )
    
    # 按照索引排序（從小到大）
    pivot_table = pivot_table.sort_index(ascending=True)
    pivot_table = pivot_table.sort_index(axis=1, ascending=True)
    
    print(f"\n透視表大小: {pivot_table.shape[0]} 個深度 × {pivot_table.shape[1]} 個長度")
    
    # 創建自定義配色（紅色=低分，黃色=中等，綠色=高分）
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", 
        ["#F0496E", "#EBB839", "#0CD79F"]  # 紅 -> 黃 -> 綠
    )
    
    # 創建圖表
    plt.figure(figsize=(20, 10))
    
    # 繪製熱力圖
    ax = sns.heatmap(
        pivot_table,
        annot=True,  # 顯示數值
        fmt=".0f",   # 數值格式（整數）
        cmap=cmap,
        cbar_kws={'label': 'Score (0-10)'},
        vmin=0,      # 最小值 0
        vmax=10,     # 最大值 10
        linewidths=0.5,  # 格子邊框
        linecolor='white'
    )
    
    # 標題和標籤
    plt.title(
        f'Pressure Testing {model_name} via "Needle In A Haystack"\n'
        f'Fact Retrieval Across Context Lengths & Document Depth',
        fontsize=16,
        pad=20
    )
    plt.xlabel('Context Length (# Tokens)', fontsize=14)
    plt.ylabel('Document Depth (%)', fontsize=14)
    
    # 旋轉 x 軸標籤以防重疊
    plt.xticks(rotation=45, ha='right')
    
    # 調整布局
    plt.tight_layout()
    
    # 保存圖片
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ 熱力圖已保存至: {output_file}")
    
    # 顯示圖片
    try:
        plt.show()
    except:
        print("(無法顯示圖片，但已保存)")

def main():
    parser = argparse.ArgumentParser(
        description='生成 Needle In A Haystack 測試結果的熱力圖'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='openai/gpt-oss-20b',
        help='模型名稱（必須與測試時使用的名稱一致）'
    )
    parser.add_argument(
        '--results_folder',
        type=str,
        default='results',
        help='結果文件夾路徑'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='輸出文件名（默認：model_name_heatmap.png）'
    )
    
    args = parser.parse_args()
    
    # 設置輸出文件名
    if args.output is None:
        safe_name = args.model_name.replace('/', '_').replace(' ', '_')
        args.output = f'{safe_name}_heatmap.png'
    
    print(f"🔍 正在加載 {args.model_name} 的測試結果...")
    print(f"📁 結果文件夾: {args.results_folder}")
    
    # 加載數據
    df = load_results(args.results_folder, args.model_name)
    
    if df is None or len(df) == 0:
        print("\n❌ 錯誤：沒有找到有效的測試結果")
        print("\n💡 請確保:")
        print(f"   1. 已經運行過測試（使用 --model_name '{args.model_name}'）")
        print(f"   2. 結果保存在 '{args.results_folder}' 目錄下")
        print(f"   3. 模型名稱完全匹配")
        return
    
    # 創建熱力圖
    print(f"\n📊 正在生成熱力圖...")
    create_heatmap(df, args.model_name, args.output)
    
    print("\n✨ 完成！")

if __name__ == "__main__":
    main()
