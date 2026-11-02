import pandas as pd
from collections import defaultdict
import pickle
import os

# --- 配置文件路径 ---
DRUG_DATA_FILE = 'drug_pairs_targets_interactions-已去重.txt'
FILTERED_ZSCORE_FILE = 'filtered_zscores.csv'
ZSCORE_KB_PKL_FILE = 'zscore_knowledge_base.pkl'  # 新增：保存知识库的pkl文件
OUTPUT_PREDICTIONS_FILE = 'drug_pair_predictions.csv'
OUTPUT_EVALUATION_FILE = 'evaluation_report.csv'

# --- 1. 加载并预处理 Z-score 数据 (或从缓存加载) ---
zscore_kb_path = ZSCORE_KB_PKL_FILE
if os.path.exists(zscore_kb_path):
    print(f"从缓存加载 Z-score 知识库: {zscore_kb_path} ...")
    with open(zscore_kb_path, 'rb') as f:
        zscore_kb = pickle.load(f)
    print("Z-score 知识库加载完成。")
else:
    print(f"缓存文件 {zscore_kb_path} 不存在，正在构建知识库...")

    # 强制将第一行作为列名读取
    zscore_df = pd.read_csv(FILTERED_ZSCORE_FILE, sep=',', header=0)
    print(f"成功读取 Z-score 数据，列名为: {list(zscore_df.columns)}")

    # 构建 Z-score 知识库 (优化版)
    # 一个嵌套字典: {(node1, node2): {context: z_score}}
    zscore_kb = defaultdict(dict)
    for _, row in zscore_df.iterrows():
        # 确保列名正确
        node1, node2, context, z_score = row['node1'], row['node2'], row['context'], row['z_score']
        # 由于组合无方向性，同时存储 (A,B) 和 (B,A)
        zscore_kb[(node1, node2)][context] = z_score
        zscore_kb[(node2, node1)][context] = z_score

    print(f"Z-score 知识库构建完成，包含 {len(zscore_kb)} 个独特的靶点对。")

    # 保存到pkl文件，供下次使用
    print(f"正在将知识库保存到缓存文件: {zscore_kb_path} ...")
    with open(zscore_kb_path, 'wb') as f:
        pickle.dump(zscore_kb, f)
    print("知识库缓存已保存。")


# --- 2. 定义预测函数 ---
def predict_ddi_from_targets(drug1_targets_str, drug2_targets_str, zscore_kb):
    """
    根据药物的靶点列表，预测可能的 DDI。
    """
    targets1 = set([t.strip() for t in drug1_targets_str.split(';')])
    targets2 = set([t.strip() for t in drug2_targets_str.split(';')])

    # 存储每个 context 的累计 z_score
    context_scores = defaultdict(float)

    # 遍历所有跨药物的靶点对组合
    for t1 in targets1:
        for t2 in targets2:
            # 从预处理好的字典中快速查询
            pair_key = (t1, t2)
            if pair_key in zscore_kb:
                for context, z_score in zscore_kb[pair_key].items():
                    context_scores[context] += z_score

    # 按累计 z_score 降序排列
    sorted_contexts = sorted(context_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_contexts


# --- 3. 分批处理大型药物数据集 ---
print("正在批量处理药物对数据...")

# 用于存储所有结果
all_predictions = []

# 分块读取，以节省内存
chunk_size = 10000
total_processed = 0

for chunk_df in pd.read_csv(DRUG_DATA_FILE, sep='\t', chunksize=chunk_size):
    chunk_predictions = []
    for _, row in chunk_df.iterrows():
        drug1, drug2, known_ddi = row['drug1'], row['drug2'], row['归一化']
        targets1_str, targets2_str = row['drug1_targets'], row['drug2_targets']

        # 获取 Top-K 预测
        all_scores = predict_ddi_from_targets(targets1_str, targets2_str, zscore_kb)
        top_5_predictions = all_scores[:5]

        # 记录预测结果
        predicted_contexts = [pred[0] for pred in top_5_predictions]
        predicted_scores = [pred[1] for pred in top_5_predictions]

        # 计算真实 DDI 是否在 Top-5 中
        is_in_top5 = 1 if any(known_ddi in pred for pred in predicted_contexts) else 0

        chunk_predictions.append({
            'drug1': drug1,
            'drug2': drug2,
            'known_ddi': known_ddi,
            'is_known_in_top5': is_in_top5,
            'top5_predicted_ddis': '; '.join(predicted_contexts),
            'top5_scores': '; '.join(map(str, predicted_scores))
        })

    all_predictions.extend(chunk_predictions)
    total_processed += len(chunk_predictions)
    print(f"已处理 {total_processed} 对药物...")

print("\n批量预测完成。")

# --- 4. 保存预测结果 ---
print("正在保存预测结果...")
predictions_df = pd.DataFrame(all_predictions)
predictions_df.to_csv(OUTPUT_PREDICTIONS_FILE, index=False)
print(f"预测结果已保存至: {OUTPUT_PREDICTIONS_FILE}")

# --- 5. 评估模型性能 (合并脚本2的逻辑) ---
print("\n正在评估模型性能...")

# 获取所有预测结果
total_pairs = len(predictions_df)
correct_predictions_all = predictions_df['is_known_in_top5'].sum()

# 1. 包括所有案例的 Accuracy@5
accuracy_all = correct_predictions_all / total_pairs

# 2. 只计算 top5 非空的案例的 Accuracy@5
# 使用 str.len() > 0 来检查，这比 != '' 更可靠
non_empty_mask = predictions_df['top5_predicted_ddis'].str.len() > 0
non_empty_predictions_df = predictions_df[non_empty_mask]

total_non_empty_pairs = len(non_empty_predictions_df)
correct_predictions_non_empty = non_empty_predictions_df['is_known_in_top5'].sum()
accuracy_non_empty = correct_predictions_non_empty / total_non_empty_pairs if total_non_empty_pairs > 0 else 0

# 3. 计算无法预测的案例数量
unpredictable_cases = total_pairs - total_non_empty_pairs

# --- 6. 生成并保存详细的评估报告 ---
evaluation_report = [
    {'metric': 'Total Drug Pairs Evaluated', 'value': total_pairs},
    {'metric': 'Correct Predictions (Top-5, Overall)', 'value': correct_predictions_all},
    {'metric': 'Overall Accuracy@5', 'value': f"{accuracy_all:.4f}"},
    {'metric': 'Predictable Pairs (Non-Empty Top-5)', 'value': total_non_empty_pairs},
    {'metric': 'Correct Predictions (Top-5, Predictable Subset)', 'value': correct_predictions_non_empty},
    {'metric': 'Accuracy@5 (On Predictable Subset)', 'value': f"{accuracy_non_empty:.4f}"},
    {'metric': 'Unpredictable Pairs (Empty Top-5)', 'value': unpredictable_cases},
    {'metric': 'Coverage Rate', 'value': f"{(total_non_empty_pairs / total_pairs) * 100:.2f}%"},
    {'metric': 'Unpredictable Rate', 'value': f"{(unpredictable_cases / total_pairs) * 100:.2f}%"}
]

evaluation_df = pd.DataFrame(evaluation_report)
evaluation_df.to_csv(OUTPUT_EVALUATION_FILE, index=False)
print(f"详细评估报告已保存至: {OUTPUT_EVALUATION_FILE}")

# --- 7. 打印摘要 ---
print("\n--- 预测与评估完成 ---")
print(f"1. 总体 Accuracy@5 (包括无法预测的案例): {accuracy_all:.4f} ({correct_predictions_all}/{total_pairs})")
print(f"2. 有效预测 Accuracy@5 (仅针对有预测结果的案例): {accuracy_non_empty:.4f} ({correct_predictions_non_empty}/{total_non_empty_pairs})")
print(f"3. 无法预测的案例数: {unpredictable_cases}")

print("\n--- 分析 ---")
print(f"模型能够提供预测的案例占比: {total_non_empty_pairs / total_pairs * 100:.2f}%")
print(f"模型无法提供预测的案例占比: {unpredictable_cases / total_pairs * 100:.2f}%")

# --- 可选：打印一些示例 ---
print("\n--- 示例预测结果 (前3行) ---")
print(predictions_df[['drug1', 'drug2', 'known_ddi', 'is_known_in_top5', 'top5_predicted_ddis']].head(3).to_string(
    index=False))