import pandas as pd
from collections import defaultdict
import pickle
import os
import numpy as np

# --- 配置文件路径 ---
ZSCORE_KB_PKL_FILE = 'zscore_knowledge_base.pkl'

print("正在初始化 DDI 预测引擎...")

# --- 1. 加载 Z-score 知识库 (必须已存在) ---
if not os.path.exists(ZSCORE_KB_PKL_FILE):
    print(f"错误: 缓存文件 {ZSCORE_KB_PKL_FILE} 不存在。请先运行主训练脚本以构建知识库。")
    exit()

print(f"从缓存加载 Z-score 知识库: {ZSCORE_KB_PKL_FILE} ...")
with open(ZSCORE_KB_PKL_FILE, 'rb') as f:
    zscore_kb = pickle.load(f)
print("Z-score 知识库加载完成。")
print("预测引擎已就绪。\n")


# --- 2. 定义预测函数 ---
def predict_ddi_from_targets(drug1_targets_str, drug2_targets_str, zscore_kb):
    """
    根据药物的靶点列表，预测可能的 DDI。
    """
    targets1 = set([t.strip() for t in drug1_targets_str.split(';') if t.strip()])
    targets2 = set([t.strip() for t in drug2_targets_str.split(';') if t.strip()])

    # 存储每个 context 的累计 z_score
    context_scores = defaultdict(float)

    # 遍历所有跨药物的靶点对组合
    for t1 in targets1:
        for t2 in targets2:
            # 从预处理好的字典中快速查询
            pair_key = (t1, t2)
            if pair_key in zscore_kb:
                for context, z_score in zscore_kb[pair_key].items():
                    # 只累加正值，因为负值代表负相关，不应计入“风险”
                    if z_score > 0:
                        context_scores[context] += z_score

    # 按累计 z_score 降序排列
    sorted_contexts = sorted(context_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_contexts


# --- 3. 定义归一化函数 ---
def normalize_to_probabilities(scores):
    """
    将 Z-scores 转换为相对概率。
    使用 Softmax 函数将分数转换为一个和为 1 的概率分布。
    """
    if not scores:
        return []
    # 为了数值稳定，减去最大值
    scores_array = np.array(scores)
    scores_shifted = scores_array - np.max(scores_array)
    exp_scores = np.exp(scores_shifted)
    sum_exp_scores = np.sum(exp_scores)
    probabilities = exp_scores / sum_exp_scores
    return probabilities


# --- 4. 主交互循环 ---
print("--- DDI Top-5 预测器 (带相对概率) ---")
print("请输入药物的靶点列表，靶点之间用分号 ';' 分隔。")
print("输入 'quit' 或 'exit' 退出程序。\n")

while True:
    try:
        drug1_targets_input = input("请输入药物 1 的靶点列表: ").strip()
        if drug1_targets_input.lower() in ['quit', 'exit']:
            break

        drug2_targets_input = input("请输入药物 2 的靶点列表: ").strip()
        if drug2_targets_input.lower() in ['quit', 'exit']:
            break

        print("\n正在预测...")
        all_scores = predict_ddi_from_targets(drug1_targets_input, drug2_targets_input, zscore_kb)

        if not all_scores:
            print("\n[警告] 无法基于现有知识库预测此药物对的相互作用。可能的原因是：")
            print("1. 两种药物没有共同或相关的靶点。")
            print("2. 这些靶点的组合在知识库中没有记录。")
        else:
            top_5_predictions = all_scores[:5]

            # 提取 Top-5 的 DDI 类型和原始 Z-scores
            ddi_types, raw_scores = zip(*top_5_predictions)

            # 计算相对概率
            relative_probs = normalize_to_probabilities(raw_scores)

            print("\n--- Top-5 预测的药物相互作用 (按相对概率排序) ---")
            for i, (ddi_type, raw_score, prob) in enumerate(zip(ddi_types, raw_scores, relative_probs), 1):
                print(f"{i}. {ddi_type}")
                print(f"   原始得分 (Z-score): {raw_score:.4f}")
                print(f"   相对概率: {prob:.4f} ({prob * 100:.2f}%)\n")

    except KeyboardInterrupt:
        print("\n\n程序被用户中断。再见！")
        break
    except Exception as e:
        print(f"\n发生错误: {e}\n")

print("\n感谢使用 DDI 预测器。再见！")