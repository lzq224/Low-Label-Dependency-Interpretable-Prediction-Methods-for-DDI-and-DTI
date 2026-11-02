import pandas as pd

# --- 配置文件路径 ---
PREDICTIONS_FILE = 'drug_pair_predictions.csv'
UNPREDICTABLE_OUTPUT_FILE = 'unpredictable_drug_pairs.csv'

print("正在加载完整的预测结果...")
predictions_df = pd.read_csv(PREDICTIONS_FILE)

print("\n检查 top5_predicted_ddis 列的数据类型和示例值:")
print(predictions_df['top5_predicted_ddis'].dtype)
print(predictions_df['top5_predicted_ddis'].head(10))

print("\n检查哪些行的 top5_predicted_ddis 是 NaN (空单元格):")
nan_mask = predictions_df['top5_predicted_ddis'].isna()
nan_count = nan_mask.sum()
print(f"NaN 值数量: {nan_count}")

print("\n检查哪些行的 top5_predicted_ddis 是空字符串 (''):")
empty_str_mask = predictions_df['top5_predicted_ddis'] == ''
empty_str_count = empty_str_mask.sum()
print(f"空字符串 ('') 数量: {empty_str_count}")

print("\n检查哪些行的 top5_predicted_ddis 长度为 0 (包括 NaN 和空字符串):")
len_zero_mask = predictions_df['top5_predicted_ddis'].str.len() == 0
len_zero_count = len_zero_mask.sum()
print(f"长度为 0 的数量: {len_zero_count}")

# 合并所有“无法预测”的条件
unpredictable_mask = nan_mask | empty_str_mask

num_unpredictable = unpredictable_mask.sum()

print(f"\n根据综合条件 (NaN 或 '')，共发现 {num_unpredictable} 个无法预测的药物对案例。")

if num_unpredictable > 0:
    unpredictable_df = predictions_df[unpredictable_mask]
    print(f"正在将这些案例保存到 '{UNPREDICTABLE_OUTPUT_FILE}' ...")
    unpredictable_df.to_csv(UNPREDICTABLE_OUTPUT_FILE, index=False)

    print(f"无法预测的案例已保存至: {UNPREDICTABLE_OUTPUT_FILE}")

    print("\n--- 无法预测案例的前几行预览 ---")
    print(unpredictable_df[['drug1', 'drug2', 'known_ddi']].head(10).to_string(index=False))
else:
    print("没有发现无法预测的案例。")

print("\n分析完成。")