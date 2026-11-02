import pandas as pd
import numpy as np

# --- 配置 ---
INPUT_FILE = '重复三元组统计结果（新）.txt'  # 假设原始文件是 TSV 格式
OUTPUT_ALL_ZSCORES_FILE = 'all_zscores.csv'


# 读取数据
print(f"正在读取原始数据: {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE, sep='\t')  # 如果文件是逗号分隔，请将 '\t' 改为 ','

# 按 (node1, node2) 分组，并计算每组的均值和标准差
group_stats = df.groupby(['node1', 'node2'])['count'].agg(['mean', 'std']).reset_index()
group_stats.columns = ['node1', 'node2', 'mean_count_in_group', 'std_count_in_group']

# 将统计信息合并回原数据框
df_with_stats = df.merge(group_stats, on=['node1', 'node2'])

# 计算 Z-score，处理标准差为0的情况以避免除零错误
df_with_stats['z_score'] = (df_with_stats['count'] - df_with_stats['mean_count_in_group']) / df_with_stats['std_count_in_group']
df_with_stats['z_score'].replace([np.inf, -np.inf], np.nan, inplace=True)

df_with_stats.to_csv(OUTPUT_ALL_ZSCORES_FILE, index=False)
print(f"所有记录的 Z-score 结果已保存至: {OUTPUT_ALL_ZSCORES_FILE}")
print("处理完成，生成 all_zscores.csv 文件。")