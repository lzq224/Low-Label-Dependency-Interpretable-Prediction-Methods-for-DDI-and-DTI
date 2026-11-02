import pandas as pd

CHEMBL_PATH = r"D:\E盘模型暂存\chembl_bioactivity_standardized.parquet"
UNIPROT_TXT = r"D:\E盘模型暂存\chembl_uniprot_ids_unique.txt"

df = pd.read_parquet(CHEMBL_PATH, columns=['uniprot_id'])
unique_ids = sorted(df['uniprot_id'].dropna().unique().tolist())

with open(UNIPROT_TXT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(unique_ids))

print(f"✅ 已导出 {len(unique_ids):,} 个唯一UniProt ID → {UNIPROT_TXT}")