import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# ===================== 配置区 =====================
TARGET_PATH = r"E:\research_project\DTI预测\bindingdb_compound_uniprot_affinity_with_gene.parquet"
CHEMBL_PATH = r"D:\E盘模型暂存\chembl_bioactivity_standardized_with_genesymbol.parquet"
OUTPUT_PATH = r"D:\E盘模型暂存\dti_merged_dataset.parquet"
# ==================================================


print("=" * 65)

# 1. 加载数据
print("\n📥 加载数据...")
target = pd.read_parquet(TARGET_PATH)
chembl = pd.read_parquet(CHEMBL_PATH)
print(f"   目标数据集: {len(target):>10,} 条 | 化合物: {target['compound_iso_smiles'].nunique():>8,} | 靶点: {target['uniprot_id'].nunique():>5,}")
print(f"   ChEMBL数据: {len(chembl):>10,} 条 | 化合物: {chembl['compound_iso_smiles'].nunique():>8,} | 靶点: {chembl['uniprot_id'].nunique():>5,}")

# 2. 统一列名 & 标记来源
target = target.assign(source='bindingdb')
chembl = chembl.assign(source='chembl')

common_cols = sorted(list(set(target.columns) & set(chembl.columns)))
target = target[common_cols]
chembl = chembl[common_cols]

# 3. 合并 + 优先级去重 (bindingdb > chembl)
print("\n🔄 合并 & 去重 (bindingdb优先)...")
merged = pd.concat([target, chembl], ignore_index=True)
merged.sort_values('source', ascending=False, inplace=True)  # bindingdb > chembl
merged.drop_duplicates(subset=['compound_iso_smiles_norm', 'uniprot_id'], keep='first', inplace=True)
merged.reset_index(drop=True, inplace=True)

# 记录过滤前各来源数量（用于最终统计）
pre_filter_counts = merged['source'].value_counts().to_dict()

# 4. 归一化后重新去重（不同互变异构体可能归并为同一SMILES）
before_dedup = len(merged)
merged.drop_duplicates(subset=['compound_iso_smiles_norm', 'uniprot_id'], keep='first', inplace=True)
merged.reset_index(drop=True, inplace=True)
after_dedup = len(merged)
new_dupes = before_dedup - after_dedup
if new_dupes > 0:
    print(f"\n🔄 互变异构体归一化后新增去重: {new_dupes:,} 条")
else:
    print(f"\n✅ 互变异构体归一化未产生新重复")

# 🧹 4.5 清洗 gene_symbol
print("\n🧹 清洗 gene_symbol...")
before_gene_clean = len(merged)

# 构建过滤掩码：保留有效记录
# 条件1: 非空（排除 NaN, None, "", "NA"）【保险起见再去除一遍】
gene_str = merged['gene_symbol'].astype(str).str.strip()
valid_mask = merged['gene_symbol'].notna() & (gene_str != '') & (gene_str.str.upper() != 'NA')& (gene_str.str.upper() != 'NAN')& (gene_str.str.upper() != 'NONE')
# 条件2: 不包含小写字母（只保留纯大写/数字/符号组成的基因名）
valid_mask &= ~gene_str.str.contains(r'[a-z]', na=False)
# 条件3: 长度 > 1（去除单字符）
valid_mask &= gene_str.str.len() > 1

merged = merged[valid_mask].copy()
merged.reset_index(drop=True, inplace=True)

removed_gene = before_gene_clean - len(merged)
print(f"   🗑️ 因 gene_symbol 无效移除 {removed_gene:,} 条 ({removed_gene / before_gene_clean * 100:.1f}%)")
print(f"   ✅ 剩余 {len(merged):,} 条有效记录")

# 5. 统计报告
post_filter_counts = merged['source'].value_counts().to_dict()

print(f"\n{'=' * 65}")
print(f"✅ 合并完成!")
print(f"{'=' * 65}")
print(f"   最终记录数:      {len(merged):>10,}")
print(f"   唯一化合物:      {merged['compound_iso_smiles'].nunique():>10,}")
print(f"   唯一靶点:        {merged['uniprot_id'].nunique():>10,}")
print(f"   唯一Gene Symbol: {merged['gene_symbol'].nunique():>7,}")
print(f"   ├─ 来自BindingDB: {post_filter_counts.get('bindingdb', 0):>10,}  (过滤前: {pre_filter_counts.get('bindingdb', 0):,})")
print(f"   └─ 来自ChEMBL:            {post_filter_counts.get('chembl', 0):>10,}  (过滤前: {pre_filter_counts.get('chembl', 0):,})")
print(f"\n📈 Affinity 统计:")
print(merged['affinity'].describe().to_string())

# 6. Gene Symbol 清洗后验证
print(f"\n✅ gene_symbol 清洗完成:")
print(f"   唯一Gene Symbol: {merged['gene_symbol'].nunique():>7,}")
sample_genes = merged['gene_symbol'].value_counts().head(5)
print(f"   Top5 高频基因: {dict(sample_genes)}")

# 7. SMILES 抽样验证
print(f"\n🧬 SMILES 验证 (随机抽样5条):")
sample = merged.sample(n=min(5, len(merged)), random_state=42)
for _, row in sample.iterrows():
    smi = row['compound_iso_smiles_norm']
    print(f"   {smi[:75]}")

# 8. 保存（保留source列以便后续追溯）
print(f"\n💾 保存至: {OUTPUT_PATH}")
merged.to_parquet(OUTPUT_PATH, index=False)
print("=" * 65)