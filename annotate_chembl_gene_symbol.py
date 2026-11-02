import pandas as pd

# ===================== 配置区 =====================
CHEMBL_PATH = r"D:\E盘模型暂存\chembl_bioactivity_standardized.parquet"
GENE_MAP_PATH = r"E:\research_project\DTI预测\bioDBnet_db2db_chembl.txt"
OUTPUT_PATH = r"D:\E盘模型暂存\chembl_bioactivity_standardized_with_genesymbol.parquet"
# ==================================================

print("=" * 65)
print("🧬 Gene Symbol 回填流程（未匹配数据将被删除）")
print("=" * 65)

# 1. 读取 Gene Symbol 映射表
print("\n📥 加载 Gene Symbol 映射表...")
gene_df = pd.read_csv(GENE_MAP_PATH, sep='\t', dtype=str)
gene_df.columns = gene_df.columns.str.strip()  # 防止列名含隐藏空格
gene_map = dict(zip(gene_df['UniProt Accession'], gene_df['Gene Symbol']))
print(f"  映射表大小: {len(gene_map):,} 条")

# 2. 读取 ChEMBL 数据并回填
print("\n🔄 回填 gene_symbol...")
df = pd.read_parquet(CHEMBL_PATH)
total_before = len(df)

# 使用 map 进行匹配，未匹配的值为 NaN
df['gene_symbol'] = df['uniprot_id'].map(gene_map)

# 3. 删除未匹配到 gene_symbol 的数据（即 NaN 行）
df = df.dropna(subset=['gene_symbol'])
total_after = len(df)
deleted_count = total_before - total_after

print(f"  原始记录数: {total_before:,}")
print(f"  ✅ 成功匹配并保留: {total_after:,} ({total_after/total_before*100:.2f}%)")
if deleted_count > 0:
    print(f"  🗑️ 已删除未匹配数据: {deleted_count:,} ({deleted_count/total_before*100:.2f}%)")
print(f"  唯一 Gene Symbol: {df['gene_symbol'].nunique():,}")

# 4. 保存
print(f"\n💾 保存至: {OUTPUT_PATH}")
df.to_parquet(OUTPUT_PATH, index=False)
print(f"\n✅ 处理完成! 最终记录: {len(df):,}")
print("=" * 65)