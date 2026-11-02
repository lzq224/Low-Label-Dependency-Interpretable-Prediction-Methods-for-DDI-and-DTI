import polars as pl

# ==================== 配置区 ====================
PARQUET_PATH = r"E:\research_project\DTI预测\bindingdb_compound_uniprot_affinity_cleaned.parquet"
GENE_MAP_PATH = r"E:\research_project\DTI预测\bioDBnet_db2db_bindingdb.txt"
OUTPUT_PATH = r"E:\research_project\DTI预测\bindingdb_compound_uniprot_affinity_with_gene.parquet"
# ================================================

# 1. 读取基因映射表（制表符分隔，跳过可能的注释行）
print("⏳ 读取基因映射表...")
df_gene = (
    pl.read_csv(GENE_MAP_PATH, separator="\t", comment_prefix="#")
    .rename({"UniProt Accession": "uniprot_id_upper", "Gene Symbol": "gene_symbol"})
    # 统一转大写作为匹配键，保留原始 gene_symbol 不变
    .with_columns(pl.col("uniprot_id_upper").str.to_uppercase())
    .unique(subset=["uniprot_id_upper"], maintain_order=True)  # 防止一对多导致数据膨胀
)
print(f"  基因映射条目数: {len(df_gene):,}")

# 2. 读取亲和力数据并创建大写匹配键
print("⏳ 读取亲和力数据...")
df_affinity = pl.read_parquet(PARQUET_PATH)
print(f"  原始行数: {len(df_affinity):,}")

# 3. Inner Join 匹配基因符号（仅保留匹配成功的数据，自动删除未匹配项）
print("⏳ 执行 Inner Join 匹配（未匹配数据将被删除）...")
df_result = (
    df_affinity
    .with_columns(pl.col("uniprot_id").str.to_uppercase().alias("_join_key"))
    .join(df_gene, left_on="_join_key", right_on="uniprot_id_upper", how="inner")
    .drop("_join_key")  # 移除临时匹配键
)

# 4. 统计匹配情况
matched = len(df_result)
total = len(df_affinity)
print(f"  ✅ 成功匹配并保留: {matched:,} / {total:,} ({matched/total*100:.1f}%)")
if matched < total:
    print(f"  🗑️ 已删除未匹配数据: {total - matched:,} 条")

# 5. 写入新 Parquet
df_result.write_parquet(OUTPUT_PATH)
print(f"✅ 已保存至: {OUTPUT_PATH}")