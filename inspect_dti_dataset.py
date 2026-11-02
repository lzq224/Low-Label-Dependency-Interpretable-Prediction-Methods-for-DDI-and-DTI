import polars as pl

df = pl.scan_parquet(r"D:\E盘模型暂存\dti_merged_with_descriptors.parquet")

# 1. 查看列名和类型
print(df.collect_schema())

# 2. 查看前10行
print(df.head(10).collect())

# 3. 基本统计
print(f"总行数: {df.select(pl.len()).collect().item():,}")
print(f"唯一化合物数: {df.select(pl.col('compound_iso_smiles_norm').n_unique()).collect().item():,}")
print(f"唯一靶点数: {df.select(pl.col('gene_symbol').n_unique()).collect().item():,}")

# ✅ 修复：用实际存在的 source 列替代 assay_type
print(f"数据来源(source)分布:\n{df.select('source').collect().to_series().value_counts()}")

# 🆕 补充诊断：同一化合物-靶点对的记录数分布（评估affinity混合程度）
diag = (
    df.group_by(["compound_iso_smiles_norm", "gene_symbol"])
    .agg(pl.len().alias("n_records"), pl.col("affinity").std().alias("aff_std"))
    .filter(pl.col("n_records") > 1)
    .sort("n_records", descending=True)
    .collect()
)
print(f"\n有多条记录的化合物-靶点对: {diag.height:,}")
print(diag.head(10))