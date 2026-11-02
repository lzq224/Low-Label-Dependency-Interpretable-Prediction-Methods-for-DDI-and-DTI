import polars as pl

# ==================== 配置区 ====================
PARQUET_PATH = r"E:\research_project\DTI预测\bindingdb_compound_uniprot_affinity_cleaned.parquet"
OUTPUT_CSV_PATH = r"E:\research_project\DTI预测\bindingdb_uniprot_ids_unique.csv"
# ================================================

# 1. 仅读取 uniprot_id 列（列投影，避免加载整表）
print("⏳ 读取 uniprot_id 列...")
df = pl.read_parquet(PARQUET_PATH, columns=["uniprot_id"])
print(f"   原始行数: {len(df):,}")

# 2. 统一转大写后去重（不区分大小写）
df_unique = (
    df.with_columns(pl.col("uniprot_id").str.to_uppercase())
      .unique(subset=["uniprot_id"], maintain_order=True)
)
print(f"   去重后行数: {len(df_unique):,}")

# 3. 写入 CSV（不含索引列）
df_unique.write_csv(OUTPUT_CSV_PATH)
print(f"✅ 已保存至: {OUTPUT_CSV_PATH}")