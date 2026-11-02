import re
import time
import logging
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import polars as pl
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import MolStandardize

# ==================== 全局配置区 ====================
# 输入文件路径
# BINDINGDB_PATH = "bindingdb_highconf_pKd_clean.csv"
# MAPPING_PATH = "序列-蛋白ID映射表.csv"


BINDINGDB_PATH = r"E:\research_project\DTI预测\bindingdb_highconf_pKd_clean.csv"
MAPPING_PATH = r"E:\research_project\DTI预测\序列-蛋白ID映射表.csv"


# 输出文件路径
INTERMEDIATE_CSV = "bindingdb_compound_uniprot_affinity.csv"      # 步骤1输出 / 步骤2输入
FINAL_PARQUET = "bindingdb_compound_uniprot_affinity_cleaned.parquet"  # 最终标准化输出

# 步骤1: 大分子/非类药分子过滤配置
MAX_HEAVY_ATOMS = 300  # 重原子上限
CHUNK_SIZE = 100_000   # Pandas分块读取大小

# 步骤2: SMILES标准化配置
N_WORKERS = max(1, mp.cpu_count() - 1)
BATCH_SIZE = 50_000
TARGET_COL = "compound_iso_smiles"
KNOWN_COLUMNS = ["compound_iso_smiles", "uniprot_id", "affinity"]

# 日志与RDKit警告配置
RDLogger.DisableLog('rdApp.*')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 预编译正则：匹配连续肽键或连续糖苷键重复模式
PEPTIDE_PATTERN = re.compile(r'C\(=O\)N\[C@H\].*?C\(=O\)N', re.IGNORECASE)
GLYCAN_PATTERN = re.compile(r'\[C@@?H\]\d*O\[C@@?H\].*?\[C@@?H\]\d*O', re.IGNORECASE)


# ==================== 步骤1: 序列映射与大分子过滤 ====================
def estimate_heavy_atom_count(smi: str) -> int:
    """基于SMILES字符串估算重原子数（不含H），纯字符串操作"""
    count = 0
    i = 0
    n = len(smi)
    while i < n:
        c = smi[i]
        if c == '[':
            count += 1
            end = smi.find(']', i)
            i = end + 1 if end != -1 else n
        elif c.isupper():
            count += 1
            i += 1
            while i < n and smi[i].islower():
                i += 1
        else:
            i += 1
    return count


def is_valid_drug_smiles(smi) -> bool:
    """轻量级SMILES预过滤，排除多肽、多糖及超大分子"""
    if not isinstance(smi, str):
        return False
    if len(smi) > 1500:
        return False
    if estimate_heavy_atom_count(smi) > MAX_HEAVY_ATOMS:
        return False
    if PEPTIDE_PATTERN.search(smi) or GLYCAN_PATTERN.search(smi):
        return False
    return True


def step1_mapping_and_filtering():
    """执行序列映射、大分子过滤，输出中间CSV"""
    logger.info("📖 [步骤1] 加载序列-蛋白ID映射表...")
    df_mapping = pd.read_csv(MAPPING_PATH, usecols=["sequence", "uniprot_id"])
    seq_to_uniprot = dict(zip(df_mapping["sequence"], df_mapping["uniprot_id"]))
    logger.info(f"   映射表大小: {len(seq_to_uniprot):,} 条唯一序列")

    logger.info("🔄 [步骤1] 开始分块处理BindingDB...")
    matched_chunks = []
    total_matched = 0
    total_rows = 0
    total_filtered = 0

    for chunk in tqdm(pd.read_csv(BINDINGDB_PATH, chunksize=CHUNK_SIZE), desc="   映射与过滤进度"):
        total_rows += len(chunk)
        uniprot_ids = chunk["target_sequence"].map(seq_to_uniprot)
        valid_mask = uniprot_ids.notna()

        if valid_mask.any():
            matched_smiles = chunk.loc[valid_mask, "compound_iso_smiles"]
            drug_mask_local = [is_valid_drug_smiles(s) for s in matched_smiles]
            valid_mask.loc[valid_mask] = drug_mask_local
            total_filtered += sum(not x for x in drug_mask_local)

        matched = chunk.loc[valid_mask, ["compound_iso_smiles", "affinity"]].copy()
        matched["uniprot_id"] = uniprot_ids[valid_mask]

        if len(matched) > 0:
            matched_chunks.append(matched)
            total_matched += len(matched)

    # 合并并保存中间结果
    if matched_chunks:
        df_result = pd.concat(matched_chunks, ignore_index=True)
        df_result = df_result[["compound_iso_smiles", "uniprot_id", "affinity"]]
        df_result.to_csv(INTERMEDIATE_CSV, index=False, encoding="utf-8-sig")
    else:
        logger.warning("⚠️ 未找到任何匹配记录！请检查序列格式或过滤条件")
        df_result = pd.DataFrame(columns=KNOWN_COLUMNS)
        df_result.to_csv(INTERMEDIATE_CSV, index=False)

    match_rate = total_matched / total_rows * 100 if total_rows > 0 else 0
    filter_rate = total_filtered / total_rows * 100 if total_rows > 0 else 0
    logger.info(f"✅ [步骤1] 完成! 总行数: {total_rows:,} | 匹配: {total_matched:,} ({match_rate:.2f}%) | 过滤: {total_filtered:,} ({filter_rate:.2f}%)")
    logger.info(f"💾 [步骤1] 中间文件已保存: {INTERMEDIATE_CSV}")
    return total_matched


# ==================== 步骤2: SMILES多进程标准化 ====================
def normalize_or_fallback(smi: str) -> str:
    """RDKit标准化：去盐、中和电荷、规范化SMILES；失败则返回原始字符串"""
    if not isinstance(smi, str) or len(smi.strip()) == 0:
        return ""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi.strip()
        lfc = MolStandardize.rdMolStandardize.LargestFragmentChooser()
        mol = lfc.choose(mol)
        uncharger = MolStandardize.rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return smi.strip()


def process_batch(smiles_list: list[str]) -> list[str]:
    """纯函数：只接收SMILES列表，返回标准化结果列表（用于多进程）"""
    return [normalize_or_fallback(s) for s in smiles_list]


def step2_normalization():
    """读取中间CSV，多进程标准化SMILES，输出最终Parquet"""
    start_time = time.time()

    logger.info(f"📥 [步骤2] 正在读取 {INTERMEDIATE_CSV} ...")
    df = pl.read_csv(INTERMEDIATE_CSV, new_columns=KNOWN_COLUMNS, has_header=True)
    read_elapsed = time.time() - start_time
    logger.info(f"✅ [步骤2] 读取完成: {df.height:,} 行 | 耗时: {read_elapsed:.1f}s")

    all_smiles = df[TARGET_COL].to_list()
    batches = [all_smiles[i:i + BATCH_SIZE] for i in range(0, len(all_smiles), BATCH_SIZE)]
    total_batches = len(batches)
    logger.info(f"🚀 [步骤2] 开始标准化 ({N_WORKERS} 进程 | {total_batches} 批次)...")

    norm_results = []
    with mp.Pool(processes=N_WORKERS) as pool:
        for i, result_list in enumerate(pool.imap(process_batch, batches, chunksize=4), 1):
            norm_results.extend(result_list)
            if i % 10 == 0 or i == total_batches:
                logger.info(f"   ✅ 已完成 {i}/{total_batches} 批次")

    logger.info("⏳ [步骤2] 合并与保存...")
    final_df = df.with_columns(pl.Series(f"{TARGET_COL}_norm", norm_results, dtype=pl.String))

    stats = final_df.select([
        pl.col(f"{TARGET_COL}_norm").is_not_null().sum().alias("valid"),
        (pl.col(f"{TARGET_COL}_norm") == pl.col(TARGET_COL).str.strip_chars()).sum().alias("fallback_count"),
    ])

    final_df.write_parquet(FINAL_PARQUET)

    elapsed = time.time() - start_time
    speed = df.height / elapsed if elapsed > 0 else 0
    logger.info(f"✅ [步骤2] 完成! 总行数: {df.height:,} | 耗时: {elapsed:.1f}s | 速度: {speed:,.0f} 条/秒")
    logger.info(f"📊 标准化成功: {stats['valid'][0]:,} | Fallback: {stats['fallback_count'][0]:,}")
    logger.info(f"💾 [步骤2] 最终文件已保存: {FINAL_PARQUET}")


# ==================== 主入口 ====================
if __name__ == "__main__":
    pipeline_start = time.time()
    logger.info("=" * 60)
    logger.info("🚀 BindingDB 端到端处理流水线启动")
    logger.info("=" * 60)

    # 步骤1: 序列映射 + 大分子过滤
    matched_count = step1_mapping_and_filtering()

    if matched_count > 0:
        # 步骤2: SMILES标准化 + Parquet输出
        step2_normalization()
    else:
        logger.warning("⚠️ 步骤1无有效数据，跳过步骤2标准化")

    total_elapsed = time.time() - pipeline_start
    logger.info("=" * 60)
    logger.info(f"🎉 全部流水线完成! 总耗时: {total_elapsed:.1f}s")
    logger.info("=" * 60)