from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import os
import sys
import time
import shutil
import multiprocessing as mp
import pandas as pd
import polars as pl
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

# ==================== 配置区 ====================
INPUT_PARQUET = r"D:\E盘模型暂存\dti_merged_dataset.parquet"
OUTPUT_DIR = r"E:\research_project\DTI预测\descriptors_output"
FINAL_OUTPUT_PATH = r"D:\E盘模型暂存\dti_merged_with_descriptors.parquet"

CHECKPOINT_INTERVAL = 50000
NUM_WORKERS = None  # None时自动使用 cpu_count()-1
BATCH_SIZE = 500

# 互变异构体列举参数限制
TAUTOMER_MAX_TRANSFORMS = 100
TAUTOMER_MAX_TAUTOMERS = 200
SMILES_LENGTH_SOFT_LIMIT = 300

# O(1)拦截阈值
HEAVY_ATOM_LIMIT = 150
TAUTOMER_SITE_LIMIT = 12
# ================================================


def _count_tautomer_sites(mol) -> int:
    """O(1) 互变复杂度估算：统计可参与互变的极性杂原子数(N/O/P/S)"""
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in (7, 8, 15, 16):
            count += 1
    return count


def _create_safe_taut_enumerator():
    """创建带安全限制的互变异构体枚举器"""
    te = rdMolStandardize.TautomerEnumerator()
    te.SetMaxTransforms(TAUTOMER_MAX_TRANSFORMS)
    te.SetMaxTautomers(TAUTOMER_MAX_TAUTOMERS)
    return te


def calc_all_descriptors(smiles: str) -> dict:
    """一次性计算所有2D描述符，含O(1)拦截 + 互变异构体归一化"""  # 对互变异构体做一致性归一处理，使同一分子不因互变异构形式不同而产生描述符偏移
    result = {
        'MW': None, 'LogP': None, 'TPSA': None,
        'HBD': None, 'HBA': None, 'LabuteASA': None,
    }
    for i in range(1, 15):
        result[f'PEOE_VSA{i}'] = None
    for i in range(1, 11):
        result[f'SMR_VSA{i}'] = None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result['_calc_success'] = False
            return result

        # Step 1: LFC
        lfc = rdMolStandardize.LargestFragmentChooser()
        mol = lfc.choose(mol)

        # Step 2: Uncharger
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        # Step 2.5: O(1) 拦截检查
        ha = mol.GetNumHeavyAtoms()
        ts = _count_tautomer_sites(mol)
        if ha > HEAVY_ATOM_LIMIT or ts > TAUTOMER_SITE_LIMIT:
            normalized_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        else:
            normalized_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            smiles_len = len(normalized_smi)
            if smiles_len <= SMILES_LENGTH_SOFT_LIMIT:
                try:
                    te = _create_safe_taut_enumerator()
                    mol_canon = te.Canonicalize(mol)
                    normalized_smi = Chem.MolToSmiles(mol_canon, canonical=True, isomericSmiles=True)
                except Exception:
                    pass

        mol_final = Chem.MolFromSmiles(normalized_smi)
        if mol_final is None:
            mol_final = mol

        result['MW'] = float(Descriptors.MolWt(mol_final))
        result['LogP'] = float(Crippen.MolLogP(mol_final))
        result['TPSA'] = float(rdMolDescriptors.CalcTPSA(mol_final))
        result['HBD'] = int(Lipinski.NumHDonors(mol_final))
        result['HBA'] = int(Lipinski.NumHAcceptors(mol_final))
        result['LabuteASA'] = float(rdMolDescriptors.CalcLabuteASA(mol_final))

        for i in range(1, 15):
            val = getattr(Descriptors, f'PEOE_VSA{i}')(mol_final)
            result[f'PEOE_VSA{i}'] = float(val) if val is not None else None

        for i in range(1, 11):
            val = getattr(Descriptors, f'SMR_VSA{i}')(mol_final)
            result[f'SMR_VSA{i}'] = float(val) if val is not None else None

        result['_calc_success'] = True

    except Exception as e:
        print(f"❌ 描述符计算异常 | SMILES: {smiles[:50]}... | Error: {e}")
        result['_calc_success'] = False

    return result


def process_batch(batch_with_idx: list[tuple[int, str]]) -> list[dict]:
    """处理一批带索引的SMILES"""
    results = []
    for idx, smi in batch_with_idx:
        desc = calc_all_descriptors(smi.strip())
        desc['idx'] = idx
        results.append(desc)
    return results


def load_checkpoint(output_dir: str) -> set[int]:
    completed = set()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        return completed
    for fname in os.listdir(output_dir):
        if fname.startswith("chunk_") and fname.endswith(".parquet"):
            try:
                df = pl.read_parquet(os.path.join(output_dir, fname), columns=["idx"])
                completed.update(df["idx"].to_list())
            except Exception:
                continue
    return completed


def main():
    num_workers = NUM_WORKERS or (mp.cpu_count() - 1)
    print(f"📂 输入文件: {INPUT_PARQUET}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"🎯 最终输出: {FINAL_OUTPUT_PATH}")
    print(f"🔧 并行核心数: {num_workers}")
    print(f"🧬 标准化管线: LFC → Uncharger → TautomerEnumerator(max={TAUTOMER_MAX_TAUTOMERS})")
    print(f"⚡ 安全限制: max_transforms={TAUTOMER_MAX_TRANSFORMS}, "
          f"max_tautomers={TAUTOMER_MAX_TAUTOMERS}, "
          f"smiles_skip>{SMILES_LENGTH_SOFT_LIMIT}")
    print(f"🆕 O(1)拦截: heavy_atom<{HEAVY_ATOM_LIMIT}, tautomer_site<{TAUTOMER_SITE_LIMIT}")
    print(f"📦 Batch大小: {BATCH_SIZE}")

    # ✅ 全量重算时清空旧chunk
    if os.path.exists(OUTPUT_DIR):
        old_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("chunk_")]
        if old_files:
            print(f"🗑️  检测到 {len(old_files)} 个旧chunk文件，正在清空...")
            shutil.rmtree(OUTPUT_DIR)
            print("   ✅ 旧数据已清除")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 📥 从 dti_merged_dataset.parquet 读取唯一SMILES
    print("\n⏳ 正在从 dti_merged_dataset.parquet 提取唯一SMILES...")
    dti_df = pl.read_parquet(INPUT_PARQUET, columns=['compound_iso_smiles_norm'])
    all_smiles = sorted(dti_df['compound_iso_smiles_norm'].unique().to_list())
    total = len(all_smiles)
    print(f"✅ 共 {total:,} 条唯一SMILES")

    completed_idx = load_checkpoint(OUTPUT_DIR)
    remaining = [(i, s) for i, s in enumerate(all_smiles) if i not in completed_idx]
    print(f"🔄 已完成 {len(completed_idx):,} 条，剩余 {len(remaining):,} 条")

    if not remaining:
        print("🎉 全部计算已完成！请运行 --merge 步骤。")
        return

    batches = [remaining[i:i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    expected_chunks = (len(remaining) + CHECKPOINT_INTERVAL - 1) // CHECKPOINT_INTERVAL
    print(f"📦 预计写入 {expected_chunks} 个chunk (每个{CHECKPOINT_INTERVAL:,}条)")

    processed_in_current_chunk = 0
    total_processed_since_resume = 0
    chunk_counter = 0
    start_time = time.time()
    last_save_time = start_time

    with mp.Pool(processes=num_workers) as pool:
        buffer = []
        for batch_result in pool.imap_unordered(process_batch, batches):
            buffer.extend(batch_result)
            batch_size = len(batch_result)
            processed_in_current_chunk += batch_size
            total_processed_since_resume += batch_size

            if processed_in_current_chunk >= CHECKPOINT_INTERVAL:
                chunk_counter += 1
                save_path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_counter:05d}.parquet")
                pl.DataFrame(buffer).write_parquet(save_path)

                now = time.time()
                chunk_elapsed = now - last_save_time
                speed = processed_in_current_chunk / chunk_elapsed if chunk_elapsed > 0 else 0
                done_total = len(completed_idx) + total_processed_since_resume
                eta = (total - done_total) / speed if speed > 0 else float('inf')

                print(f"  💾 chunk_{chunk_counter:05d} | "
                      f"{done_total:,}/{total:,} ({done_total / total * 100:.1f}%) | "
                      f"{speed:.0f} mol/s | ETA: {eta / 60:.1f} min")

                buffer = []
                processed_in_current_chunk = 0
                last_save_time = now

    # 保存尾部数据
    if buffer:
        chunk_counter += 1
        save_path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_counter:05d}.parquet")
        pl.DataFrame(buffer).write_parquet(save_path)
        now = time.time()
        chunk_elapsed = now - last_save_time
        speed = len(buffer) / chunk_elapsed if chunk_elapsed > 0 else 0
        done_total = len(completed_idx) + total_processed_since_resume
        print(f"  💾 尾部 chunk_{chunk_counter:05d} | "
              f"{done_total:,}/{total:,} ({done_total / total * 100:.1f}%) | "
              f"{speed:.0f} mol/s")

    elapsed = time.time() - start_time
    print(f"\n✅ 所有计算完成! 总耗时: {elapsed / 3600:.1f} 小时")
    print(f"📁 Chunk结果保存在: {OUTPUT_DIR}")
    print(f"💡 下一步: python {__file__} --merge")


def merge_results():
    """合并chunk → 与原始DTI数据join → 删除失败行 → 保存最终文件"""
    # 1. 合并所有chunk
    files = sorted([
        os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR)
        if f.startswith("chunk_") and f.endswith(".parquet")
    ])
    if not files:
        print("❌ 未找到任何chunk文件，请先运行计算步骤")
        return

    print(f"🔗 正在合并 {len(files)} 个chunk文件...")
    dfs = [pl.read_parquet(f) for f in files]
    desc_df = pl.concat(dfs).sort("idx")
    print(f"   描述符总行数: {len(desc_df):,}")

    # 2. 构建 idx → SMILES 映射（与计算时排序一致）
    dti_pl = pl.read_parquet(INPUT_PARQUET, columns=['compound_iso_smiles_norm'])
    all_smiles = sorted(dti_pl['compound_iso_smiles_norm'].unique().to_list())
    idx_to_smi = {i: s for i, s in enumerate(all_smiles)}

    # 3. 将idx映射回SMILES，用于后续join
    desc_pd = desc_df.to_pandas()
    desc_pd['compound_iso_smiles_norm'] = desc_pd['idx'].map(idx_to_smi)
    desc_pd.drop(columns=['idx'], inplace=True)

    # 4. 过滤计算失败的行
    before_filter = len(desc_pd)
    if '_calc_success' in desc_pd.columns:
        desc_pd = desc_pd[desc_pd['_calc_success'] == True].copy()
        desc_pd.drop(columns=['_calc_success'], inplace=True)
    else:
        desc_pd = desc_pd[desc_pd['MW'].notna()].copy()
    after_filter = len(desc_pd)
    failed_count = before_filter - after_filter
    print(f"🗑️  描述符计算失败: {failed_count:,} 条SMILES已移除")

    # 5. 与原始DTI数据集合并（inner join自动丢弃无描述符的行）
    print(f"\n🔄 正在与原始DTI数据集合并...")
    dti_pd = pd.read_parquet(INPUT_PARQUET)
    before_merge = len(dti_pd)

    merged = dti_pd.merge(desc_pd, on='compound_iso_smiles_norm', how='inner')
    after_merge = len(merged)
    dropped_rows = before_merge - after_merge

    print(f"   合并前DTI记录数: {before_merge:,}")
    print(f"   合并后DTI记录数: {after_merge:,}")
    print(f"   🗑️  因描述符缺失删除: {dropped_rows:,} 条数据行")

    # 6. 保存最终文件
    print(f"\n💾 保存至: {FINAL_OUTPUT_PATH}")
    merged.to_parquet(FINAL_OUTPUT_PATH, index=False)

    # 7. 最终统计
    desc_cols = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'LabuteASA']
    desc_cols = [c for c in desc_cols if c in merged.columns]
    null_summary = merged[desc_cols].isnull().sum()
    has_null = null_summary[null_summary > 0]
    if len(has_null) > 0:
        print(f"\n⚠️ 警告: 以下描述符列仍有空值:")
        for col, cnt in has_null.items():
            print(f"   {col}: {cnt:,}")
    else:
        print(f"\n✅ 所有描述符列完整性: 100%")

    print(f"\n{'=' * 65}")
    print(f"✅ 最终数据集已生成!")
    print(f"   文件路径: {FINAL_OUTPUT_PATH}")
    print(f"   总记录数: {len(merged):,}")
    print(f"   唯一化合物: {merged['compound_iso_smiles_norm'].nunique():,}")
    print(f"   描述符列数: {len(desc_pd.columns) - 1}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge_results()
    else:
        main()