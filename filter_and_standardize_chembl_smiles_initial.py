import pandas as pd
import numpy as np
import re
import pyarrow.parquet as pq
from tqdm import tqdm

# ===================== 配置区 =====================
CHEMBL_PATH = r"D:\E盘模型暂存\chembl_bioactivity_with_smiles.parquet"
TARGET_REF_PATH = r"E:\research_project\DTI预测\bindingdb_compound_uniprot_affinity_with_gene.parquet"
OUTPUT_PATH = r"D:\E盘模型暂存\chembl_bioactivity_standardized.parquet"

MAX_HEAVY_ATOMS = 300
MAX_SMILES_STR_LEN = 1500
# ==================================================

PEPTIDE_PATTERN = re.compile(r'C\(=O\)N\[C@H\].*?C\(=O\)N', re.IGNORECASE)
GLYCAN_PATTERN = re.compile(r'\[C@@?H\]\d*O\[C@@?H\].*?\[C@@?H\]\d*O', re.IGNORECASE)


def estimate_heavy_atom_count(smi: str) -> int:
    """基于SMILES字符串估算重原子数（不含H）"""
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
    """轻量级SMILES预过滤"""
    if not isinstance(smi, str):
        return False
    if len(smi) > MAX_SMILES_STR_LEN:
        return False
    if estimate_heavy_atom_count(smi) > MAX_HEAVY_ATOMS:
        return False
    if PEPTIDE_PATTERN.search(smi) or GLYCAN_PATTERN.search(smi):
        return False
    return True


print("=" * 65)
print("🔄 ChEMBL → 标准DTI格式 转换流程")
print("=" * 65)

# 1. 读取参考文件获取列名规范
ref_cols = pq.read_schema(TARGET_REF_PATH).names
print(f"📋 目标列名: {ref_cols}")

# 2. 读取ChEMBL数据
print("\n📥 加载ChEMBL数据...")
df = pd.read_parquet(CHEMBL_PATH)
print(f"   原始记录: {len(df):,}")

# 3. ⚡ 过滤（带进度条）
print(f"\n🔬 执行过滤 (重原子≤{MAX_HEAVY_ATOMS} + 多肽/多糖排除)...")
before = len(df)
# ✅ 使用tqdm包裹列表推导式，实时显示过滤进度
drug_mask = [is_valid_drug_smiles(s) for s in tqdm(df['smiles'], desc="   🔍 SMILES过滤", unit="条", ncols=90)]
df = df[drug_mask].reset_index(drop=True)
filtered_count = before - len(df)
print(f"   ✂️ 过滤结果: {before:,} → {len(df):,} (移除{filtered_count:,}, {filtered_count / before * 100:.2f}%)")

# 4. 类型修正
df['pchembl_value'] = pd.to_numeric(df['pchembl_value'], errors='coerce')
df['standard_type'] = df['standard_type'].astype('category')

# 5. 靶点ID转换: ChEMBL Target ID → UniProt ID（带进度条）
print("\n🔗 转换靶点ID (ChEMBL → UniProt)...")


def get_uniprot_from_chembl_targets(target_ids):
    """批量查询ChEMBL Target对应的UniProt ID"""
    import requests, time
    unique_targets = list(set(target_ids))
    mapping = {}
    batch_size = 100
    total_batches = (len(unique_targets) + batch_size - 1) // batch_size

    # ✅ 使用tqdm包裹API请求批次循环
    pbar = tqdm(range(0, len(unique_targets), batch_size),
                total=total_batches, desc="   🌐 API映射", unit="批", ncols=90)

    for idx, i in enumerate(pbar):
        batch = unique_targets[i:i + batch_size]
        url = f"https://www.ebi.ac.uk/chembl/api/data/target.json?target_chembl_id__in={','.join(batch)}&limit={batch_size}"
        try:
            resp = requests.get(url, timeout=30)
            data = resp.json()
            for target in data.get('targets', []):
                tid = target.get('target_chembl_id')
                components = target.get('target_components', [])
                uniprot = None
                for comp in components:
                    for xref in comp.get('target_component_xrefs', []):
                        if xref.get('xref_src_db') == 'UniProt':
                            uniprot = xref.get('xref_id')
                            break
                    if uniprot:
                        break
                if tid and uniprot:
                    mapping[tid] = uniprot
        except Exception as e:
            tqdm.write(f"   ⚠️ 批次{idx + 1}/{total_batches}请求失败: {e}")

        # ✅ 动态更新进度条后缀信息
        pbar.set_postfix(
            {"已映射": f"{len(mapping):,}", "成功率": f"{len(mapping) / (idx + 1) / batch_size * 100:.0f}%"})
        time.sleep(0.3)

    pbar.close()
    return mapping


target_map = get_uniprot_from_chembl_targets(df['target_chembl_id'].tolist())
print(f"   成功映射: {len(target_map):,} / {df['target_chembl_id'].nunique():,} 个靶点")

df['uniprot_id'] = df['target_chembl_id'].map(target_map)
unmapped = df['uniprot_id'].isna().sum()
print(f"   ⚠️ 未映射记录: {unmapped:,} ({unmapped / len(df) * 100:.1f}%)")

df = df.dropna(subset=['uniprot_id']).reset_index(drop=True)
print(f"   保留记录: {len(df):,}")

# 6. 生成标准化SMILES（带进度条）
try:
    from rdkit import Chem
    from rdkit.Chem import MolStandardize

    print("\n🧪 生成标准化SMILES (compound_iso_smiles_norm)...")
    normalizer = MolStandardize.rdMolStandardize.Uncharger()


    def normalize_smiles(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return smi
            mol = normalizer.uncharge(mol)
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        except Exception:
            return smi


    # ✅ RDKit标准化也加上进度条
    df['compound_iso_smiles_norm'] = [
        normalize_smiles(s) for s in tqdm(df['smiles'], desc="   🧪 RDKit标准化", unit="条", ncols=90)
    ]
    norm_success = (df['compound_iso_smiles_norm'] != df['smiles']).sum()
    print(f"   标准化变更: {norm_success:,} 条")
except ImportError:
    print("\n⚠️ RDKit未安装，compound_iso_smiles_norm 将复制原始SMILES")
    df['compound_iso_smiles_norm'] = df['smiles']

# 7. 列名映射 & 列顺序对齐
COLUMN_MAPPING = {
    'smiles': 'compound_iso_smiles',
    'uniprot_id': 'uniprot_id',
    'pchembl_value': 'affinity',
    'compound_iso_smiles_norm': 'compound_iso_smiles_norm',
}
df = df.rename(columns=COLUMN_MAPPING)

if 'gene_symbol' in ref_cols:
    df['gene_symbol'] = ''

df = df[[c for c in ref_cols if c in df.columns]]

# 8. 保存
print("\n💾 保存Parquet文件...")
df.to_parquet(OUTPUT_PATH, index=False)

print(f"\n{'=' * 65}")
print(f"✅ 转换完成!")
print(f"   输出: {OUTPUT_PATH}")
print(f"   最终记录: {len(df):,}")
print(f"   化合物: {df['compound_iso_smiles'].nunique():,}")
print(f"   靶点(UniProt): {df['uniprot_id'].nunique():,}")
print(f"\n📊 数据类型:")
print(df.dtypes)
print(f"\n📈 affinity统计:")
print(df['affinity'].describe())
print("=" * 65)