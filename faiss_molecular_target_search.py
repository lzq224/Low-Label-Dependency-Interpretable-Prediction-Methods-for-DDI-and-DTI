import os, gc, csv, ctypes, logging, time, pickle
import numpy as np
import polars as pl
import pandas as pd
from sklearn.preprocessing import StandardScaler
import faiss
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')
os.environ["POLARS_MAX_THREADS"] = "6"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# ==================== 配置区 ====================
DESC_PARQUET_PATH = r"E:\research_project\DTI预测\descriptors_final.parquet"
AFFINITY_PARQUET_PATH = r"D:\E盘模型暂存\dti_merged_dataset.parquet"
OUTPUT_DIR = r"E:\research_project\DTI预测\query_results"

# ★ FP缓存路径（与描述符文件同目录，自动关联）
FP_CACHE_PATH = r"E:\research_project\DTI预测\fp_cache.pkl"

QUERY_SOURCE = "list"
QUERY_FILE_PATH = ""
QUERY_SMILES_COL = "smiles"
QUERY_ID_COL = ""

QUERY_SMILES_LIST = [
    "C1CN(CCN1CCCN2C3=CC=CC=C3SC4=C2C=C(C=C4)C(F)(F)F)CCO", # 氟奋乃静
    "CC/C(=C(/CC)\C1=CC=C(C=C1)O)/C2=CC=C(C=C2)O", # 己烯雌酚
    "CC(=O)OC1=CC=CC=C1C(=O)O", # 阿司匹林
]

TOP_N = 20
SIM_THRESHOLD = 0.70
FP_TANIMOTO_THRESHOLD = 0.40
FP_RADIUS = 2
FP_NBITS = 2048
BASE_POOL_SIZE = 100
MAX_POOL_SIZE = 500
DENSE_REGION_THRESHOLD = 5000
POOL_DIVERSITY_WEIGHT = 0.3
FP_FALLBACK_THRESHOLDS = [0.30, 0.25, 0.20]
MIN_FP_RESULTS = 3
MIN_MW, MAX_MW = 50, 1000
MIN_NONZERO_VSA = 3
TPSA_TOLERANCE = 15.0
LOGP_TOLERANCE = 1.5
FAISS_SEARCH_K = 3000
AFFINITY_THRESHOLD_STEPS = [6.0, 5.0, 4.0, 3.0]

DESC_COLS = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'LabuteASA'] \
    + [f'PEOE_VSA{i}' for i in range(1, 15)] \
    + [f'SMR_VSA{i}' for i in range(1, 11)]
SMILES_COL_DESC = "SMILES"
SMILES_COL_AFF = "compound_iso_smiles_norm"

FILE_EXACT_MATCHES = "exact_matches.csv"
FILE_TOP_SIMILAR = "top_similar_compounds.csv"
FILE_TGT_DETAIL = "recommended_targets_detail.csv"
FILE_TGT_SUMMARY = "recommended_targets_summary.csv"
FILE_DTI_EDGES = "drug_target_edges.csv"


# ==================== 工具函数 ====================
def normalize_smiles(smi):
    if not isinstance(smi, str) or not smi.strip():
        return ""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi.strip()
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return smi.strip()


def calc_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES无法解析: {smiles}")
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    vals = [float(Descriptors.MolWt(mol)), float(Crippen.MolLogP(mol)),
            float(rdMolDescriptors.CalcTPSA(mol)), int(Lipinski.NumHDonors(mol)),
            int(Lipinski.NumHAcceptors(mol)), float(rdMolDescriptors.CalcLabuteASA(mol))]
    for i in range(1, 15):
        v = getattr(Descriptors, f'PEOE_VSA{i}')(mol)
        vals.append(float(v) if v is not None else 0.0)
    for i in range(1, 11):
        v = getattr(Descriptors, f'SMR_VSA{i}')(mol)
        vals.append(float(v) if v is not None else 0.0)
    return vals


def diverse_topk(sims, X_scaled, k, div_weight):
    if len(sims) <= k:
        return np.arange(len(sims))
    selected = [int(np.argmax(sims))]
    remaining = set(range(len(sims))) - {selected[0]}
    for _ in range(1, k):
        if not remaining:
            break
        rem_list = list(remaining)
        dists = np.min(np.linalg.norm(
            X_scaled[rem_list][:, np.newaxis, :] - X_scaled[selected][np.newaxis, :, :],
            axis=2), axis=1)
        max_dist = dists.max() if dists.max() > 1e-8 else 1.0
        scores = (1.0 - div_weight) * sims[rem_list] + div_weight * (dists / max_dist)
        next_idx = rem_list[int(np.argmax(scores))]
        selected.append(next_idx)
        remaining.remove(next_idx)
    return np.array(selected)


def safe_get_fp(smiles, radius=FP_RADIUS, nbits=FP_NBITS):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            try:
                Chem.SanitizeMol(mol, catchErrors=True)
            except Exception:
                return None
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        bv = DataStructs.ExplicitBitVect(nbits)
        for i in fp.GetOnBits():
            bv.SetBit(i)
        return bv
    except Exception:
        return None


# ==================== FP缓存读写（仅修改此处键名为 _radius/_nbits/_cache）====================
def load_fp_cache(cache_path, expected_smiles_set):
    """
    加载FP缓存，校验完整性。
    返回: (cache_dict, is_valid)
    """
    if not os.path.exists(cache_path):
        return {}, False
    try:
        with open(cache_path, "rb") as f:
            meta = pickle.load(f)

        # 优先识别新格式（带下划线前缀，与2.2兼容）
        if "_cache" in meta and "_radius" in meta and "_nbits" in meta:
            if meta["_radius"] != FP_RADIUS or meta["_nbits"] != FP_NBITS:
                logging.warning("⚠️ FP缓存参数不匹配(新格式)，将重新生成")
                return {}, False
            cache = meta["_cache"]
        # 向后兼容旧格式（无下划线前缀）
        elif "cache" in meta and "radius" in meta and "nbits" in meta:
            if meta["radius"] != FP_RADIUS or meta["nbits"] != FP_NBITS:
                logging.warning("⚠️ FP缓存参数不匹配(旧格式)，将重新生成")
                return {}, False
            cache = meta["cache"]
        else:
            logging.warning("⚠️ FP缓存格式无法识别，将重新生成")
            return {}, False

        # 校验覆盖率（允许少量缺失，后续补算）
        coverage = len(cache) / max(len(expected_smiles_set), 1)
        if coverage < 0.95:
            logging.warning(f"⚠️ FP缓存覆盖率仅{coverage:.1%}，将重新生成")
            return {}, False

        logging.info(f"📂 FP缓存加载成功: {len(cache):,} 条 (来源: {cache_path})")
        return cache, True
    except Exception as e:
        logging.warning(f"⚠️ FP缓存读取失败: {e}，将重新生成")
        return {}, False


def save_fp_cache(cache_path, cache_dict):
    """保存FP缓存到磁盘（使用 _radius/_nbits/_cache 包装层，与2.2兼容）"""
    meta = {
        "_radius": FP_RADIUS,
        "_nbits": FP_NBITS,
        "_cache": cache_dict,
        "n_entries": len(cache_dict),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    logging.info(f"💾 FP缓存已保存: {cache_path} ({file_size_mb:.1f} MB, {len(cache_dict):,} 条)")


# ==================== 待查询分子加载 ====================
def load_query_set():
    if QUERY_SOURCE == "csv":
        df = pd.read_csv(QUERY_FILE_PATH)
        id_col = QUERY_ID_COL if QUERY_ID_COL and QUERY_ID_COL in df.columns else None
        items = []
        for i, row in df.iterrows():
            cid = str(row[id_col]) if id_col else f"{i + 1:04d}"
            items.append((cid, str(row[QUERY_SMILES_COL])))
        return items
    elif QUERY_SOURCE == "parquet":
        df = pd.read_parquet(QUERY_FILE_PATH)
        id_col = QUERY_ID_COL if QUERY_ID_COL and QUERY_ID_COL in df.columns else None
        items = []
        for i, row in df.iterrows():
            cid = str(row[id_col]) if id_col else f"{i + 1:04d}"
            items.append((cid, str(row[QUERY_SMILES_COL])))
        return items
    else:
        return [(f"{i + 1:04d}", smi) for i, smi in enumerate(QUERY_SMILES_LIST)]


# ==================== FAISS 搜索引擎 ====================
class DTISimilaritySearcher:
    def __init__(self, desc_path, affinity_path, fp_cache_path=FP_CACHE_PATH):
        logging.info("📥 加载描述符数据库...")
        df_desc = pl.read_parquet(desc_path)
        self.df_valid = df_desc.filter(pl.col("MW").is_not_null())
        del df_desc

        self.X_db = self.df_valid.select(DESC_COLS).to_numpy()
        self.all_smiles = self.df_valid[SMILES_COL_DESC].to_list()
        self.vsa_col_indices = [DESC_COLS.index(c) for c in DESC_COLS if 'VSA' in c]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X_db).astype(np.float32)
        if not X_scaled.flags['C_CONTIGUOUS']:
            X_scaled = np.ascontiguousarray(X_scaled)
        faiss.normalize_L2(X_scaled)
        self.X_db_scaled = X_scaled

        dimension = X_scaled.shape[1]
        logging.info(f"🚀 构建 FAISS 索引 (维度={dimension}, 数据量={len(X_scaled):,})...")
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(X_scaled)
        logging.info("✅ FAISS 索引构建完成")

        min_aff = min(AFFINITY_THRESHOLD_STEPS)
        logging.info(f"📥 预加载DTI索引 (affinity >= {min_aff})...")
        df_aff = pl.read_parquet(affinity_path).filter(pl.col("affinity") >= min_aff)

        self.smiles_to_targets = {}
        self.smiles_target_affinity = {}
        for r in tqdm(df_aff.select([SMILES_COL_AFF, "gene_symbol", "affinity"]).to_dicts(),
                      desc=" 📇 构建靶点索引", unit="rec", ncols=90):
            smi = r[SMILES_COL_AFF]
            gene = r["gene_symbol"]
            aff = float(r["affinity"])
            if smi and gene:
                self.smiles_to_targets.setdefault(smi, set()).add(gene)
                key = (smi, gene)
                if key not in self.smiles_target_affinity or aff > self.smiles_target_affinity[key]:
                    self.smiles_target_affinity[key] = aff
        del df_aff

        # ★ FP缓存：有则加载，无则生成并保存
        smiles_set = set(self.all_smiles)
        self._fp_cache, cache_valid = load_fp_cache(fp_cache_path, smiles_set)

        if not cache_valid:
            logging.info("🧬 预计算FP缓存...")
            self._fp_cache = {}
            for smi in tqdm(self.all_smiles, desc=" 💾 FP预计算", unit="mol", ncols=90):
                fp = safe_get_fp(smi)
                if fp:
                    self._fp_cache[smi] = fp
            save_fp_cache(fp_cache_path, self._fp_cache)
        else:
            # 补算缓存中缺失的条目
            missing = [s for s in self.all_smiles if s not in self._fp_cache]
            if missing:
                logging.info(f"🔧 补算 {len(missing):,} 条缺失FP...")
                for smi in tqdm(missing, desc=" 🔧 FP补算", unit="mol", ncols=90):
                    fp = safe_get_fp(smi)
                    if fp:
                        self._fp_cache[smi] = fp
                save_fp_cache(fp_cache_path, self._fp_cache)

        logging.info(f"✅ 初始化完成: {len(self._fp_cache):,} FP缓存")
        gc.collect()

    def search(self, query_smiles, top_k=TOP_N):
        result = {
            'exact_matches': [],
            'similar_candidates': [],
            'recommended_genes': [],
            'gene_score_map': {},
        }

        norm_query = normalize_smiles(query_smiles)
        q_vec = np.array(calc_descriptors(norm_query), dtype=np.float32).reshape(1, -1)
        q_scaled = self.scaler.transform(q_vec).astype(np.float32)
        if not q_scaled.flags['C_CONTIGUOUS']:
            q_scaled = np.ascontiguousarray(q_vec)
        faiss.normalize_L2(q_scaled)
        q_logp, q_tpsa = q_vec[0, 1], q_vec[0, 2]

        sims, indices = self.index.search(q_scaled, FAISS_SEARCH_K)
        sims, indices = sims[0], indices[0]

        valid_mask = (indices != -1) & (sims >= SIM_THRESHOLD)
        filtered_indices = indices[valid_mask]
        filtered_sims = sims[valid_mask]
        if len(filtered_indices) == 0:
            return result

        actual_pool_size = MAX_POOL_SIZE if len(filtered_indices) > DENSE_REGION_THRESHOLD else BASE_POOL_SIZE

        X_filt = self.X_db[filtered_indices]
        mw_ok = (X_filt[:, 0] >= MIN_MW) & (X_filt[:, 0] <= MAX_MW)
        logp_ok = np.abs(X_filt[:, 1] - q_logp) <= LOGP_TOLERANCE
        tpsa_ok = np.abs(X_filt[:, 2] - q_tpsa) <= TPSA_TOLERANCE
        vsa_ok = np.sum(np.abs(X_filt[:, self.vsa_col_indices]) > 1e-6, axis=1) >= MIN_NONZERO_VSA
        valid_mask_prop = mw_ok & logp_ok & tpsa_ok & vsa_ok

        filtered_indices = filtered_indices[valid_mask_prop]
        filtered_sims = filtered_sims[valid_mask_prop]
        X_filtered_scaled = self.X_db_scaled[filtered_indices]

        if len(filtered_sims) == 0:
            return result

        if len(filtered_sims) > actual_pool_size and POOL_DIVERSITY_WEIGHT > 0:
            pool_order = diverse_topk(filtered_sims, X_filtered_scaled,
                                      actual_pool_size, POOL_DIVERSITY_WEIGHT)
        else:
            pool_order = np.argsort(filtered_sims)[::-1][:actual_pool_size]

        pool_indices = filtered_indices[pool_order]
        pool_desc_sims = filtered_sims[pool_order]

        q_fp = safe_get_fp(norm_query)
        if q_fp is None:
            return result

        pool_smiles = [self.all_smiles[i] for i in pool_indices]
        pool_fps = [self._fp_cache.get(s) or safe_get_fp(s) for s in pool_smiles]

        valid_fp_mask = np.array([fp is not None for fp in pool_fps])
        if valid_fp_mask.sum() == 0:
            return result

        tanimoto_sims_full = np.zeros(len(pool_fps), dtype=np.float32)
        tanimoto_sims_full[valid_fp_mask] = np.array(
            DataStructs.BulkTanimotoSimilarity(q_fp, [f for f in pool_fps if f]),
            dtype=np.float32)

        fp_threshold_used = FP_TANIMOTO_THRESHOLD
        fp_mask = tanimoto_sims_full >= fp_threshold_used
        min_acceptable = max(MIN_FP_RESULTS, top_k + 5)
        if fp_mask.sum() < min_acceptable:
            for ft in FP_FALLBACK_THRESHOLDS:
                fp_mask = tanimoto_sims_full >= ft
                if fp_mask.sum() >= min_acceptable:
                    fp_threshold_used = ft
                    break
            else:
                fp_threshold_used = FP_FALLBACK_THRESHOLDS[-1]
                fp_mask = tanimoto_sims_full >= fp_threshold_used

        final_indices = pool_indices[fp_mask]
        final_desc_sims = pool_desc_sims[fp_mask]
        final_tani_sims = tanimoto_sims_full[fp_mask]
        sort_keys = np.lexsort((-final_desc_sims, -final_tani_sims))

        exact_matches, similar_candidates = [], []
        for idx_pos in sort_keys:
            tani = float(final_tani_sims[idx_pos])
            db_idx = int(final_indices[idx_pos])
            smi = self.all_smiles[db_idx]
            entry = {
                'tanimoto': round(tani, 4),
                'desc_cosine': round(float(final_desc_sims[idx_pos]), 4),
                'SMILES': smi,
                'MW': round(float(self.X_db[db_idx, 0]), 1),
                'LogP': round(float(self.X_db[db_idx, 1]), 2),
                'TPSA': round(float(self.X_db[db_idx, 2]), 1),
            }
            if tani >= 0.999:
                entry['match_type'] = 'EXACT'
                exact_matches.append(entry)
            else:
                entry['match_type'] = 'SIMILAR'
                similar_candidates.append(entry)

        result['exact_matches'] = exact_matches
        result['similar_candidates'] = similar_candidates[:top_k]

        all_hit_smiles = list({e['SMILES'] for e in exact_matches + similar_candidates[:top_k]})
        query_norm = normalize_smiles(query_smiles)
        candidates = [s for s in all_hit_smiles if normalize_smiles(s) != query_norm]

        gene_scores = {}
        for smi in candidates:
            ns = normalize_smiles(smi)
            for g in self.smiles_to_targets.get(ns, set()):
                gene_scores[g] = gene_scores.get(g, 0.0) + 1.0

        result['recommended_genes'] = [g for g, _ in sorted(gene_scores.items(), key=lambda x: -x[1])]
        result['gene_score_map'] = gene_scores
        return result


# ==================== 结果保存 ====================
def save_compound_results(compound_dir, query_smiles, compound_id, result, searcher):
    os.makedirs(compound_dir, exist_ok=True)
    norm_query = normalize_smiles(query_smiles)

    exact_rows = []
    for e in result['exact_matches']:
        exact_rows.append({
            'query_smiles': query_smiles, 'compound_id': compound_id,
            'match_smiles': e['SMILES'], 'match_type': e['match_type'],
            'tanimoto': e['tanimoto'], 'desc_cosine': e['desc_cosine'],
            'MW': e['MW'], 'LogP': e['LogP'], 'TPSA': e['TPSA'],
        })
    _write_csv(os.path.join(compound_dir, FILE_EXACT_MATCHES), exact_rows,
               ['query_smiles', 'compound_id', 'match_smiles', 'match_type',
                'tanimoto', 'desc_cosine', 'MW', 'LogP', 'TPSA'])

    sim_rows = []
    for rank, e in enumerate(result['similar_candidates'], 1):
        sim_rows.append({
            'query_smiles': query_smiles, 'compound_id': compound_id, 'rank': rank,
            'match_smiles': e['SMILES'], 'match_type': e['match_type'],
            'tanimoto': e['tanimoto'], 'desc_cosine': e['desc_cosine'],
            'MW': e['MW'], 'LogP': e['LogP'], 'TPSA': e['TPSA'],
        })
    _write_csv(os.path.join(compound_dir, FILE_TOP_SIMILAR), sim_rows,
               ['query_smiles', 'compound_id', 'rank', 'match_smiles', 'match_type',
                'tanimoto', 'desc_cosine', 'MW', 'LogP', 'TPSA'])

    all_hits = result['exact_matches'] + result['similar_candidates']
    detail_rows = []
    gene_agg = {}

    for e in all_hits:
        hit_smi = e['SMILES']
        hit_norm = normalize_smiles(hit_smi)
        tani = e['tanimoto']
        targets = searcher.smiles_to_targets.get(hit_norm, set())
        for gene in targets:
            aff = searcher.smiles_target_affinity.get((hit_norm, gene), None)
            detail_rows.append({
                'query_smiles': query_smiles, 'compound_id': compound_id,
                'source_smiles': hit_smi, 'tanimoto': tani,
                'gene_symbol': gene,
                'affinity': round(aff, 4) if aff is not None else "",
            })
            if gene not in gene_agg:
                gene_agg[gene] = {'score': 0.0, 'sources': []}
            gene_agg[gene]['score'] += 1.0
            gene_agg[gene]['sources'].append({
                'smiles': hit_smi, 'tanimoto': tani,
                'affinity': round(aff, 4) if aff is not None else ""
            })

    detail_rows.sort(key=lambda x: (-x['tanimoto'], x['gene_symbol']))
    _write_csv(os.path.join(compound_dir, FILE_TGT_DETAIL), detail_rows,
               ['query_smiles', 'compound_id', 'source_smiles', 'tanimoto',
                'gene_symbol', 'affinity'])

    summary_rows = []
    for gene in result['recommended_genes']:
        info = gene_agg.get(gene, {'score': 0, 'sources': []})
        n_sources = len(info['sources'])
        max_tani = max((s['tanimoto'] for s in info['sources']), default=0)
        max_aff = max((s['affinity'] for s in info['sources'] if s['affinity'] != ""), default="")
        summary_rows.append({
            'query_smiles': query_smiles, 'compound_id': compound_id,
            'rank': len(summary_rows) + 1, 'gene_symbol': gene,
            'vote_score': info['score'], 'n_source_compounds': n_sources,
            'max_tanimoto': max_tani, 'max_affinity': max_aff,
        })
    _write_csv(os.path.join(compound_dir, FILE_TGT_SUMMARY), summary_rows,
               ['query_smiles', 'compound_id', 'rank', 'gene_symbol',
                'vote_score', 'n_source_compounds', 'max_tanimoto', 'max_affinity'])

    query_targets = searcher.smiles_to_targets.get(norm_query, set())
    dti_rows = []
    for e in all_hits:
        hit_smi = e['SMILES']
        hit_norm = normalize_smiles(hit_smi)
        hit_targets = searcher.smiles_to_targets.get(hit_norm, set())
        shared = query_targets & hit_targets
        for gene in shared:
            dti_rows.append({
                'query_smiles': query_smiles, 'compound_id': compound_id,
                'node_a': query_smiles, 'node_b': hit_smi,
                'shared_target': gene, 'tanimoto_ab': e['tanimoto'],
                'edge_type': 'shared_target',
            })
        if not shared and hit_targets:
            for gene in hit_targets:
                dti_rows.append({
                    'query_smiles': query_smiles, 'compound_id': compound_id,
                    'node_a': query_smiles, 'node_b': hit_smi,
                    'shared_target': gene, 'tanimoto_ab': e['tanimoto'],
                    'edge_type': 'inferred_by_similarity',
                })
    _write_csv(os.path.join(compound_dir, FILE_DTI_EDGES), dti_rows,
               ['query_smiles', 'compound_id', 'node_a', 'node_b',
                'shared_target', 'tanimoto_ab', 'edge_type'])

    return {
        'n_exact': len(exact_rows), 'n_similar': len(sim_rows),
        'n_tgt_detail': len(detail_rows), 'n_tgt_summary': len(summary_rows),
        'n_dti_edges': len(dti_rows),
    }


def _write_csv(filepath, rows, fieldnames):
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ==================== 主流程 ====================
if __name__ == "__main__":
    t0 = time.perf_counter()

    searcher = DTISimilaritySearcher(DESC_PARQUET_PATH, AFFINITY_PARQUET_PATH, FP_CACHE_PATH)

    query_set = load_query_set()
    logging.info(f"📋 待查询分子数: {len(query_set)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pbar = tqdm(query_set, desc="🔬 查询", unit="mol", ncols=100)
    success_count = 0
    fail_count = 0
    global_stats = []

    for compound_id, smiles in pbar:
        try:
            result = searcher.search(smiles, TOP_N)
            compound_dir = os.path.join(OUTPUT_DIR, compound_id)
            stats = save_compound_results(compound_dir, smiles, compound_id, result, searcher)
            stats['compound_id'] = compound_id
            stats['smiles'] = smiles
            global_stats.append(stats)
            success_count += 1
        except Exception as e:
            logging.warning(f"⚠️ [{compound_id}] {smiles[:50]} 失败: {e}")
            fail_count += 1
            continue

        if (success_count + 1) % 20 == 0:
            gc.collect()
            try:
                ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
            except Exception:
                pass

        pbar.set_postfix({"✅": success_count, "❌": fail_count})

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 80}")
    print(f"🏁 查询完成 | 成功={success_count} 失败={fail_count} | 耗时={elapsed:.1f}s")
    print(f"📂 结果目录: {OUTPUT_DIR}")
    print(f"{'=' * 80}")

    if global_stats:
        print(f"\n{'ID':<8}{'SMILES':<35}{'Exact':>6}{'Sim':>6}{'Tgts':>6}{'DTI':>6}")
        print("-" * 70)
        for s in global_stats:
            print(f"{s['compound_id']:<8}{s['smiles'][:33]:<35}"
                  f"{s['n_exact']:>6}{s['n_similar']:>6}"
                  f"{s['n_tgt_summary']:>6}{s['n_dti_edges']:>6}")

    logging.info("✅ 全部完成")