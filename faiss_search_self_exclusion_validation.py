import os, gc, csv, ctypes, logging, random, time, math, pickle
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
import faiss
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')
os.environ["POLARS_MAX_THREADS"] = "6"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# ==================== 配置区 ====================
DESC_PARQUET_PATH = r"E:\research_project\DTI预测\descriptors_final.parquet"
AFFINITY_PARQUET_PATH = r"D:\E盘模型暂存\dti_merged_dataset.parquet"
OUTPUT_DIR = r"E:\research_project\DTI预测"
RESULT_CSV = os.path.join(OUTPUT_DIR, "validation_results_faiss_2k.csv")
FAILED_TXT = os.path.join(OUTPUT_DIR, "validation_failed_smiles_2k.txt")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
FP_CACHE_PATH = os.path.join(OUTPUT_DIR, "fp_cache.pkl")
os.makedirs(FIGURE_DIR, exist_ok=True)

TOP_N = 20
EVAL_KS = list(range(1, 21))
AFFINITY_THRESHOLD_STEPS = [6.0, 5.0, 4.0, 3.0]
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

VALIDATION_CONFIG = {
    "samples_per_bin": 500,
    "target_bins": [(3, 5), (6, 10), (11, 20), (21, 50)],
    "random_seed": 42,
}

DESC_COLS = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'LabuteASA'] \
    + [f'PEOE_VSA{i}' for i in range(1, 15)] \
    + [f'SMR_VSA{i}' for i in range(1, 11)]

SMILES_COL_DESC = "SMILES"
SMILES_COL_AFF = "compound_iso_smiles_norm"

CSV_HEADER = ["smiles", "bin", "n_true_targets", "n_recommended", "first_hit_rank", "mrr"] \
    + [f"recall@{k}" for k in EVAL_KS] \
    + [f"precision@{k}" for k in EVAL_KS] \
    + [f"ndcg@{k}" for k in EVAL_KS]


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
            X_scaled[rem_list][:, np.newaxis, :] - X_scaled[selected][np.newaxis, :, :], axis=2), axis=1)
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
        for r in tqdm(df_aff.select([SMILES_COL_AFF, "gene_symbol"]).to_dicts(),
                      desc=" 📇 构建靶点索引", unit="rec", ncols=90):
            if r[SMILES_COL_AFF] and r["gene_symbol"]:
                self.smiles_to_targets.setdefault(r[SMILES_COL_AFF], set()).add(r["gene_symbol"])
        del df_aff

        # ✅ FP缓存：读取 → 校验 → 缺失补算 → 持久化
        self._fp_cache = {}
        cache_loaded = False
        if os.path.exists(fp_cache_path):
            try:
                with open(fp_cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                if (isinstance(cached_data, dict)
                        and cached_data.get("_radius") == FP_RADIUS
                        and cached_data.get("_nbits") == FP_NBITS
                        and "_cache" in cached_data):
                    self._fp_cache = cached_data["_cache"]
                    cache_loaded = True
                    logging.info(f"💾 FP缓存已从磁盘加载: {len(self._fp_cache):,} 条 "
                                 f"(radius={FP_RADIUS}, nbits={FP_NBITS})")
                else:
                    logging.warning("⚠️ FP缓存元信息不匹配或格式无效，将重新生成")
            except Exception as e:
                logging.warning(f"⚠️ FP缓存加载失败 ({e})，将重新生成")

        missing_smiles = [s for s in self.all_smiles if s not in self._fp_cache]
        if missing_smiles:
            action = "补算缺失" if cache_loaded else "全量生成"
            logging.info(f"🧬 FP缓存{action}: {len(missing_smiles):,}/{len(self.all_smiles):,} 个分子待计算...")
            for smi in tqdm(missing_smiles, desc=f" 💾 FP{action}", unit="mol", ncols=90):
                fp = safe_get_fp(smi)
                if fp is not None:
                    self._fp_cache[smi] = fp
            try:
                cache_payload = {
                    "_radius": FP_RADIUS,
                    "_nbits": FP_NBITS,
                    "_cache": self._fp_cache
                }
                with open(fp_cache_path, "wb") as f:
                    pickle.dump(cache_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                logging.info(f"💾 FP缓存已保存至: {fp_cache_path} ({len(self._fp_cache):,} 条)")
            except Exception as e:
                logging.warning(f"⚠️ FP缓存保存失败: {e}")
        elif cache_loaded:
            logging.info("✅ FP缓存完整，无需补算")
        else:
            logging.info(f"✅ FP缓存就绪: {len(self._fp_cache):,} 条")

        logging.info(f"✅ 初始化完成: {len(self._fp_cache):,} FP缓存可用")
        gc.collect()

    def get_ground_truth(self, smiles):
        return self.smiles_to_targets.get(normalize_smiles(smiles), set())

    def search(self, query_smiles, top_k=TOP_N, self_hit_weight=0.0, prefer_self_hit=False):
        result = {'exact_matches': [], 'similar_candidates': [], 'recommended_genes': []}
        norm_query = normalize_smiles(query_smiles)

        q_vec = np.array(calc_descriptors(norm_query), dtype=np.float32).reshape(1, -1)
        q_scaled = self.scaler.transform(q_vec).astype(np.float32)
        if not q_scaled.flags['C_CONTIGUOUS']:
            q_scaled = np.ascontiguousarray(q_scaled)
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
            pool_order = diverse_topk(filtered_sims, X_filtered_scaled, actual_pool_size, POOL_DIVERSITY_WEIGHT)
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
            DataStructs.BulkTanimotoSimilarity(q_fp, [f for f in pool_fps if f]), dtype=np.float32)

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
            entry = {'tanimoto': round(tani, 4), 'desc_cosine': round(float(final_desc_sims[idx_pos]), 4),
                     'SMILES': smi, 'MW': round(float(self.X_db[db_idx, 0]), 1),
                     'LogP': round(float(self.X_db[db_idx, 1]), 2), 'TPSA': round(float(self.X_db[db_idx, 2]), 1)}
            if tani >= 0.999:
                entry['match_type'] = 'EXACT'
                exact_matches.append(entry)
            else:
                entry['match_type'] = 'SIMILAR'
                similar_candidates.append(entry)

        result['exact_matches'] = exact_matches
        result['similar_candidates'] = similar_candidates[:top_k]

        all_smiles = list({e['SMILES'] for e in exact_matches + similar_candidates[:top_k]})
        query_norm = normalize_smiles(query_smiles)
        candidates = list(set(all_smiles + [query_norm])) if self_hit_weight > 0 \
            else [s for s in all_smiles if normalize_smiles(s) != query_norm]

        gene_scores = {}
        for smi in candidates:
            ns = normalize_smiles(smi)
            w = self_hit_weight if ns == query_norm else 1.0
            for g in self.smiles_to_targets.get(ns, set()):
                gene_scores[g] = gene_scores.get(g, 0.0) + w
        result['recommended_genes'] = [g for g, _ in sorted(gene_scores.items(), key=lambda x: -x[1])]
        return result


# ==================== 评估指标 ====================
@dataclass
class EvalMetrics:
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    first_hit_rank: Optional[int] = None
    n_true_targets: int = 0
    n_recommended: int = 0


def compute_metrics(rec_genes, gt, eval_ks=EVAL_KS):
    m = EvalMetrics(n_true_targets=len(gt), n_recommended=len(rec_genes))
    for k in eval_ks:
        m.recall_at_k[k] = 0.0
        m.precision_at_k[k] = 0.0
        m.ndcg_at_k[k] = 0.0
    if not gt:
        return m
    fhr = next((i + 1 for i, g in enumerate(rec_genes) if g in gt), None)
    m.first_hit_rank = fhr
    m.mrr = 1.0 / fhr if fhr else 0.0
    for k in eval_ks:
        topk = rec_genes[:k]
        hits = sum(1 for g in topk if g in gt)
        m.recall_at_k[k] = hits / len(gt)
        m.precision_at_k[k] = hits / k if k else 0.0
        dcg = sum(1 / np.log2(i + 2) for i, g in enumerate(topk) if g in gt)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(gt))))
        m.ndcg_at_k[k] = dcg / idcg if idcg else 0.0
    return m


def build_validation_set(searcher):
    cfg = VALIDATION_CONFIG
    random.seed(cfg["random_seed"])
    tc = {s: len(t) for s, t in searcher.smiles_to_targets.items()}
    val_set = []
    for lo, hi in cfg["target_bins"]:
        cands = [s for s, c in tc.items() if lo <= c <= hi]
        n = min(cfg["samples_per_bin"], len(cands))
        for smi in random.sample(cands, n):
            val_set.append({"smiles": smi, "ground_truth": searcher.get_ground_truth(smi),
                            "n_true_targets": len(searcher.get_ground_truth(smi)), "bin": f"{lo}-{hi}"})
        logging.info(f" 📦 [{lo:>2}-{hi:>2}]: {len(cands):>6} → {n}")
    random.shuffle(val_set)
    logging.info(f"✅ 验证集: {len(val_set)} 分子")
    return val_set


def load_completed_smiles(csv_path):
    completed = set()
    if os.path.exists(csv_path):
        try:
            df = pl.read_csv(csv_path)
            completed = set(df["smiles"].to_list())
            logging.info(f"🔄 检测到已有结果文件，已处理 {len(completed)} 个分子，将自动跳过")
        except Exception as e:
            logging.warning(f"⚠️ 读取已有结果文件失败: {e}，将从头开始")
    return completed


def append_result_to_csv(csv_path, record: dict):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writeheader()
        writer.writerow(record)


# ✅ 修复：改为批量写入，避免逐行追加导致的数据丢失
def save_failed_smiles(txt_path, failed_smiles_list: list):
    """在主循环结束后一次性写入所有失败SMILES"""
    if not failed_smiles_list:
        return
    with open(txt_path, "w", encoding="utf-8") as f:
        for smi in failed_smiles_list:
            f.write(smi + "\n")
    logging.info(f"💾 失败记录已保存: {len(failed_smiles_list)} 条 → {txt_path}")


# ==================== 可视化模块 (Top-10) ====================
def generate_figures(csv_path, figure_dir):
    """生成4张核心评估图表"""
    plt.rcParams.update({
        'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight'
    })

    df = pl.read_csv(csv_path).to_pandas()
    bins_order = ["3-5", "6-10", "11-20", "21-50"]
    bin_colors = {"3-5": "#2196F3", "6-10": "#4CAF50", "11-20": "#FF9800", "21-50": "#F44336"}

    # ========== 图1: 分层 Recall/Precision 柱状图 + 误差棒 ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics_to_plot = [("recall@10", "Recall@10"), ("precision@10", "Precision@10")]
    for ax, (col, title) in zip(axes, metrics_to_plot):
        means = [df[df["bin"] == b][col].mean() for b in bins_order]
        sems = [df[df["bin"] == b][col].sem() for b in bins_order]
        colors = [bin_colors[b] for b in bins_order]
        bars = ax.bar(bins_order, means, yerr=[1.96 * s for s in sems],
                      capsize=5, color=colors, edgecolor='white', linewidth=0.8)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Target Count Bin")
        ax.set_ylabel(title)
        ax.set_ylim(0, max(means) * 1.3)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle("Stratified Performance by Target Abundance", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "01_stratified_recall_precision.png"))
    plt.close()
    logging.info(" 📊 图1: 分层Recall/Precision 已保存")

    # ========== 图2: nDCG@K 曲线图 ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    ks = list(range(1, 11))
    for b in bins_order:
        sub = df[df["bin"] == b]
        ndcg_vals = [sub[f"ndcg@{k}"].mean() for k in ks]
        ax.plot(ks, ndcg_vals, 'o-', label=b, color=bin_colors[b], linewidth=2, markersize=4)
    global_ndcg = [df[f"ndcg@{k}"].mean() for k in ks]
    ax.plot(ks, global_ndcg, 's--', label='Overall', color='#333333', linewidth=2.5, markersize=5)
    ax.set_xlabel("K (Top-K Recommendations)")
    ax.set_ylabel("nDCG@K")
    ax.set_title("Ranking Quality Decay Curve (nDCG@K)", fontweight='bold')
    ax.legend(title="Target Bin", loc='lower left')
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "02_ndcg_curve.png"))
    plt.close()
    logging.info(" 📊 图2: nDCG@K 曲线 已保存")

    # ========== 图3: MRR 分布小提琴图 ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    violin_data = [df[df["bin"] == b]["mrr"].dropna().values for b in bins_order]
    parts = ax.violinplot(violin_data, positions=range(len(bins_order)), showmeans=True, showmedians=True)
    for pc, b in zip(parts['bodies'], bins_order):
        pc.set_facecolor(bin_colors[b])
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(bins_order)))
    ax.set_xticklabels(bins_order)
    ax.set_xlabel("Target Count Bin")
    ax.set_ylabel("MRR")
    ax.set_title("MRR Distribution by Target Abundance", fontweight='bold')
    ax.axhline(y=df["mrr"].mean(), color='gray', linestyle='--', alpha=0.5,
               label=f'Overall Mean={df["mrr"].mean():.3f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "03_mrr_violin.png"))
    plt.close()
    logging.info(" 📊 图3: MRR小提琴图 已保存")

    # ========== 图4: 失败案例分析散点图 ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    df["is_fail"] = (df["recall@10"] == 0).astype(int)
    prop_pairs = [("n_true_targets", "Number of True Targets", "log"),
                  ("first_hit_rank", "First Hit Rank", "linear"),
                  ("n_recommended", "N Recommended Genes", "linear")]
    for ax, (col, xlabel, scale) in zip(axes, prop_pairs):
        success = df[df["is_fail"] == 0]
        fail = df[df["is_fail"] == 1]
        ax.scatter(success[col], success["mrr"], c='#4CAF50', alpha=0.3, s=15, label=f'Success (n={len(success)})')
        ax.scatter(fail[col], fail["mrr"], c='#F44336', alpha=0.6, s=25, marker='x',
                   label=f'Fail R@10=0 (n={len(fail)})')
        ax.set_xlabel(xlabel)
        ax.set_ylabel("MRR")
        ax.set_title(f"Failure Analysis: {xlabel}", fontweight='bold')
        if scale == "log":
            ax.set_xscale("log")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.suptitle("R@10=0 Failure Case Characterization", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "04_failure_analysis.png"))
    plt.close()
    logging.info(" 📊 图4: 失败案例分析 已保存")

    logging.info(f"✅ 全部4张图表已保存至: {figure_dir}")


# ==================== 主流程 ====================
if __name__ == "__main__":
    t0 = time.perf_counter()
    searcher = DTISimilaritySearcher(DESC_PARQUET_PATH, AFFINITY_PARQUET_PATH)
    val_set = build_validation_set(searcher)

    completed_smiles = load_completed_smiles(RESULT_CSV)
    pending_set = [item for item in val_set if item["smiles"] not in completed_smiles]
    logging.info(f"📋 待处理: {len(pending_set)}/{len(val_set)} 分子")

    # ✅ 修复：用列表收集失败SMILES，循环结束后统一写入
    failed_smiles_list = []

    if not pending_set:
        logging.info("✅ 所有分子已处理完毕，直接进入汇总阶段")
    else:
        logging.info("\n🚀 开始验证 (FAISS加速 + 增量保存模式)...")
        pbar = tqdm(pending_set, desc="🔬 验证", unit="mol", ncols=100)
        success_count = 0
        fail_count = 0

        for i, item in enumerate(pbar):
            try:
                res = searcher.search(item["smiles"], TOP_N, self_hit_weight=0.0, prefer_self_hit=False)
                met = compute_metrics(res['recommended_genes'], item['ground_truth'])

                record = {
                    "smiles": item["smiles"], "bin": item["bin"],
                    "n_true_targets": item["n_true_targets"], "n_recommended": met.n_recommended,
                    "first_hit_rank": met.first_hit_rank, "mrr": round(met.mrr, 4),
                }
                for k in EVAL_KS:
                    record[f"recall@{k}"] = round(met.recall_at_k[k], 4)
                    record[f"precision@{k}"] = round(met.precision_at_k[k], 4)
                    record[f"ndcg@{k}"] = round(met.ndcg_at_k[k], 4)

                append_result_to_csv(RESULT_CSV, record)
                success_count += 1

            except Exception as e:
                logging.warning(f"⚠️ {item['smiles'][:50]} 失败: {e}")
                failed_smiles_list.append(item["smiles"])  # ✅ 收集到内存列表
                fail_count += 1
                continue

            if (i + 1) % 50 == 0:
                gc.collect()
                try:
                    ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
                except Exception:
                    pass

            pbar.set_postfix({"✅": success_count, "❌": fail_count})

        logging.info(f"🏁 验证循环结束: 成功={success_count}, 失败={fail_count}")

    # ✅ 修复：循环结束后一次性写入所有失败记录
    save_failed_smiles(FAILED_TXT, failed_smiles_list)

    # ==================== 汇总报告 (Top-10) ====================
    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 100}")

    if not os.path.exists(RESULT_CSV):
        logging.error("❌ 无结果文件，无法生成汇总报告")
        exit(1)

    df_results = pl.read_csv(RESULT_CSV)
    n_total = len(df_results)
    print(f"📊 验证结果汇总 | 总记录={n_total} | 本次耗时={elapsed:.1f}s")
    print("=" * 100)

    if n_total == 0:
        logging.error("❌ 结果为空")
        exit(1)

    summary_ks = [1, 5, 10]
    print(f"\n{'指标':<15}", end="")
    for k in summary_ks:
        print(f"{'@' + str(k):>10}", end="")
    print(f"{'MRR':>10}\n{'-' * 55}")

    for name, col_prefix in [("Recall", "recall@"), ("Precision", "precision@"), ("nDCG", "ndcg@")]:
        print(f"{name:<15}", end="")
        for k in summary_ks:
            col = f"{col_prefix}{k}"
            val = df_results[col].mean() if col in df_results.columns else 0.0
            print(f"{val:>10.4f}", end="")
        print(f"{'—':>10}")

    print(f"{'MRR':<15}", end="")
    for _ in summary_ks:
        print(f"{'—':>10}", end="")
    mrr_val = df_results["mrr"].mean() if "mrr" in df_results.columns else 0.0
    print(f"{mrr_val:>10.4f}")

    print(f"\n📈 分层分析\n{'-' * 60}")
    print(f"{'分桶':<10}{'N':>6}{'R@10':>10}{'P@10':>10}{'MRR':>8}\n{'-' * 60}")
    for lo, hi in VALIDATION_CONFIG["target_bins"]:
        bl = f"{lo}-{hi}"
        sub = df_results.filter(pl.col("bin") == bl)
        if len(sub) > 0:
            r10 = sub["recall@10"].mean() if "recall@10" in sub.columns else 0.0
            p10 = sub["precision@10"].mean() if "precision@10" in sub.columns else 0.0
            mrr = sub["mrr"].mean() if "mrr" in sub.columns else 0.0
            print(f"{bl:<10}{len(sub):>6}{r10:>10.4f}{p10:>10.4f}{mrr:>8.4f}")

    if "recall@10" in df_results.columns:
        zr = df_results.filter(pl.col("recall@10") == 0).height
        print(f"\n⚠️ R@10=0: {zr}/{n_total} ({zr / n_total * 100:.1f}%)")

    print("=" * 100)

    logging.info("\n🎨 正在生成可视化图表...")
    generate_figures(RESULT_CSV, FIGURE_DIR)

    logging.info(f"💾 结果文件: {RESULT_CSV}")
    if os.path.exists(FAILED_TXT):
        logging.info(f"💾 失败记录: {FAILED_TXT}")