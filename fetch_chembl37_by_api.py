import os
import json
import time
import random
import gc
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ==================== 全局配置 ====================
ACTIVITY_CACHE = r"D:\E盘模型暂存\chembl_activity_cache.parquet"
PROGRESS_FILE = "activity_progress.json"
CHEMREPS_FILE = r"C:\Users\lzq\Downloads\chembl_37_chemreps.txt.gz"
OUTPUT_FILE = r"D:\E盘模型暂存\chembl_bioactivity_with_smiles.parquet"

PAGE_SIZE = 1000       # ChEMBL API 单页推荐大小
MAX_RETRIES = 5        # 单页最大重试次数
BASE_DELAY = 2         # 指数退避基础秒数
REQUEST_TIMEOUT = 30   # 单次请求超时(秒)
POLITE_DELAY = 0.3     # 每页之间的礼貌性延迟(秒)

# ✅ 直接使用 REST API 参数（不再依赖 chembl_webresource_client）
CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
ACTIVITY_PARAMS = {
    "target_organism": "Homo sapiens",
    "pchembl_value__isnull": "false",
    "assay_type": "B",
    "standard_relation": "=",
    "type__in": "IC50,Ki,Kd",       # REST API 用逗号分隔多值
    "only_fields": "molecule_chembl_id,target_chembl_id,pchembl_value,standard_type",
    "limit": PAGE_SIZE,              # ✅ 显式指定，服务端必定遵守
}
LOCAL_FIELDS = ["molecule_chembl_id", "target_chembl_id", "pchembl_value", "standard_type"]


# ==================== 辅助函数：安全删除文件 ====================
def safe_remove(filepath, max_tries=3):
    for i in range(max_tries):
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"   🗑️ 已成功删除: {filepath}", flush=True)
            return True
        except PermissionError:
            print(f"   ⚠️ 文件被占用，尝试释放 ({i + 1}/{max_tries})...", flush=True)
            gc.collect()
            time.sleep(1)
        except Exception as e:
            print(f"   ❌ 删除失败: {e}", flush=True)
            return False
    print(f"   🚫 无法删除 {filepath}，请手动关闭占用该文件的程序后重试！", flush=True)
    return False


# ==================== 1. 注入全局请求超时 ====================
_original_get = requests.Session.get
_original_post = requests.Session.post

def _patched_get(self, url, **kwargs):
    kwargs.setdefault('timeout', REQUEST_TIMEOUT)
    return _original_get(self, url, **kwargs)

def _patched_post(self, url, **kwargs):
    kwargs.setdefault('timeout', REQUEST_TIMEOUT)
    return _original_post(self, url, **kwargs)

requests.Session.get = _patched_get
requests.Session.post = _patched_post
print("✅ 已注入全局请求超时保护 (%ds)" % REQUEST_TIMEOUT, flush=True)


# ==================== 2. 直接调用 REST API 获取单页数据 ====================
session = requests.Session()
session.headers.update({"Accept": "application/json"})


def fetch_page_with_retry(offset, page_size):
    """
    ✅ 彻底绕过 chembl_webresource_client，直接调用 REST API
    确保 limit 参数被服务端正确识别，每页稳定返回 page_size 条
    """
    params = {**ACTIVITY_PARAMS, "offset": offset, "limit": page_size}

    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(CHEMBL_API_BASE, params=params)

            if resp.status_code == 429:
                retry_after = resp.headers.get('Retry-After')
                wait = int(retry_after) if retry_after else BASE_DELAY * (2 ** attempt)
                print(f"   ⏳ 429限流 | offset={offset} | 等待{wait:.1f}s | "
                      f"尝试{attempt + 1}/{MAX_RETRIES}", flush=True)
                if attempt == MAX_RETRIES - 1:
                    raise RuntimeError(f"连续{MAX_RETRIES}次429限流, offset={offset}")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            records = data.get("activities", [])
            return records

        except requests.exceptions.RequestException as e:
            err_msg = str(e).lower()
            is_last = (attempt == MAX_RETRIES - 1)

            if 'timeout' in err_msg or 'connection' in err_msg:
                wait = BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                tag = "🔌 网络异常"
            else:
                wait = BASE_DELAY * (2 ** attempt)
                tag = "❌ 请求错误"

            print(f"   {tag}: {type(e).__name__}: {e} | offset={offset} | "
                  f"等待{wait:.1f}s | 尝试{attempt + 1}/{MAX_RETRIES}", flush=True)

            if is_last:
                raise RuntimeError(f"连续{MAX_RETRIES}次失败, offset={offset}") from e
            time.sleep(wait)

    return None


# ==================== 3. Step 1: 活性数据提取 ====================
def extract_activities():
    offset = 0
    total_fetched = 0
    writer = None

    # --- 缓存校验与安全删除 ---
    cache_valid = False
    if os.path.exists(ACTIVITY_CACHE):
        file_size = os.path.getsize(ACTIVITY_CACHE)
        if file_size == 0:
            print(f"⚠️ 缓存文件为空 (0 bytes)，将删除并重新提取", flush=True)
            if not safe_remove(ACTIVITY_CACHE):
                raise PermissionError(f"无法删除空缓存文件 {ACTIVITY_CACHE}，请手动删除后重试")
        else:
            try:
                pf = pq.ParquetFile(ACTIVITY_CACHE)
                n_rows = pf.metadata.num_rows
                if n_rows > 0:
                    cache_valid = True
                    print(f"📂 缓存文件有效 ({n_rows:,} 行)", flush=True)
                else:
                    print(f"⚠️ 缓存文件无数据行，将删除并重新提取", flush=True)
                    safe_remove(ACTIVITY_CACHE)
            except Exception as e:
                print(f"⚠️ 缓存文件损坏: {e}", flush=True)
                safe_remove(ACTIVITY_CACHE)

    if cache_valid and os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            state = json.load(f)
        offset = state['offset']
        total_fetched = state['total_fetched']
        print(f"🔄 断点续传: 从 offset={offset} 继续 (已获取 {total_fetched:,} 条)", flush=True)
    elif cache_valid:
        print(f"📂 活性数据缓存完整，跳过提取", flush=True)
        return pd.read_parquet(ACTIVITY_CACHE)
    else:
        if os.path.exists(PROGRESS_FILE):
            safe_remove(PROGRESS_FILE)
        print("🌐 首次提取活性数据 (REST API 直连模式)...", flush=True)

    # --- 分页提取 ---
    start_time = time.time()
    try:
        while True:
            page_start = time.time()

            records = fetch_page_with_retry(offset, PAGE_SIZE)
            if not records:
                break

            actual_count = len(records)
            df_page = pd.DataFrame.from_records(records)

            # 只保留需要的字段（防御性过滤）
            df_page = df_page[[c for c in LOCAL_FIELDS if c in df_page.columns]]

            table = pa.Table.from_pandas(df_page, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(ACTIVITY_CACHE, table.schema)
            writer.write_table(table)

            total_fetched += actual_count
            offset += PAGE_SIZE

            # 原子写入进度
            tmp_progress = PROGRESS_FILE + ".tmp"
            with open(tmp_progress, 'w') as f:
                json.dump({'offset': offset, 'total_fetched': total_fetched}, f)
            os.replace(tmp_progress, PROGRESS_FILE)

            elapsed = time.time() - start_time
            speed = total_fetched / elapsed if elapsed > 0 else 0
            page_cost = time.time() - page_start

            print(
                f"   📥 offset={offset:>7,} | "
                f"本页{actual_count:>4}条({page_cost:.1f}s) | "
                f"累计{total_fetched:>9,}条 | "
                f"速率{speed:>6.0f}条/s",
                flush=True
            )

            if actual_count < PAGE_SIZE:
                print(f"   ℹ️ 本页返回 {actual_count} 条 < {PAGE_SIZE}，已到达数据集末尾", flush=True)
                break  # ✅ 真正到达末尾时立即退出，不再发无效请求

            time.sleep(POLITE_DELAY)

    finally:
        if writer is not None:
            writer.close()

    print(f"\n✅ 活性数据提取完成: {total_fetched:,} 条\n", flush=True)
    return pd.read_parquet(ACTIVITY_CACHE)


# ==================== 4. Step 2: 本地 chemreps 合并 SMILES ====================
def merge_smiles(df_act):
    print("📂 加载本地 chemreps 文件...", flush=True)
    df_chemreps = pd.read_csv(
        CHEMREPS_FILE, sep="\t", compression="gzip",
        usecols=["chembl_id", "canonical_smiles"], dtype={"chembl_id": "str"}
    )
    df_chemreps.rename(columns={
        "chembl_id": "molecule_chembl_id",
        "canonical_smiles": "smiles"
    }, inplace=True)
    print(f"   chemreps 共 {len(df_chemreps):,} 条记录", flush=True)

    df_final = df_act.merge(df_chemreps, on="molecule_chembl_id", how="inner")
    n_unique = df_final['molecule_chembl_id'].nunique()
    print(f"\n🎉 最终数据集: {len(df_final):,} 条记录, {n_unique:,} 个唯一化合物", flush=True)

    df_final.to_parquet(OUTPUT_FILE, index=False)
    print(f"💾 已保存至 {OUTPUT_FILE}", flush=True)
    return df_final


# ==================== 5. 主入口 ====================
if __name__ == "__main__":
    df_activities = extract_activities()
    df_result = merge_smiles(df_activities)