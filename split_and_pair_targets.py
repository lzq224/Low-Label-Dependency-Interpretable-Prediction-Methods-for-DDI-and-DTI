import pandas as pd
from multiprocessing import Pool, cpu_count
import os
import glob


def get_last_processed_line_index(status_file):
    """
    从状态文件中读取最后处理的行索引。
    """
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            try:
                last_idx = f.read().strip()
                return int(last_idx) if last_idx else -1
            except ValueError:
                return -1
    return -1


def save_current_index(status_file, current_index):
    """
    将当前处理的行索引保存到状态文件。
    """
    with open(status_file, 'w') as f:
        f.write(str(current_index))


def process_chunk_for_truly_resumable_matching(args):
    """
    处理单个块的任务函数，支持真正的断点续传。
    args: (chunk_data, chunk_id, tsv_file_path)
    """
    chunk_lines, chunk_id, tsv_file_path = args
    temp_output_file = f'temp_indep_match_results_chunk_{chunk_id}_resumable.txt'
    status_file = f'temp_status_chunk_{chunk_id}.txt'

    # --- 核心断点续传逻辑 ---
    last_processed_idx = get_last_processed_line_index(status_file)
    total_lines = len(chunk_lines)

    # 判断是否已完成
    if last_processed_idx >= total_lines - 1:
        print(f"  -> [块 {chunk_id}] 已完全处理完成，跳过。")
        return temp_output_file, chunk_id, True

    # 判断是从头开始还是从断点继续
    start_idx = max(0, last_processed_idx + 1)
    if start_idx > 0:
        print(f"  -> [块 {chunk_id}] 发现断点，将从内部行 {start_idx}/{total_lines} 继续处理。")
        file_mode = 'a'  # 追加模式
    else:
        print(f"  -> [块 {chunk_id}] 开始从头处理 {total_lines} 行数据...")
        file_mode = 'w'  # 写入模式，会清空文件

    # 每个进程独立加载一次预测数据
    print(f"  -> [块 {chunk_id}] 的子进程 {os.getpid()} 开始加载预测数据...")
    df_pred = pd.read_csv(tsv_file_path, sep='\t')

    pred_dict = {}
    for _, row in df_pred.iterrows():
        p_node1, p_node2 = str(row['node1']), str(row['node2'])
        nodes_tuple = tuple(sorted([p_node1, p_node2]))
        pred_dict[nodes_tuple] = (p_node1, p_node2)

    print(f"  -> [块 {chunk_id}] 的子进程 {os.getpid()} 预测数据加载完成。")

    matched_count = 0

    with open(temp_output_file, file_mode, encoding='utf-8') as f_temp:
        # 如果是从头开始，需要写入header
        if start_idx == 0:
            f_temp.write("node1\tnode2\tlag\toriginal_input_line\n")

        for i in range(start_idx, total_lines):
            line_content, original_line_num = chunk_lines[i]

            parts = line_content.split('\t')
            if len(parts) < 3:
                save_current_index(status_file, i)
                continue

            lag, target1_str, target2_str = parts[0], parts[1], parts[2]

            set2 = set(target1_str.split(';')) if target1_str else set()
            set3 = set(target2_str.split(';')) if target2_str else set()
            set4 = set2.union(set3)

            if not set2 or not set3:
                save_current_index(status_file, i)
                continue

            for (p_node1, p_node2), (orig_p_node1, orig_p_node2) in pred_dict.items():
                set1 = {p_node1, p_node2}

                if set1.issubset(set4):
                    if not set1.issubset(set2) and not set1.issubset(set3):
                        f_temp.write(f"{orig_p_node1}\t{orig_p_node2}\t{lag}\t{original_line_num}\n")
                        matched_count += 1

            if (i + 1) % 500 == 0:
                save_current_index(status_file, i)
                print(f"    -> [块 {chunk_id}] 已处理并保存断点至 {i + 1}/{total_lines} 行...")

        # 循环结束，更新状态为最终行号
        save_current_index(status_file, total_lines - 1)
        print(f"    -> [块 {chunk_id}] 所有行处理完毕，最终断点已更新。")

    print(f"  -> [块 {chunk_id}] 处理完成，本次新增找到 {matched_count} 个匹配项。")
    return temp_output_file, chunk_id, False


def find_and_write_truly_resumable_matches(
        predictions_file='predictions_part.tsv',
        targets_file='靶点未拆分.txt',
        output_file='匹配结果_可断点续传.txt',
        chunk_size=10000
):
    """
    支持真正断点续传的并行匹配。
    """
    num_processes = max(1, cpu_count() - 1)
    print(f"开始基于断点续传的并行匹配，使用 {num_processes} 个进程...")

    # 1. 不再清理文件，而是扫描以发现已完成的块
    print("正在扫描历史断点...")
    completed_chunks = set()
    for status_file in glob.glob('temp_status_chunk_*.txt'):
        try:
            chunk_id = int(status_file.split('_chunk_')[1].split('.')[0])
            # 检查对应的结果文件是否存在且不为空
            result_file = f'temp_indep_match_results_chunk_{chunk_id}_resumable.txt'
            if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
                # 检查状态是否真的完成了
                last_idx = get_last_processed_line_index(status_file)
                # 我们无法在这里知道该块的总行数，所以假设状态文件里的数字是准确的
                # 如果需要绝对精确，需要将总行数也写入状态文件，逻辑会更复杂
                # 这里简化处理，只要状态文件里记录的index很大，就认为完成了
                # 更严谨的做法是在创建块时也将总行数存下来
                # 为了简化，我们直接认为如果状态文件存在，我们就去检查它。
                completed_chunks.add(chunk_id)
                print(f"  -> 发现已标记的块: {chunk_id} (需进一步验证)")
        except (ValueError, IndexError):
            pass

    # 2. 读取目标文件并分割成块
    print(f"正在分割输入文件 {targets_file}...")
    chunks = []
    with open(targets_file, 'r', encoding='utf-8') as f_in:
        header = next(f_in)
        current_chunk = []
        current_line_num = 1
        for line in f_in:
            current_line_num += 1
            current_chunk.append((line.strip(), current_line_num))
            if len(current_chunk) >= chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
        if current_chunk:
            chunks.append(current_chunk)

    print(f"文件已分割为 {len(chunks)} 个块。")

    # 3. 过滤掉已完成的块
    remaining_chunks = [(chunks[i], i) for i in range(len(chunks)) if i not in completed_chunks]

    # 为了更精确地判断块是否完成，我们重新加载所有已完成的块来确认其状态文件
    truly_completed_chunks = set()
    for i in completed_chunks:
        if i < len(chunks):  # 防止索引越界，以防文件被修改
            total_lines_in_chunk = len(chunks[i])
            status_file = f'temp_status_chunk_{i}.txt'
            last_idx = get_last_processed_line_index(status_file)
            if last_idx >= total_lines_in_chunk - 1:
                truly_completed_chunks.add(i)
            else:
                # 状态不一致，视为未完成，加入待处理队列
                remaining_chunks.append((chunks[i], i))
                print(f"  -> 块 {i} 状态不一致，将重新处理。")

    print(f"扫描完成：{len(truly_completed_chunks)} 个块已完成，{len(remaining_chunks)} 个块待处理或需重新处理。")

    # 4. 创建任务参数列表
    tasks = [(chunk, chunk_id, predictions_file) for chunk, chunk_id in remaining_chunks]

    # 5. 启动进程池
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk_for_truly_resumable_matching, tasks)

    # 6. 统计结果
    total_matched_count = 0
    for temp_file, chunk_id, skipped in results:
        if os.path.exists(temp_file):
            with open(temp_file, 'r', encoding='utf-8') as f:
                total_matched_count += sum(1 for line in f) - 1

    # 7. 清理状态文件
    print("正在清理临时状态文件...")
    for status_file in glob.glob('temp_status_chunk_*.txt'):
        os.remove(status_file)

    print("\n" + "=" * 60)
    print(f"基于断点续传的并行匹配完成！")
    print(f"- 使用了 {num_processes} 个进程。")
    print(f"- 本次新增找到 {total_matched_count} 个符合条件的匹配项。")
    print("- 临时状态文件已清理。")
    print("=" * 60)


if __name__ == "__main__":
    find_and_write_truly_resumable_matches(
        predictions_file='predictions_part.tsv',
        targets_file='靶点未拆分.txt',
        output_file='匹配结果_可断点续传.txt',
        chunk_size=10000
    )