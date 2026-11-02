import glob
from collections import Counter


def count_duplicate_combinations(pattern="temp_indep_match_results_chunk_*_resumable.txt"):
    """
    统计所有临时结果文件中，node1, node2, tag 三元组的出现频率。
    """
    print(f"正在扫描匹配模式 '{pattern}' 的文件...")
    files = glob.glob(pattern)

    if not files:
        print(f"错误：没有找到符合模式 '{pattern}' 的文件。")
        return

    print(f"找到 {len(files)} 个文件，开始统计...")

    # 使用 Counter 来高效地统计三元组的出现次数
    combination_counter = Counter()

    for file_path in files:
        print(f"  -> 正在处理文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline()  # 读取并跳过header行
            for line_num, line in enumerate(f, start=2):  # 从第2行开始计数
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    print(f"警告: 文件 {file_path} 第 {line_num} 行格式不正确，已跳过。")
                    continue

                # 取出前三列作为元组，并排序 node1 和 node2 以确保一致性
                # 例如，(A, B, C) 和 (B, A, C) 应该被视为同一个组合
                node1, node2, tag = parts[0], parts[1], parts[2]
                # 创建一个标准化的键，确保基因对的顺序不影响统计
                key = (min(node1, node2), max(node1, node2), tag)

                combination_counter[key] += 1

    print(f"\n统计完成！共发现 {len(combination_counter)} 个唯一的三元组组合。")

    # 将结果写入新文件
    output_filename = "重复三元组统计结果.txt"
    print(f"正在将结果写入文件: {output_filename}")

    with open(output_filename, 'w', encoding='utf-8') as f_out:
        f_out.write("node1\tnode2\ttag\tcount\n")  # 写入header
        # 按照出现次数降序排列
        for (node1, node2, tag), count in combination_counter.most_common():
            f_out.write(f"{node1}\t{node2}\t{tag}\t{count}\n")

    print(f"结果已保存至: {output_filename}")

    # 打印出现频率最高的几个组合作为示例
    print("\n出现频率最高的前10个组合:")
    print("node1\tnode2\ttag\t\t\t\t\tcount")
    print("-" * 60)
    for (node1, node2, tag), count in combination_counter.most_common(10):
        print(f"{node1}\t{node2}\t{tagf[:30]}...\t{count}")


if __name__ == "__main__":
    count_duplicate_combinations()