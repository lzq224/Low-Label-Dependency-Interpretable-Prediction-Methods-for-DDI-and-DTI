def remove_duplicate_drug_pairs(input_file, output_file):
    """
    去除药物对数据中的重复项，drug1和drug2位置互换视为相同记录
    """
    seen_combinations = set()
    unique_lines = []

    with open(input_file, 'r', encoding='utf-8') as f:
        header = f.readline()  # 读取标题行
        unique_lines.append(header)

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                drug1, drug2 = parts[0], parts[1]

                # 创建标准化的键，确保drug1和drug2顺序不影响比较
                sorted_pair = tuple(sorted([drug1, drug2]))

                # 检查是否已存在此组合
                if sorted_pair not in seen_combinations:
                    seen_combinations.add(sorted_pair)
                    unique_lines.append(line)

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(unique_lines)

    print(f"原始数据行数: {sum(1 for _ in open(input_file, 'r', encoding='utf-8'))}")
    print(f"去重后行数: {len(unique_lines)}")
    print(f"结果已保存到: {output_file}")


# 使用示例
input_file = "drug_pairs_targets_interactions-未去重.txt"
output_file = "drug_pairs_targets_interactions-已去重.txt"

remove_duplicate_drug_pairs(input_file, output_file)