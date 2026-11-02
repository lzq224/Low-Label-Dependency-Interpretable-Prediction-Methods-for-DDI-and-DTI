import pandas as pd

# 1. 读取原始文件
# 注意：由于您的文件是制表符分隔的，使用 sep='\t'
df = pd.read_csv('药物及对应靶点plus，靶点未去重.txt', sep='\t')

# 2. 定义一个函数来处理单个基因字符串的去重
def deduplicate_genes(gene_string):
    """
    接收一个以分号分隔的基因字符串，返回去重后的字符串。
    """
    # 按分号分割字符串得到基因列表
    gene_list = gene_string.split(';')
    # 使用 set() 进行去重，然后转换回列表
    unique_gene_list = list(set(gene_list))
    # 将去重后的列表按分号重新连接成字符串
    return ';'.join(unique_gene_list)

# 3. 对 'Target_Genes' 列应用去重函数
# 注意：在处理前先检查是否为缺失值 (NaN)，避免报错
df['Target_Genes'] = df['Target_Genes'].apply(
    lambda x: deduplicate_genes(x) if pd.notna(x) else x
)

# 4. 将处理后的数据框保存到新文件
# index=False 表示不将行索引写入文件
df.to_csv('药物及对应靶点plus，靶点唯一.txt', sep='\t', index=False)

print("去重完成！结果已保存到 '药物及对应靶点plus，靶点唯一.txt'")