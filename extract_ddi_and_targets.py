import pandas as pd
from lxml import etree as ET
import os
import time


def parse_drugbank_split_output(xml_file):
    """
    生产版本 v3.0 (双文件输出):
    1. 主文件: 基础信息 + 基因列表 (无相互作用长文本)
    2. 交互文件: 药物相互作用 (一行一条记录，扁平化)
    3. 流式解析，内存安全
    """
    print(f"🚀 开始解析全量文件 (双输出模式)：{xml_file}")
    start_time = time.time()

    NS_URI = "http://www.drugbank.ca"
    ns = {'db': NS_URI}
    drug_tag = f'{{{NS_URI}}}drug'

    # 两个列表分别存储数据
    basic_results = []
    interaction_results = []

    count_total = 0
    count_small_mol = 0
    count_interactions = 0

    context = ET.iterparse(xml_file, events=('end',), tag=drug_tag)

    for event, elem in context:
        count_total += 1

        # 1. 快速过滤：只处理 small molecule
        drug_type = elem.get('type')
        if not drug_type or drug_type.strip().lower() != 'small molecule':
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
            continue

        count_small_mol += 1

        # 2. 提取基础信息
        name_elem = elem.find('db:name', namespaces=ns)
        drug_name = name_elem.text if name_elem is not None else 'N/A'

        id_elem = elem.find('db:drugbank-id[@primary="true"]', namespaces=ns)
        drug_id = id_elem.text if id_elem is not None else 'N/A'

        smiles = 'N/A'
        props = elem.find('db:calculated-properties', namespaces=ns)
        if props is not None:
            for prop in props.findall('db:property', namespaces=ns):
                kind = prop.find('db:kind', namespaces=ns)
                if kind is not None and kind.text == 'SMILES':
                    val = prop.find('db:value', namespaces=ns)
                    if val is not None:
                        smiles = val.text
                        break

        # --- 辅助函数：提取基因 ---
        def extract_genes_from_items(items):
            genes = set()
            for item in items:
                peptides = item.xpath('db:polypeptide', namespaces=ns)
                if not peptides: peptides = item.xpath('polypeptide')
                for pep in peptides:
                    g_nodes = pep.xpath('db:gene-name', namespaces=ns)
                    if not g_nodes: g_nodes = pep.xpath('gene-name')
                    for g in g_nodes:
                        if g.text: genes.add(g.text.strip())

                g_nodes_direct = item.xpath('db:gene-name', namespaces=ns)
                if not g_nodes_direct: g_nodes_direct = item.xpath('gene-name')
                for g in g_nodes_direct:
                    if g.text: genes.add(g.text.strip())
            return genes

        # 3. 提取 Enzymes
        enzyme_genes = set()
        enzymes_container = elem.xpath('db:enzymes', namespaces=ns)
        if not enzymes_container: enzymes_container = elem.xpath('enzymes')
        for cont in enzymes_container:
            items = cont.xpath('db:enzyme', namespaces=ns) or cont.xpath('enzyme')
            enzyme_genes.update(extract_genes_from_items(items))

        reactions = elem.xpath('db:reactions', namespaces=ns)
        if not reactions: reactions = elem.xpath('reactions')
        for reac_cont in reactions:
            items = reac_cont.xpath('.//db:enzyme', namespaces=ns) or reac_cont.xpath('.//enzyme')
            enzyme_genes.update(extract_genes_from_items(items))

        # 4. 提取 Targets
        target_genes = set()
        targets_container = elem.xpath('db:targets', namespaces=ns)
        if not targets_container: targets_container = elem.xpath('targets')
        for cont in targets_container:
            items = cont.xpath('db:target', namespaces=ns) or cont.xpath('target')
            target_genes.update(extract_genes_from_items(items))

        # 5. 提取 Transporters
        trans_genes = set()
        trans_container = elem.xpath('db:transporters', namespaces=ns)
        if not trans_container: trans_container = elem.xpath('transporters')
        for cont in trans_container:
            items = cont.xpath('db:transporter', namespaces=ns) or cont.xpath('transporter')
            trans_genes.update(extract_genes_from_items(items))

        # --- 存入主文件数据 ---
        basic_results.append({
            'DrugBank_ID': drug_id,
            'Name': drug_name,
            'SMILES': smiles,
            'Reaction_Enzyme_Genes': '; '.join(list(enzyme_genes)),
            'Target_Genes': '; '.join(list(target_genes)),
            'Transporter_Genes': '; '.join(list(trans_genes))
        })

        # 6. 【新增】提取并扁平化 Drug Interactions
        interactions_container = elem.find('db:drug-interactions', namespaces=ns)
        if interactions_container is None:
            interactions_container = elem.find('drug-interactions')

        if interactions_container is not None:
            # 尝试带命名空间查找
            interaction_nodes = interactions_container.findall('db:drug-interaction', namespaces=ns)
            # 容错：如果没找到，尝试不带命名空间
            if not interaction_nodes:
                interaction_nodes = interactions_container.findall('drug-interaction')

            for interaction in interaction_nodes:
                # 提取相互作用药物名称
                i_name_elem = interaction.find('db:name', namespaces=ns)
                if i_name_elem is None: i_name_elem = interaction.find('name')
                i_name = i_name_elem.text.strip() if i_name_elem is not None and i_name_elem.text else 'N/A'

                # 提取描述
                i_desc_elem = interaction.find('db:description', namespaces=ns)
                if i_desc_elem is None: i_desc_elem = interaction.find('description')
                i_desc = i_desc_elem.text.strip() if i_desc_elem is not None and i_desc_elem.text else 'N/A'

                # 清理换行符，保证 CSV 整洁
                if i_desc != 'N/A':
                    i_desc = i_desc.replace('\n', ' ').replace('\r', ' ')

                # 添加到独立列表 (扁平化：一行一个相互作用)
                interaction_results.append({
                    'DrugBank_ID': drug_id,
                    'Drug_Name': drug_name,
                    'Interacting_Drug_Name': i_name,
                    'Interaction_Description': i_desc
                })
                count_interactions += 1

        # 进度汇报
        if count_small_mol % 5000 == 0:
            elapsed = time.time() - start_time
            speed = count_small_mol / elapsed if elapsed > 0 else 0
            print(
                f"⏳ 已处理 {count_small_mol:,} 个小分子... (发现 {count_interactions:,} 条相互作用) | 速度: {speed:.1f} 条/秒")

        # 内存清理
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    elapsed_total = time.time() - start_time
    print(f"\n✅ 解析完成！")
    print(f"   - 总扫描节点: {count_total:,}")
    print(f"   - 成功提取小分子: {count_small_mol:,}")
    print(f"   - 提取相互作用记录: {count_interactions:,}")
    print(f"   - 总耗时: {elapsed_total:.2f} 秒 ({elapsed_total / 60:.2f} 分钟)")

    # 返回两个 DataFrame
    basic_cols = ['DrugBank_ID', 'Name', 'SMILES', 'Reaction_Enzyme_Genes', 'Target_Genes', 'Transporter_Genes']
    inter_cols = ['DrugBank_ID', 'Drug_Name', 'Interacting_Drug_Name', 'Interaction_Description']

    df_basic = pd.DataFrame(basic_results, columns=basic_cols) if basic_results else pd.DataFrame(columns=basic_cols)
    df_inter = pd.DataFrame(interaction_results, columns=inter_cols) if interaction_results else pd.DataFrame(
        columns=inter_cols)

    return df_basic, df_inter


if __name__ == '__main__':
    # ================= 配置区域 =================
    xml_path = r'D:\下载\drugbank_all_full_database.xml\full database.xml'
    output_basic_csv = 'drugbank_full_basic.csv'  # 主文件
    output_inter_csv = 'drugbank_interactions_flat.csv'  # 相互作用独立文件
    # ===========================================

    if not os.path.exists(xml_path):
        print(f"❌ 错误：文件未找到 {xml_path}")
    else:
        file_size_gb = os.path.getsize(xml_path) / (1024 ** 3)
        print(f"📁 检测到文件大小: {file_size_gb:.2f} GB")
        print("⏳ 开始处理全量数据 (双文件输出)，请稍候...\n")

        df_basic, df_inter = parse_drugbank_split_output(xml_path)

        # --- 保存主文件 ---
        if not df_basic.empty:
            print(f"\n💾 正在保存主文件 ({output_basic_csv})...")
            df_basic.to_csv(output_basic_csv, index=False, encoding='utf-8-sig')
            print(f"✅ 主文件保存成功: {len(df_basic):,} 行")

            print("\n📊 [主文件] 统计:")
            total_enz = (df_basic['Reaction_Enzyme_Genes'].str.len() > 0).sum()
            total_tar = (df_basic['Target_Genes'].str.len() > 0).sum()
            total_tra = (df_basic['Transporter_Genes'].str.len() > 0).sum()
            print(f"   - 含酶基因: {total_enz:,}")
            print(f"   - 含靶点基因: {total_tar:,}")
            print(f"   - 含转运体基因: {total_tra:,}")
        else:
            print("\n❌ 主文件未提取到数据。")

        # --- 保存相互作用文件 ---
        if not df_inter.empty:
            print(f"\n💾 正在保存相互作用文件 ({output_inter_csv})...")
            df_inter.to_csv(output_inter_csv, index=False, encoding='utf-8-sig')
            print(f"✅ 相互作用文件保存成功: {len(df_inter):,} 行")

            print("\n📊 [相互作用文件] 统计:")
            # 统计有多少个药物至少有一个相互作用
            drugs_with_inter = df_inter['DrugBank_ID'].nunique()
            print(f"   - 涉及药物数量: {drugs_with_inter:,}")
            print(f"   - 平均每个药物相互作用数: {len(df_inter) / drugs_with_inter:.2f}")

            # 找出相互作用最多的药物
            top_drug = df_inter['Drug_Name'].value_counts().head(1)
            print(f"   - 相互作用最多的药物: {top_drug.index[0]} ({top_drug.values[0]} 条)")

            print("\n📄 [相互作用文件] 前 5 行预览:")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            pd.set_option('display.max_colwidth', 80)
            print(df_inter.head(5).to_string())
        else:
            print("\n⚠️ 未提取到任何相互作用数据。")

        print("\n🎉 全部处理完成！")