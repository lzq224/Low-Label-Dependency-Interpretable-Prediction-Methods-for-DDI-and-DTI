
Low-Label-Dependency Interpretable Prediction Methods for DTI and DDI
低标注依赖的DTI与DDI可解释预测方法
本仓库提供学位论文《低标注依赖的DTI与DDI可解释预测方法研究》相关的核心代码与数据说明。
This repository provides the core code and data notes for the thesis “Research on Interpretable Prediction Methods for DTI and DDI with Low Label Dependency”.

Overview / 项目简介
本研究提出一种低标注依赖的 DTI 与 DDI 可解释预测框架。方法不依赖端到端监督训练，而是分别从"以药找靶"和"以靶点集合为中介"两条互补路径展开：

DTI 预测：基于分子描述符与 FAISS 向量索引进行相似药物检索与候选靶点补全。
DDI 预测：基于双药靶点组合的正 Z-score 累计结果对互作类型排序，并经 Softmax 归一化输出 Top-5 互作类型预测。
This work proposes an interpretable DTI/DDI prediction framework under low label dependency. Instead of end-to-end supervised training, it adopts two complementary paths: drug retrieval plus candidate target completion (DTI), and DDI-type ranking by positive Z-score accumulation over dual-drug target sets (DDI).

Directory Structure / 目录结构
.
|-- README.md
|-- requirements.txt
|-- dataREADME.txt
|-- dti/                      # DTI 侧代码（12 个脚本）
|-- ddi/                      # DDI 侧代码（10 个脚本）
|-- results/                  # 论文关键结果 CSV（可直接核对 5.1 / 5.2 节指标）
|-- checkpoints/              # 可选：预存模型权重 / 运行结果
Environment / 运行环境
Python 3.8+，Windows / Linux 均可
支持纯 CPU 运行，无需 GPU / CPU-only environment, GPU is not required
安装依赖：

pip install -r requirements.txt
Code Modules / 代码模块清单
DTI 侧（dti/，对应论文第3章流水线）
脚本	作用
filter_and_standardize_bindingdb_smiles_initial.py	BindingDB 大分子剔除与初次标准化
extract_bindingdb_uniprot_ids.py	抽取唯一 UniProt ID
annotate_bindingdb_gene_symbol.py	回填 Gene Symbol
fetch_chembl37_by_api.py	通过 REST API 抓取 ChEMBL37
filter_and_standardize_chembl_smiles_initial.py	ChEMBL 剔除与初次标准化
extract_chembl_uniprot_ids.py	抽取 ChEMBL 唯一 UniProt ID
annotate_chembl_gene_symbol.py	ChEMBL 回填 Gene Symbol
merge_and_normalize_dti_sources_secondary.py	双源合并去重
compute_dti_descriptors.py	计算 36 维分子描述符
inspect_dti_dataset.py	数据集结构探查统计
faiss_molecular_target_search.py	"以药找靶"检索与靶点预测
faiss_search_self_exclusion_validation.py	自排除回查性能验证
DDI 侧（ddi/，对应论文第4章流水线）
脚本	作用
extract_ddi_and_targets.py	流式解析 DrugBank XML
deduplicate_targets.py	药物唯一靶点表去重
split_and_pair_targets.py	靶点组合拆分与配对
count_label_frequency.py	三元组频率统计
target_pair_zscore.py	频次转 Z-score 降维
pair_classification_probability.py	构建知识库 + 批量预测评估
deduplicate_triplets.py	药物组合去重
predict_ddi_by_targets.py	交互式 Top-5 预测器
batch_predict_ddi.py	批量预测与性能验证
inspect_unpredictable_cases.py	提取无法预测样本
Quick Start / 快速开始
1. Prepare data / 准备数据
将所需数据放入 data/ 目录（原始数据来源见 dataREADME.txt）。
Place the datasets under data/ (see dataREADME.txt for sources).

2. Run DTI prediction / 运行 DTI 预测
按 dti/ 中脚本的论文顺序运行，最终执行 faiss_molecular_target_search.py 输出候选靶点结果。

3. Run DDI prediction / 运行 DDI 预测
按 ddi/ 中脚本的论文顺序运行，最终执行 predict_ddi_by_targets.py / batch_predict_ddi.py 输出 Top-5 互作类型预测。

主要输出包括 DTI 候选结果、DDI 互作类型排序以及 Top-5 预测结果。

Data Sources / 数据来源
由于体积限制（GitHub 单文件上限 100MB），GB 级原始数据未直接附带，请按下列来源获取后放入 data/。完整说明见 dataREADME.txt。

DTI data / DTI 数据
Kaggle 数据集：BindingDB for Drug-Target Interaction
https://www.kaggle.com/datasets/rajaryan2315/bindingdb-for-drug-target-interaction
原始数据库：BindingDB (UCSD) - https://www.bindingdb.org/rwd/bind/index.jsp
ChEMBL37：由 fetch_chembl37_by_api.py 通过官方 REST API（https://www.ebi.ac.uk/chembl/）在线抓取。
数据结构 / Dataset Schema（供链接失效时核对）
若 Kaggle 链接日后失效，可依据以下结构与清洗规则确认获取到的数据与论文所用一致。
If the Kaggle link becomes unavailable, the schema below lets you verify that the data you obtain matches the one used in this thesis.

文件名 / File name：bindingdb_highconf_pKd_clean.csv

内容说明 / Description：BindingDB 经过整理与清洗（curated and cleaned）的版本，面向 DTI 预测任务，包含化合物 SMILES 字符串、蛋白序列以及对应的结合亲和力（pKd 值）。
A curated and cleaned version of BindingDB focused on DTI prediction, containing compound SMILES strings, protein sequences, and their corresponding binding affinities (pKd values).

列 / Columns：

列名 Column	含义 Meaning
compound_iso_smiles	化合物 SMILES 字符串 / compound SMILES string
target_sequence	靶点蛋白序列 / target protein sequence
affinity	结合亲和力（pKd 值）/ binding affinity (pKd value)
清洗规则 / Cleaning rule：重复条目、不完整条目，以及缺少 SMILES 或没有有效序列靶点的化合物均已删除。
Duplicated and incomplete entries, as well as compounds missing SMILES or a valid sequence target, have been removed.

数据准备 / Data Preparation:

从上述链接下载 bindingdb_highconf_pKd_clean.csv，放入 data/dti/。
Download bindingdb_highconf_pKd_clean.csv and place it under data/dti/.
ChEMBL37 由脚本自动抓取。
按论文顺序运行 dti/ 预处理脚本。
DDI data / DDI 数据
DDI 与靶点信息提取自 DrugBank 官方全量 XML（drugbank_all_full_database.xml），需注册后从 https://go.drugbank.com/releases 下载，放入 data/ddi/，再由 extract_ddi_and_targets.py 流式解析。

Results / 结果文件
results/ 目录已附带论文代表性结果 CSV（如 recommended_targets_summary.csv、drug_target_edges.csv、DTI 自排除回查指标、DDI Top-5 命中率等），可直接核对论文 5.1、5.2 节图表数据。

Notes / 说明
不同运行环境、随机种子或依赖版本可能导致结果存在轻微差异。
受硬件条件限制，本实现面向 CPU 并行执行，可在脚本中调整并行进程数。
原始开发环境中的绝对路径已替换为相对路径。
FAISS 索引与指纹缓存（如 fp_cache.pkl）会在首次运行时自动生成，无需手动下载。
Citation / 引用
如果使用本仓库中的代码或数据，请引用对应学位论文：

@mastersthesis{DTI_DDI_2026,
  title  = {低标注依赖的DTI与DDI可解释预测方法研究},
  author = {Your Name},
  school = {Your University},
  year   = {2026}
}
License / 许可
本仓库代码与数据仅用于学术研究。MIT License.
