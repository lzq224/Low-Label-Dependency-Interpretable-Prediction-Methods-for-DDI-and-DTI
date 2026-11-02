================================================================================
Data Sources and Reproduction Guide
数据来源与复现指南
================================================================================

This document details the raw data sources used in this project and provides 
instructions for reproducing the datasets. Due to file size limits or licensing 
restrictions, some large files are not included directly in the repository.

本文档详细列出了本项目使用的原始数据来源及复现方法。受限于文件大小或许可协议，
部分大文件未直接包含在仓库中。

--------------------------------------------------------------------------------
1. DTI Data (BindingDB)
   DTI 数据（BindingDB）
--------------------------------------------------------------------------------

- Source: Kaggle Dataset "BindingDB High Confidence pKd"
  来源：Kaggle 数据集 "BindingDB High Confidence pKd"
  
- Direct Link: 
  https://www.kaggle.com/datasets/your-kaggle-dataset-link (Please replace with actual link)
  
- File Name: `bindingdb_highconf_pKd_clean.csv`
  文件名：`bindingdb_highconf_pKd_clean.csv`

- Description / 描述:
  This dataset contains a curated and cleaned version of BindingDB, focused on 
  drug-target interaction (DTI) prediction tasks. It includes compound SMILES 
  strings, protein sequences, and their corresponding binding affinities (pKd values).
  该数据集包含 BindingDB 的整理清洗版本，专注于药物-靶点相互作用（DTI）预测任务。
  包括化合物 SMILES 字符串、蛋白质序列及其对应的结合亲和力（pKd 值）。

- Schema / 数据结构:

  | Column Name          | Description (CN)       | Description (EN)                  |
  |----------------------|------------------------|-----------------------------------|
  | compound_iso_smiles  | 化合物 SMILES          | Compound SMILES string            |
  | target_sequence      | 靶点蛋白序列           | Target protein sequence           |
  | affinity             | 亲和力 (pKd)           | Binding affinity (pKd value)      |

- Cleaning Rules / 清洗规则:
  Duplicates, incomplete entries, compounds missing SMILES, or targets without 
  valid sequences have been removed.
  已删除重复项、不完整条目、缺少 SMILES 的化合物或无有效序列的靶点。

--------------------------------------------------------------------------------
2. DDI Data (DrugBank & Interactions)
   DDI 数据（DrugBank 与相互作用）
--------------------------------------------------------------------------------

- Source A: DrugBank Database
  来源 A：DrugBank 数据库
  - URL: https://go.drugbank.com/releases/latest
  - Note: Requires free registration to download the full `all_drugbank_vocabulary.xml`.
    注意：需免费注册方可下载全量 XML 文件。

- Source B: ChEMBL 37
  来源 B：ChEMBL 37
  - URL: https://www.ebi.ac.uk/chembl/
  - Access: Retrieved via official REST API using provided scripts.
    获取方式：通过提供的脚本调用官方 REST API 获取。

- Large Interaction File / 大型相互作用文件:
  - File Name: `drug_pairs_targets_interactions-未去重.txt`
  - Location: **GitHub Releases (v1.0)**
    位置：**GitHub Releases (v1.0)**
  - Reason: The compressed file exceeds GitHub's 25MB single-file upload limit.
    原因：压缩后文件仍超过 GitHub 25MB 单文件上传限制。
  - Download: Please go to the "Releases" section on the right sidebar of the 
    repository homepage to download the asset `drug_pairs_targets_interactions-.zip`.
    下载：请前往仓库主页右侧栏的 "Releases" 板块下载资源包 `drug_pairs_targets_interactions-.zip`。

--------------------------------------------------------------------------------
3. Reproduction Steps
   复现步骤
--------------------------------------------------------------------------------

1. Download `bindingdb_highconf_pKd_clean.csv` from Kaggle and place it in `data/raw_dti/`.
   从 Kaggle 下载 DTI 数据并放入 `data/raw_dti/`。

2. Register at DrugBank, download the XML, and place it in `data/raw_ddi/`.
   注册 DrugBank，下载 XML 文件并放入 `data/raw_ddi/`。

3. Run the ChEMBL scraping scripts to generate intermediate target sets.
   运行 ChEMBL 抓取脚本生成中间靶点集合。

4. Download the large interaction zip from Releases, extract it, and place the txt file in `data/raw_ddi/`.
   从 Releases 下载大型相互作用压缩包，解压后将 txt 文件放入 `data/raw_ddi/`。

5. Follow the "Quick Start" in README.md to run DTI and DDI pipelines.
   按照 README.md 中的“快速开始”运行 DTI 和 DDI 流程。
