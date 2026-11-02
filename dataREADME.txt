
================================================================================
Data Sources
数据来源说明
================================================================================

This document details the raw data sources used in this project. Due to file 
size limits or licensing restrictions, some large files are not included 
directly in the repository.

本文档详细列出了本项目使用的原始数据来源。受限于文件大小或许可协议，
部分大文件未直接包含在仓库中。

--------------------------------------------------------------------------------
1. DTI Data (BindingDB)
   DTI 数据（BindingDB）
--------------------------------------------------------------------------------

- Source: Kaggle Dataset "BindingDB High Confidence pKd"
  来源：Kaggle 数据集 "BindingDB High Confidence pKd"
- Direct Link: https://www.kaggle.com/datasets/vinayak123tyagi/bindingdb-high-confidence-pkd
- File Name: bindingdb_highconf_pKd_clean.csv

- Description / 描述:
  This dataset contains a curated and cleaned version of BindingDB, focused on 
  drug-target interaction (DTI) prediction tasks. It includes compound SMILES 
  strings, protein sequences, and their corresponding binding affinities (pKd values).
  该数据集包含 BindingDB 的整理清洗版本，专注于药物-靶点相互作用（DTI）预测任务。
  包括化合物 SMILES 字符串、蛋白质序列及其对应的结合亲和力（pKd 值）。

- Schema / 数据结构:
  | Column Name          | Description (CN) | Description (EN)             |
  |----------------------|------------------|------------------------------|
  | compound_iso_smiles  | 化合物 SMILES    | Compound SMILES string       |
  | target_sequence      | 靶点蛋白序列     | Target protein sequence      |
  | affinity             | 亲和力 (pKd)     | Binding affinity (pKd value) |

- Cleaning Rules / 清洗规则:
  Duplicates, incomplete entries, compounds missing SMILES, or targets without 
  valid sequences have been removed.
  已删除重复项、不完整条目、缺少 SMILES 的化合物或无有效序列的靶点。

--------------------------------------------------------------------------------
2. DDI Data (DrugBank & ChEMBL37)
   DDI 数据（DrugBank 与 ChEMBL37）
--------------------------------------------------------------------------------

A. DrugBank (Target Information)
   DrugBank（靶点信息）
   - Source: DrugBank Online (Full Database XML)
     来源：DrugBank Online（全量数据库 XML）
   - Access: Requires academic/free account registration.
     获取方式：需注册学术/免费账户。
   - Usage: Extracted target sequences and drug-target links to build the 
     intermediate target set.
     用途：提取靶点序列和药物-靶点链接以构建中介靶点集合。

B. ChEMBL37 (Interaction Labels)
   ChEMBL37（相互作用标签）
   - Source: ChEMBL37 REST API / FTP
     来源：ChEMBL37 REST API / FTP
   - Script: fetch_chembl_ddi.py (provided in scripts/)
     脚本：fetch_chembl_ddi.py（位于 scripts/ 目录中）
   - Description: Raw drug-drug interaction pairs fetched via API.
     描述：通过 API 获取的原始药物-药物相互作用对。

C. Manual Curation Evidence (Excel Logs)
   人工清洗证据（Excel 记录）
   - Location: GitHub Releases (Assets)
     位置：GitHub Releases（附件资源）
   - Files:
       flat1-868736.xlsx
       flat2-868736.xlsx
       flat3-868737.xlsx
   - Description: These three Excel files document the manual cleaning process 
     of the DDI labels. They record the step-by-step verification, filtering, 
     and correction of interaction pairs derived from ChEMBL and DrugBank.
     Due to file size limits (>25MB), they are attached to the v1.0 Release 
     instead of the main repository.
     描述：这三个 Excel 文件记录了 DDI 标签的人工清洗过程，详细记载了从 ChEMBL 
     和 DrugBank 衍生的相互作用对的逐步验证、过滤和修正步骤。由于文件大小超过
     限制（>25MB），它们被附在 v1.0 Release 中而非主仓库内。

D. Final Processed DDI Dataset
   最终处理后的 DDI 数据集
   - Original Name: drug_pairs_targets_interactions-未去重.txt
     原始文件名：drug_pairs_targets_interactions-未去重.txt
   - Location: GitHub Releases (Assets)
     位置：GitHub Releases（附件资源）
   - Note: The original text file exceeded the 25MB upload limit for individual 
     files on GitHub. It has been compressed and uploaded as an Asset in the 
     v1.0 Release. Please download it from the "Releases" section on the right 
     sidebar of the repository page.
     注意：原始文本文件超过了 GitHub 单文件 25MB 的上传限制，已压缩后作为 
     v1.0 Release 的资源附件上传，请从仓库页面右侧边栏的 "Releases" 部分下载。

--------------------------------------------------------------------------------
3. Results
   结果文件
--------------------------------------------------------------------------------

- All final result tables and figures are included in the results/ folder 
  within the repository.
  所有最终结果表格和图表均已包含在仓库的 results/ 文件夹中。

================================================================================
End of Document
文档结束
================================================================================
