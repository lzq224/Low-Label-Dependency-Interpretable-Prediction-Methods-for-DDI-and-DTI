================================================================================
Data Sources
数据来源说明
================================================================================

This document details the raw data sources used in this project. Due to file 
size limits or licensing restrictions, some large files are not included 
directly in the repository and are provided as Release assets instead.

本文档详细列出了本项目使用的原始数据来源。受限于文件大小或许可协议，
部分大文件未直接包含在仓库中，而是作为 Release 附件提供。

--------------------------------------------------------------------------------
1. DTI Data (BindingDB)
   DTI 数据（BindingDB）
--------------------------------------------------------------------------------

- Source: Kaggle Dataset "BindingDB High Confidence pKd"
  来源：Kaggle 数据集 "BindingDB High Confidence pKd"
- Direct Link: https://www.kaggle.com/datasets/vinayak123tyagi/bindingdb-high-confidence-pkd
- File Name: bindingdb_highconf_pKd_clean.csv

- Description / 描述:
  Raw binding data between drugs and targets. This dataset serves as the 
  foundation for constructing the Drug-Target Interaction (DTI) graph.
  药物与靶点之间的原始结合数据。该数据集是构建药物-靶点相互作用（DTI）图的基础。

--------------------------------------------------------------------------------
2. DDI Data (DrugBank & Manual Curation)
   DDI 数据（DrugBank 与人工清洗）
--------------------------------------------------------------------------------

- Source: DrugBank Database & Manual Verification
  来源：DrugBank 数据库及人工验证
  
- Files in Release / Release 中的文件:
  - flat1-868736.xlsx (59.1 MB)
  - flat2-868736.xlsx (59.4 MB)
  - flat3-868737.xlsx (58.8 MB)

- Description / 描述:
  These three Excel files represent the raw and processed data for Drug-Drug 
  Interactions (DDI). They document the manual cleaning process of DDI labels, 
  verifying interactions extracted from DrugBank to ensure high-quality ground 
  truth labels for model training.
  这三个 Excel 文件代表了药物-药物相互作用（DDI）的原始及处理数据。它们记录了
  DDI 标签的人工清洗过程，即对从 DrugBank 提取的相互作用进行人工验证，以确
  保模型训练拥有高质量的真值标签。

--------------------------------------------------------------------------------
3. Performance Validation Results
   性能验证结果
--------------------------------------------------------------------------------

- File in Release / Release 中的文件:
  - drug_pair_predictions.csv (239.24 MB)

- Description / 描述:
  This file contains the comprehensive prediction results for drug pairs, used 
  to validate the model's performance. Due to its large size, it is hosted in 
  the Release section rather than the code repository.
  该文件包含了用于验证模型性能的药物对综合预测结果。由于文件体积较大，它被托管在 
  Release 区域而非代码仓库中。

================================================================================
Notes on Data Usage
数据使用注意事项
================================================================================

1. Large Files / 大文件获取:
   Please download the files listed above from the "Releases" page on the 
   right side of the GitHub repository.
   请从 GitHub 仓库右侧的 "Releases" 页面下载上述列出的文件。

2. Privacy & Licensing / 隐私与许可:
   Ensure compliance with the original licenses of BindingDB and DrugBank when 
   using these datasets.
   使用这些数据集时，请确保遵守 BindingDB 和 DrugBank 的原始许可协议。
