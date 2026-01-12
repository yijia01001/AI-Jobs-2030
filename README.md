# AI Impact on Jobs 2030 - 数据分析项目

## 📋 项目概述

这是一个全面的数据分析项目，深入研究了 AI 对 2030 年就业市场的影响。项目使用 Python 进行数据分析和机器学习建模，并使用 Tableau 进行数据可视化，展示了从数据清洗到洞察生成的完整数据分析流程。在此基础上，通过量化风险，为企业人才战略、个人职业转型路径提供了关键的业务洞察与决策依据。

## 🎯 项目目标

1. **数据分析**：通过数据处理与统计建模，挖掘数据底层规律
2. **数据可视化**：将复杂的数据转化为直观的决策图表
3. **业务洞察**：识别高风险职位、安全职位，并提供可操作的建议
4. **预测建模**：构建机器学习模型预测自动化风险

## 📁 项目结构

```
AI & Jobs 2030/
│
├── README.md                            # 项目说明文档
├── requirements.txt                     # Python依赖包
├── AI_Impact_on_Jobs_2030.csv           # 原始数据集
├── AI Job Impact Analysis.ipynb         # 主分析Notebook
├── AI Job Impact Analysis.twbx          # Tableau仪表盘
├── scripts.py                           # Python分析脚本
│   
```

## 📊 数据集

- **来源**: Kaggle - AI Impact on Jobs 2030
- **记录数**: 3,000+ 条职位记录
- **关键变量**:
  - Job_Title: 职位名称
  - Average_Salary: 平均年薪（USD）
  - Years_Experience: 工作经验年限
  - Education_Level: 教育水平
  - AI_Exposure_Index: AI暴露度指数（0=无接触，1=高度依赖）
  - Tech_Growth_Factor: 技术增长因子
  - Automation_Probability_2030: 2030年自动化概率
  - Risk_Category: 风险类别
  - Skill_1 到 Skill_10: 10项技能评分

## 🔧 技术栈

### Python 库
- **数据处理**: Pandas, NumPy
- **数据可视化**: Matplotlib, Seaborn, Plotly
- **统计分析**: SciPy, Statsmodels
- **机器学习**: Scikit-learn (Random Forest, Gradient Boosting, Linear Regression)
- **模型评估**: Cross-validation, R², RMSE, MAE

### 可视化工具
- **Tableau**: 交互式仪表板
- **Python**: 静态和交互式图表

## 📈 分析内容

### 1. 探索性数据分析 (EDA)
- 数据质量评估
- 描述性统计
- 数据分布分析
- 创建衍生特征

### 2. 统计分析
- 相关性分析
- 假设检验 (Pearson 相关性，ANOVA) 
- 回归分析

### 3. 聚类分析
- K-Means 聚类
- 肘部法则确定最优聚类数
- PCA 降维可视化
- 聚类特征分析

### 4. 风险分层分析
- 按自动化风险对职位分类
- 识别高风险和低风险职位
- 薪资-风险权衡分析
- 教育投资回报率分析

### 5. 机器学习建模
- **模型类型**:
  - 线性回归
  - Ridge 回归
  - 随机森林
  - 梯度提升
- **模型评估**:
  - R² 分数
  - 交叉验证
  - RMSE 和 MAE
  - 特征重要性分析

## 📊 可视化
- **Python 图表**:
  - Correlation analysis（关键变量相关性分析图表）
  - Skills analysis（技能重要性分析图表）
  - Job-level analysis（职位分析图表）
  - Clustering_analysis（聚类分析图表）
- **Tableau 仪表盘**:
  - 2030 职业风险与机遇指数看板
  - 🔗 https://public.tableau.com/views/AIJobImpactAnalysis/2030

## 💡 业务洞察

### 1. 风险分布
- **高风险职位** (自动化概率 >60%): 约占 23.7%
- **中等风险职位** (30-60%): 约占 38.4%
- **低风险职位** (<30%): 约占 36.9%

### 2. 影响因素 ---
- **教育水平**: 更高教育水平通常对应更低自动化风险
- **技术增长因子**: 技术增长因子与自动化风险呈正相关
- **技能组合**: 某些技能组合能显著降低自动化风险
- **工作经验**: 经验与风险的关系需要进一步分析

### 3. 机会识别
- 识别出高薪资、低风险的最佳职业机会
- 发现教育投资的 ROI 差异


## 🔍 项目亮点

1. **端到端分析流程**: 从原始数据到业务洞察的完整流程
2. **多模型比较**: 比较了 4 种不同的机器学习模型
3. **统计严谨性**: 进行了假设检验和模型诊断
4. **可视化专业性**: 创建了多种类型的专业图表
5. **业务价值**: 提供了可操作的建议和洞察
6. **工具多样性**: 综合使用 Python 和 Tableau

## 🔗 资源链接

- [Kaggle 数据集](https://www.kaggle.com/datasets/khushikyad001/ai-impact-on-jobs-2030)
- [Tableau 仪表盘](https://public.tableau.com/views/AIJobImpactAnalysis/2030)

## 🤝 贡献

这是一个个人作品，用于数据分析学习和记录。所有分析均基于公开数据集，结果仅供参考学习。欢迎提出建议和改进意见。

## 📧 联系方式

如有问题或建议，欢迎联系yijia01001@gmail.com。

---

**最后更新**: 2026年

**状态**: ✅ 完成

