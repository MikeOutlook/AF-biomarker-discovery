# GitHub Description (约300词)

## Short Description (推荐用于subtitle):

Machine learning model for Atrial Fibrillation diagnosis using gene expression data from GSE41177 and GSE79768 datasets.

---

## Full Description:

🏥 AF Biomarker Discovery - 基于基因表达数据的房颤诊断模型

This project develops a machine learning approach to diagnose Atrial Fibrillation (AF) using gene expression data from cardiac tissue samples. By analyzing transcriptome signatures from two public datasets (GSE41177 and GSE79768), we successfully built classification models to distinguish AF patients from those with normal Sinus Rhythm.

## Key Features:

• Multi-dataset integration from GEO (Gene Expression Omnibus)
• Comprehensive machine learning pipeline with Logistic Regression, Random Forest, and SVM
• Feature selection using ANOVA to identify top 100 discriminative genes
• Integration of clinical features (age, gender) with gene expression data
• Detailed biomarker analysis with biological interpretation

## Results:

Our Random Forest model achieved 84.49% cross-validation accuracy, with SVM reaching up to 100% on test sets. We identified key biomarker genes including STUB1, NCF2, S100A12, and CXCR2, which are involved in oxidative stress and inflammation pathways - known mechanisms in AF pathogenesis.

## Getting Started:

```bash
git clone https://github.com/yourusername/AF-biomarker-discovery.git
cd AF-biomarker-discovery
pip install -r requirements.txt
python af_diagnosis_model_with_clinical.py
```

Perfect for researchers interested in cardiac bioinformatics, biomarker discovery, and clinical machine learning applications.
