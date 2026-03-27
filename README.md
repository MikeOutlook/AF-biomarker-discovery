# Atrial Fibrillation Diagnosis Model based on Gene Expression

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen.svg" alt="Status">
</p>

> 🏥 A machine learning approach to diagnose Atrial Fibrillation (AF) and discover biomarkers using gene expression data from GSE41177 and GSE79768 datasets.

## 🎯 Project Overview

This project develops a clinical diagnosis model for **Atrial Fibrillation (AF)** using gene expression data from cardiac tissue samples. By analyzing the transcriptome signatures, we aim to distinguish AF patients from those with normal Sinus Rhythm.

### What is Atrial Fibrillation?

Atrial Fibrillation (AF) is the most common cardiac arrhythmia, affecting millions of people worldwide. It significantly increases the risk of stroke, heart failure, and other cardiovascular complications. Early and accurate diagnosis is crucial for effective treatment.

## 📊 Dataset

| Dataset | Samples | AF | Sinus Rhythm | Platform |
|---------|---------|-----|--------------|----------|
| GSE41177 | 38 | 32 | 6 | GPL570 |
| GSE79768 | 26 | 14 | 12 | GPL570 |
| **Total** | **64** | **46** | **18** | - |

### Features
- **Gene Expression**: ~22,000 genes per sample
- **Clinical Features**: Age, Gender

## 🔬 Methodology

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1.Data Loading │ ──► │2.Data Merging   │ ──► │3.Preprocessing  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ 6.Model Saving │ ◄── │5.Model Training │ ◄── │4.Feature Select│
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Pipeline
1. **Data Loading**: Load gene expression matrices and clinical metadata
2. **Data Merging**: Combine GSE41177 and GSE79768 datasets
3. **Preprocessing**: Missing value imputation & Standardization
4. **Feature Selection**: ANOVA-based selection of top 100 discriminative genes
5. **Model Training**: Compare Logistic Regression, Random Forest, and SVM
6. **Evaluation**: Cross-validation and test set evaluation

## 📈 Results

### Model Performance (With Clinical Features)

| Model | Test Accuracy | CV Accuracy | Std Dev |
|-------|---------------|--------------|---------|
| Logistic Regression | 84.62% | 80.95% | ±17.82% |
| **Random Forest** | **100%** | **84.49%** | **±7.75%** |
| SVM (RBF kernel) | 100% | 84.42% | ±4.30% |

### Top 10 Discriminative Genes

| Rank | Gene | Score |
|------|------|-------|
| 1 | **STUB1** | 55.77 |
| 2 | **NCF2** | 49.55 |
| 3 | **S100A12** | 47.47 |
| 4 | **RAD23B** | 45.52 |
| 5 | **CXCR2** | 43.28 |
| 6 | **CLC** | 43.12 |
| 7 | **UBE2M** | 42.75 |
| 8 | **S100A6** | 40.94 |
| 9 | **NEIL2** | 40.50 |
| 10 | **DOCK1** | 40.33 |

> 📝 **Note**: These genes are involved in oxidative stress, inflammation, and protein degradation pathways, which are known to be dysregulated in AF.

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pandas
numpy
scikit-learn
matplotlib
```

### Installation

```bash
git clone https://github.com/yourusername/AF-biomarker-discovery.git
cd AF-biomarker-discovery
pip install -r requirements.txt
```

### Run the Model

```bash
# Basic model (gene expression only)
python af_diagnosis_model.py

# Model with clinical features
python af_diagnosis_model_with_clinical.py
```

## 📁 Project Structure

```
AF-biomarker-discovery/
├── data/
│   ├── GSE41177-RNA-seq-matrix.csv    # Raw gene expression
│   ├── GSE79768-RNA-seq-matrix.csv
│   ├── clinical_GSE41177.csv          # Clinical metadata
│   └── clinical_GSE79768.csv
├── src/
│   ├── af_diagnosis_model.py          # Main model script
│   └── af_diagnosis_model_with_clinical.py  # With clinical features
├── results/
│   ├── important_genes.csv            # Top discriminative genes
│   ├── feature_importance.csv         # Feature importance rankings
│   └── model_performance.csv          # Model comparison results
├── README.md
├── LICENSE
└── requirements.txt
```

## 🔬 Key Findings

1. **High Diagnostic Accuracy**: Random Forest and SVM achieve >84% cross-validation accuracy
2. **Gene Signatures**: Identified 100 genes that significantly differentiate AF from Sinus Rhythm
3. **Biological Relevance**: Top genes (STUB1, NCF2, S100A12) are involved in oxidative stress and inflammation pathways
4. **Clinical Features**: Age and gender show limited predictive value in this dataset

## 🧬 Biological Interpretation

The identified biomarker genes are involved in:

- **Oxidative Stress**: STUB1, NCF2, UBE2M
- **Inflammation**: S100A12, S100A6, CXCR2, CLC
- **DNA Repair**: RAD23B, NEIL2
- **Cell Signaling**: DOCK1

These pathways are known to be activated in atrial fibrillation and represent potential therapeutic targets.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Data sourced from [GEO (Gene Expression Omnibus)](https://www.ncbi.nlm.nih.gov/geo/)
- Original studies:
  - GSE41177: Cardiaac tissue gene expression in atrial fibrillation
  - GSE79768: Gene expression profiling in human atrial fibrillation

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{af_biomarker_discovery,
  title={AF Biomarker Discovery},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/AF-biomarker-discovery}
}
```

---

<p align="center">
  Made with ❤️ for cardiac research
</p>
