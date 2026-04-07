# Atrial Fibrillation (AF) Diagnosis Model

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" alt="Status">
</p>

> 🏥 An end-to-end machine learning pipeline for diagnosing **Atrial Fibrillation (AF)** using gene expression data and clinical features. Built with CLI interface for easy training, prediction, and evaluation.

## What is Atrial Fibrillation?

Atrial Fibrillation (AF) is the most common cardiac arrhythmia, affecting millions worldwide. It significantly increases the risk of stroke, heart failure, and cardiovascular complications. Early and accurate diagnosis is crucial for effective treatment.

This project uses gene expression profiles from cardiac tissue samples to distinguish AF patients from those with normal Sinus Rhythm.

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/MikeOutlook/AF-biomarker-discovery.git
cd AF-biomarker-discovery
pip install -r requirements.txt
```

### Train a Model

```bash
python af_diagnosis/cli.py run --data data/ --output results/
```

### Predict on New Samples

```bash
# Using CLI
python af_diagnosis/cli.py predict --model results/rf_model.pkl --sample new_samples.csv
```

### Evaluate Model Performance

```bash
python af_diagnosis/cli.py eval --model results/rf_model.pkl --test test_data.csv --plots
```

## 📊 Results

| Model | Test Accuracy | CV Accuracy |
|-------|--------------|-------------|
| Logistic Regression | 84.62% | 80.95% |
| **Random Forest** | **100%** | **84.49%** |
| SVM (RBF) | 100% | 84.42% |

### Top Discriminative Genes

| Rank | Gene | Pathway |
|------|------|---------|
| 1 | **STUB1** | Oxidative Stress |
| 2 | **NCF2** | Inflammation |
| 3 | **S100A12** | Inflammation |
| 4 | **RAD23B** | DNA Repair |
| 5 | **CXCR2** | Inflammation |

> These genes are involved in pathways known to be dysregulated in AF.

## 📁 Project Structure

```
AF-biomarker-discovery/
├── af_diagnosis/          # Main package
│   ├── cli.py            # Command-line interface
│   ├── pipeline.py       # Training pipeline
│   ├── inference.py     # Prediction API
│   ├── evaluation.py   # Metrics & visualization
│   └── io.py          # Data I/O
├── data/                 # Gene expression datasets
│   ├── GSE41177-RNA-seq-matrix.csv
│   ├── GSE79768-RNA-seq-matrix.csv
│   ├── clinical_GSE41177.csv
│   └── clinical_GSE79768.csv
├── results/              # Output results
├── README.md
└── requirements.txt
```

## 🔧 CLI Options

### Training Options

```bash
python af_diagnosis/cli.py run [options]

Options:
  --data DATA         Data directory (default: data/)
  --output OUTPUT   Output directory (default: results/)
  --models MODELS   Models: lr,rf,svm (default: lr,rf,svm)
  -k N_FEATURES   Number of features (default: 100)
  --cv CV          Cross-validation folds (default: 3)
  --test-size     Test set proportion (default: 0.2)
```

### Prediction Options

```bash
python af_diagnosis/cli.py predict [options]

Options:
  --model MODEL      Path to saved model
  --sample SAMPLE   Path to sample CSV or JSON
  --threshold     Decision threshold (default: 0.5)
```

### Evaluation Options

```bash
python af_diagnosis/cli.py eval [options]

Options:
  --model MODEL    Path to saved model
  --test TEST     Test data CSV
  --plots        Generate visualization plots
```

## 📈 Dataset

| Dataset | Samples | AF | Sinus Rhythm |
|---------|---------|-----|--------------|
| GSE41177 | 38 | 32 | 6 |
| GSE79768 | 26 | 14 | 12 |
| **Total** | **64** | **46** | **18** |

Features: ~22,000 genes per sample + clinical (Age, Gender)

## 🧬 Biological Significance

The identified biomarker genes are involved in:

- **Oxidative Stress**: STUB1, NCF2, UBE2M
- **Inflammation**: S100A12, S100A6, CXCR2, CLC
- **DNA Repair**: RAD23B, NEIL2
- **Cell Signaling**: DOCK1

These pathways are known to be activated in atrial fibrillation.

## 📝 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- Data from [GEO (Gene Expression Omnibus)](https://www.ncbi.nlm.nih.gov/geo/):
  - GSE41177: Cardiac tissue gene expression in AF
  - GSE79768: Gene expression profiling in human AF

---

<p align="center">
Made with ❤️ for cardiac research
</p>