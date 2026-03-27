#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
房颤(AF)诊断模型 - 基于GSE41177和GSE79768基因表达数据
这是一个初学者友好的机器学习教程
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("AF (Atrial Fibrillation) Diagnosis Model")
print("Gene Expression Data Analysis")
print("="*60)

# =============================================================================
# Step 1: Load Data
# =============================================================================
print("\n[Step 1] Loading data...")

# 加载基因表达矩阵
expr_41177 = pd.read_csv('../data/GSE41177-RNA-seq-matrix.csv', index_col=0)
expr_79768 = pd.read_csv('../data/GSE79768-RNA-seq-matrix.csv', index_col=0)

# 加载临床数据
clinical_41177 = pd.read_csv('../data/clinical_GSE41177.csv')
clinical_79768 = pd.read_csv('../data/clinical_GSE79768.csv')

print(f"GSE41177: {expr_41177.shape[1]} 个样本, {expr_41177.shape[0]} 个基因")
print(f"GSE79768: {expr_79768.shape[1]} 个样本, {expr_79768.shape[0]} 个基因")

# =============================================================================
# Step 2: Extract Labels (AF vs Sinus Rhythm)
# =============================================================================
print("\n[Step 2] Extracting labels...")

# GSE41177 - extract from title column
def extract_label_41177(title):
    if 'Sinus' in str(title):
        return 0  # Sinus rhythm = 0
    elif 'AF' in str(title) or 'Fibrillation' in str(title):
        return 1  # AF = 1
    return None

clinical_41177['label'] = clinical_41177['title'].apply(extract_label_41177)

# GSE79768 - extract from condition:ch1 column
def extract_label_79768(condition):
    if 'Sinus' in str(condition):
        return 0
    elif 'Fibrillation' in str(condition):
        return 1
    return None

clinical_79768['label'] = clinical_79768['condition:ch1'].apply(extract_label_79768)

# Print label distribution
print("GSE41177 Label Distribution:")
print(clinical_41177['label'].value_counts().rename({0: 'Sinus Rhythm', 1: 'AF'}))
print("\nGSE79768 Label Distribution:")
print(clinical_79768['label'].value_counts().rename({0: 'Sinus Rhythm', 1: 'AF'}))

# =============================================================================
# Step 3: Merge Datasets
# =============================================================================
print("\n[Step 3] Merging datasets...")

# 获取样本ID和标签的映射
samples_41177 = clinical_41177[['geo_accession', 'label', 'age:ch1', 'gender:ch1']].dropna()
samples_79768 = clinical_79768[['geo_accession', 'label', 'age:ch1', 'gender:ch1']].dropna()

# 重命名列
samples_41177.columns = ['sample_id', 'label', 'age', 'gender']
samples_79768.columns = ['sample_id', 'label', 'age', 'gender']

# 找到两个数据集的共同基因
common_genes = list(set(expr_41177.index) & set(expr_79768.index))
print(f"共同基因数量: {len(common_genes)}")

# 提取基因表达矩阵（转置：样本为行，基因为列）
X_41177 = expr_41177.loc[common_genes, samples_41177['sample_id'].values].T
X_79768 = expr_79768.loc[common_genes, samples_79768['sample_id'].values].T

# 添加标签
y_41177 = samples_41177['label'].values
y_79768 = samples_79768['label'].values

# Merge data
X = pd.concat([X_41177, X_79768], axis=0)
y = np.concatenate([y_41177, y_79768])

print(f"Merged dataset: {X.shape[0]} samples, {X.shape[1]} genes")
print(f"Label distribution: Sinus Rhythm={sum(y==0)}, AF={sum(y==1)}")

# =============================================================================
# Step 4: Data Preprocessing
# =============================================================================
print("\n[Step 4] Data preprocessing...")

# 处理缺失值
X = X.fillna(X.median())

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data preprocessing completed")

# =============================================================================
# Step 5: Feature Selection
# =============================================================================
print("\n[Step 5] Feature selection - finding genes related to AF...")

# Use ANOVA to select top 100 most relevant genes
selector = SelectKBest(f_classif, k=min(100, X_scaled.shape[1]))
X_selected = selector.fit_transform(X_scaled, y)

# Get selected gene names
selected_gene_indices = selector.get_support(indices=True)
selected_genes = X.columns[selected_gene_indices].tolist()

print(f"Selected {len(selected_genes)} important features (genes)")
print("Top 10 Most Important Genes:")
# Get gene scores
gene_scores = pd.DataFrame({
    'gene': X.columns,
    'score': selector.scores_
}).sort_values('score', ascending=False)
print(gene_scores.head(10).to_string(index=False))

# =============================================================================
# Step 6: Train/Test Split
# =============================================================================
print("\n[Step 6] Splitting data into train and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# =============================================================================
# Step 7: Train and Evaluate Models
# =============================================================================
print("\n[Step 7] Training and evaluating models...")

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF kernel)': SVC(kernel='rbf', random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n--- {name} ---")

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, X_selected, y, cv=3)

    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }

    print(f"Test Accuracy: {accuracy:.2%}")
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Sinus Rhythm', 'AF']))

# =============================================================================
# Step 8: Model Comparison Summary
# =============================================================================
print("\n" + "="*60)
print("Model Performance Summary")
print("="*60)

summary = pd.DataFrame({
    'Model': list(results.keys()),
    'Test Accuracy': [results[m]['accuracy'] for m in results],
    'CV Accuracy': [results[m]['cv_mean'] for m in results],
    'Std Dev': [results[m]['cv_std'] for m in results]
})
print(summary.to_string(index=False))

# Select best model
best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
print(f"\nBest Model: {best_model_name}")

# =============================================================================
# Step 9: Save Results
# =============================================================================
print("\n[Step 9] Saving results...")

# Save selected gene list
gene_scores.to_csv('important_genes.csv', index=False)
print("Important genes saved to: important_genes.csv")

# Save model performance
summary.to_csv('model_performance.csv', index=False)
print("Model performance saved to: model_performance.csv")

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)
