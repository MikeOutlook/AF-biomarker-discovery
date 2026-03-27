#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AF (Atrial Fibrillation) Diagnosis Model
Including Clinical Features (Age, Gender)
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
print("AF Diagnosis Model with Clinical Features")
print("="*60)

# =============================================================================
# Step 1: Load Data
# =============================================================================
print("\n[Step 1] Loading data...")

# Load gene expression matrices
expr_41177 = pd.read_csv('GSE41177-RNA-seq-matrix.csv', index_col=0)
expr_79768 = pd.read_csv('GSE79768-RNA-seq-matrix.csv', index_col=0)

# Load clinical data
clinical_41177 = pd.read_csv('clinical_GSE41177.csv')
clinical_79768 = pd.read_csv('clinical_GSE79768.csv')

print(f"GSE41177: {expr_41177.shape[1]} samples, {expr_41177.shape[0]} genes")
print(f"GSE79768: {expr_79768.shape[1]} samples, {expr_79768.shape[0]} genes")

# =============================================================================
# Step 2: Extract Labels and Clinical Features
# =============================================================================
print("\n[Step 2] Extracting labels and clinical features...")

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
# Step 3: Process Clinical Features
# =============================================================================
print("\n[Step 3] Processing clinical features...")

# Get sample info with clinical features
samples_41177 = clinical_41177[['geo_accession', 'label', 'age:ch1', 'gender:ch1']].dropna()
samples_79768 = clinical_79768[['geo_accession', 'label', 'age:ch1', 'gender:ch1']].dropna()

# Rename columns
samples_41177.columns = ['sample_id', 'label', 'age', 'gender']
samples_79768.columns = ['sample_id', 'label', 'age', 'gender']

# Process age - convert "62Y" to numeric
def parse_age(age_str):
    try:
        return float(str(age_str).replace('Y', ''))
    except:
        return np.nan

samples_41177['age'] = samples_41177['age'].apply(parse_age)
samples_79768['age'] = samples_79768['age'].apply(parse_age)

# Process gender - encode as 0/1
def encode_gender(gender):
    gender = str(gender).lower().strip()
    if 'male' in gender:
        return 1
    elif 'female' in gender:
        return 0
    return np.nan

samples_41177['gender'] = samples_41177['gender'].apply(encode_gender)
samples_79768['gender'] = samples_79768['gender'].apply(encode_gender)

# Drop any remaining NaN
samples_41177 = samples_41177.dropna()
samples_79768 = samples_79768.dropna()

print(f"GSE41177 samples after processing: {len(samples_41177)}")
print(f"GSE79768 samples after processing: {len(samples_79768)}")

# Print clinical feature statistics
print("\nClinical Feature Statistics:")
print(f"GSE41177 - Age: mean={samples_41177['age'].mean():.1f}, Male={samples_41177['gender'].sum()}, Female={len(samples_41177)-samples_41177['gender'].sum()}")
print(f"GSE79768 - Age: mean={samples_79768['age'].mean():.1f}, Male={samples_79768['gender'].sum()}, Female={len(samples_79768)-samples_79768['gender'].sum()}")

# =============================================================================
# Step 4: Merge Datasets (Gene Expression + Clinical Features)
# =============================================================================
print("\n[Step 4] Merging gene expression with clinical features...")

# Find common genes
common_genes = list(set(expr_41177.index) & set(expr_79768.index))
print(f"Common genes: {len(common_genes)}")

# Extract gene expression matrices (transpose: samples as rows, genes as columns)
X_41177_genes = expr_41177.loc[common_genes, samples_41177['sample_id'].values].T
X_79768_genes = expr_79768.loc[common_genes, samples_79768['sample_id'].values].T

# Get labels
y_41177 = samples_41177['label'].values
y_79768 = samples_79768['label'].values

# Get clinical features
clinical_41177 = samples_41177[['age', 'gender']].values
clinical_79768 = samples_79768[['age', 'gender']].values

# Merge gene expression data
X_genes = pd.concat([X_41177_genes, X_79768_genes], axis=0)

# Merge clinical features
clinical_combined = np.vstack([clinical_41177, clinical_79768])
clinical_df = pd.DataFrame(clinical_combined, columns=['age', 'gender'], index=X_genes.index)

# Combine: Gene expression + Clinical features
X_combined = pd.concat([X_genes, clinical_df], axis=1)

y = np.concatenate([y_41177, y_79768])

print(f"Combined dataset: {X_combined.shape[0]} samples")
print(f"  - Gene features: {X_genes.shape[1]}")
print(f"  - Clinical features: {clinical_df.shape[1]} (age, gender)")
print(f"Total features: {X_combined.shape[1]}")
print(f"Label distribution: Sinus Rhythm={sum(y==0)}, AF={sum(y==1)}")

# =============================================================================
# Step 5: Data Preprocessing
# =============================================================================
print("\n[Step 5] Data preprocessing...")

# Handle missing values
X_combined = X_combined.fillna(X_combined.median())

# Feature standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

print("Data preprocessing completed")

# =============================================================================
# Step 6: Feature Selection (Genes Only)
# =============================================================================
print("\n[Step 6] Feature selection on gene expression data...")

# First, scale genes separately for feature selection
gene_scaler = StandardScaler()
X_genes_scaled = gene_scaler.fit_transform(X_genes.fillna(X_genes.median()))

# Use ANOVA to select top 100 most relevant genes
gene_selector = SelectKBest(f_classif, k=min(100, X_genes_scaled.shape[1]))
X_genes_selected = gene_selector.fit_transform(X_genes_scaled, y)

# Get selected gene indices in the original gene matrix
selected_gene_indices = gene_selector.get_support(indices=True)
selected_gene_names = X_genes.columns[selected_gene_indices].tolist()

print(f"Selected {len(selected_gene_names)} important genes")

# Now combine selected genes with clinical features
# Get the selected gene columns from X_combined
X_combined_selected = X_combined[selected_gene_names + ['age', 'gender']]

# Scale the combined features
final_scaler = StandardScaler()
X_final = final_scaler.fit_transform(X_combined_selected)

print(f"Final feature matrix: {X_final.shape[1]} features (100 genes + 2 clinical)")

# Get gene scores
gene_scores = pd.DataFrame({
    'gene': X_genes.columns,
    'score': gene_selector.scores_
}).sort_values('score', ascending=False)
print("\nTop 10 Most Important Genes:")
print(gene_scores.head(10).to_string(index=False))

# =============================================================================
# Step 7: Train/Test Split
# =============================================================================
print("\n[Step 7] Splitting data into train and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# =============================================================================
# Step 8: Train and Evaluate Models (With Clinical Features)
# =============================================================================
print("\n[Step 8] Training models WITH clinical features...")

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF kernel)': SVC(kernel='rbf', random_state=42)
}

results_with_clinical = {}

for name, model in models.items():
    print(f"\n--- {name} ---")

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, X_final, y, cv=3)

    results_with_clinical[name] = {
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
# Step 9: Train Models WITHOUT Clinical Features (for comparison)
# =============================================================================
print("\n" + "="*60)
print("[Step 9] Comparison: WITHOUT Clinical Features")
print("="*60)

# Scale gene-only features
gene_only_scaler = StandardScaler()
X_genes_final = gene_only_scaler.fit_transform(X_genes_selected)

X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
    X_genes_final, y, test_size=0.2, random_state=42, stratify=y
)

results_no_clinical = {}

for name, model in models.items():
    # Create a new instance of the model
    if name == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif name == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = SVC(kernel='rbf', random_state=42)

    model.fit(X_train_g, y_train_g)
    y_pred = model.predict(X_test_g)
    accuracy = accuracy_score(y_test_g, y_pred)
    cv_scores = cross_val_score(model, X_genes_final, y, cv=3)

    results_no_clinical[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

    print(f"{name}: CV Accuracy = {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

# =============================================================================
# Step 10: Model Comparison Summary
# =============================================================================
print("\n" + "="*60)
print("Model Performance Summary: WITH vs WITHOUT Clinical Features")
print("="*60)

print("\n--- WITH Clinical Features (Age + Gender) ---")
summary_with = pd.DataFrame({
    'Model': list(results_with_clinical.keys()),
    'Test Accuracy': [results_with_clinical[m]['accuracy'] for m in results_with_clinical],
    'CV Accuracy': [results_with_clinical[m]['cv_mean'] for m in results_with_clinical],
    'Std Dev': [results_with_clinical[m]['cv_std'] for m in results_with_clinical]
})
print(summary_with.to_string(index=False))

print("\n--- WITHOUT Clinical Features (Gene Only) ---")
summary_without = pd.DataFrame({
    'Model': list(results_no_clinical.keys()),
    'Test Accuracy': [results_no_clinical[m]['accuracy'] for m in results_no_clinical],
    'CV Accuracy': [results_no_clinical[m]['cv_mean'] for m in results_no_clinical],
    'Std Dev': [results_no_clinical[m]['cv_std'] for m in results_no_clinical]
})
print(summary_without.to_string(index=False))

# Best model
best_model = max(results_with_clinical, key=lambda x: results_with_clinical[x]['cv_mean'])
print(f"\nBest Model (with clinical): {best_model}")

# Feature importance from Random Forest
print("\n" + "="*60)
print("Feature Importance Analysis (Random Forest)")
print("="*60)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Get feature names
feature_names = selected_gene_names + ['age', 'gender']
importances = rf_model.feature_importances_

# Sort by importance
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(importance_df.head(15).to_string(index=False))

# Check where clinical features rank
clinical_importance = importance_df[importance_df['feature'].isin(['age', 'gender'])]
print("\nClinical Feature Importance:")
print(clinical_importance.to_string(index=False))

# =============================================================================
# Step 11: Save Results
# =============================================================================
print("\n[Step 11] Saving results...")

# Save gene scores
gene_scores.to_csv('important_genes_with_clinical.csv', index=False)

# Save feature importance
importance_df.to_csv('feature_importance.csv', index=False)

# Save comparison
comparison = pd.DataFrame({
    'Model': list(results_with_clinical.keys()),
    'With_Clinical_Test_Acc': [results_with_clinical[m]['accuracy'] for m in results_with_clinical],
    'With_Clinical_CV_Acc': [results_with_clinical[m]['cv_mean'] for m in results_with_clinical],
    'Without_Clinical_Test_Acc': [results_no_clinical[m]['accuracy'] for m in results_no_clinical],
    'Without_Clinical_CV_Acc': [results_no_clinical[m]['cv_mean'] for m in results_no_clinical]
})
comparison.to_csv('clinical_feature_comparison.csv', index=False)

print("Results saved:")
print("  - important_genes_with_clinical.csv")
print("  - feature_importance.csv")
print("  - clinical_feature_comparison.csv")

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)
