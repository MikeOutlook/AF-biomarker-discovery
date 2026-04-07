"""Data loading and saving utilities."""

import json
import os
import warnings

import numpy as np
import pandas as pd


def load_gene_expression(filepath, clinical_filepath=None):
    """Load gene expression data with clinical metadata."""
    if not os.path.exists(filepath):
        raise FileNotFoundError("Data file not found: %s" % filepath)

    # Load gene expression data
    gene_df = pd.read_csv(filepath, index_col=0)

    # Transpose gene data - make samples (GSM IDs) rows, genes columns
    features = gene_df.T
    sample_ids = features.index

    # Load clinical data to get labels
    labels = None
    clinical_features = None

    if clinical_filepath and os.path.exists(clinical_filepath):
        clinical_df = pd.read_csv(clinical_filepath, index_col=0)

        # Ensure index alignment
        common_idx = sample_ids.intersection(clinical_df.index)
        if len(common_idx) == 0:
            raise ValueError("No matching samples between gene and clinical data")

        # Extract labels from title column
        # title contains "Sinus rhythm" or "AF" info
        if "title" in clinical_df.columns:
            clinical_subset = clinical_df.loc[common_idx]
            # Sinus rhythm = 0, AF = 1
            labels = clinical_subset["title"].apply(
                lambda x: 0 if "sinus" in str(x).lower() else 1
            )

        # Extract clinical features
        clinical_features = pd.DataFrame(index=common_idx)

        if "age:ch1" in clinical_df.columns:
            clinical_features["age"] = clinical_df.loc[common_idx, "age:ch1"].apply(_parse_age)

        if "gender:ch1" in clinical_df.columns:
            clinical_features["gender"] = clinical_df.loc[common_idx, "gender:ch1"].apply(_encode_gender)

        # Reindex to match features
        features = features.loc[common_idx]
        labels = labels.loc[common_idx]

    return features, labels, clinical_features


def _parse_age(age_str):
    """Parse age from string like '62Y' to numeric."""
    try:
        if pd.isna(age_str):
            return np.nan
        return float(str(age_str).replace("Y", "").strip())
    except (ValueError, AttributeError):
        return np.nan


def _encode_gender(gender):
    """Encode gender to numeric (male=1, female=0)."""
    if pd.isna(gender):
        return np.nan
    gender_lower = str(gender).lower()
    if gender_lower in ["m", "male", "males"]:
        return 1
    elif gender_lower in ["f", "female", "females"]:
        return 0
    return np.nan


def merge_datasets(data_dir, datasets=None):
    """Merge multiple datasets."""
    if datasets is None:
        datasets = ["GSE41177", "GSE79768"]

    all_features = []
    all_labels = []
    all_clinical = []

    for name in datasets:
        gene_file = os.path.join(data_dir, "%s-RNA-seq-matrix.csv" % name)
        clinical_file = os.path.join(data_dir, "clinical_%s.csv" % name)

        if not os.path.exists(gene_file):
            warnings.warn("Dataset %s not found, skipping" % name)
            continue

        try:
            features, labels, clinical = load_gene_expression(gene_file, clinical_file)

            all_features.append(features)
            all_labels.append(labels)

            if clinical is not None and not clinical.empty:
                all_clinical.append(clinical)
        except Exception as e:
            warnings.warn("Error loading %s: %s" % (name, e))
            continue

    if not all_features:
        raise ValueError("No data loaded")

    # Find common genes (columns) across all datasets
    common_genes = all_features[0].columns
    for f in all_features[1:]:
        common_genes = common_genes.intersection(f.columns)

    if len(common_genes) == 0:
        raise ValueError("No common genes found between datasets")

    # Keep only common genes
    all_features = [f[common_genes] for f in all_features]

    merged_features = pd.concat(all_features, axis=0)
    merged_labels = pd.concat(all_labels, axis=0)

    # Ensure alignment
    common_idx = merged_features.index.intersection(merged_labels.index)
    merged_features = merged_features.loc[common_idx]
    merged_labels = merged_labels.loc[common_idx]

    merged_clinical = None
    if all_clinical:
        merged_clinical = pd.concat(all_clinical, axis=0)
        try:
            merged_clinical = merged_clinical.loc[common_idx]
        except:
            merged_clinical = None

    return merged_features, merged_labels, merged_clinical


def save_model(model, path, metadata=None):
    """Save model to file with metadata."""
    import joblib

    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    joblib.dump(model, path)

    if metadata:
        metadata_path = path.replace(".pkl", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


def load_model(path):
    """Load model from file with metadata."""
    import joblib

    if not os.path.exists(path):
        raise FileNotFoundError("Model file not found: %s" % path)

    model = joblib.load(path)

    metadata_path = path.replace(".pkl", "_metadata.json")
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    return model, metadata


def save_results(results, output_dir, prefix=""):
    """Save results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    for name, df in results.items():
        if df is None:
            continue
        filename = "%s%s.csv" % (prefix, name) if prefix else "%s.csv" % name
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath)