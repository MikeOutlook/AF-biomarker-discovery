"""Model training and prediction module."""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline

from .config import ModelConfig
from .io import merge_datasets


def create_models(config):
    """Create model pipelines based on config."""
    models = {}

    if "lr" in config.models:
        models["lr"] = SkPipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=config.lr_max_iter,
                random_state=config.random_state
            ))
        ])

    if "rf" in config.models:
        models["rf"] = SkPipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=config.rf_n_estimators,
                max_depth=config.rf_max_depth,
                random_state=config.random_state
            ))
        ])

    if "svm" in config.models:
        models["svm"] = SkPipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel=config.svm_kernel,
                probability=True,
                random_state=config.random_state
            ))
        ])

    return models


def select_features(X, y, k=100):
    """Select top k features using ANOVA."""
    X_filled = X.fillna(X.median())

    k = min(k, X_filled.shape[1], X_filled.shape[0] - 1)
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X_filled, y)

    scores = selector.scores_
    feature_scores = pd.DataFrame({
        "feature": X.columns,
        "score": scores
    }).sort_values("score", ascending=False)

    selected_mask = selector.get_support()
    selected_features = X.loc[:, selected_mask]

    return selected_features, scores, selector, feature_scores


def train_models(X_train, y_train, config):
    """Train all models."""
    models = create_models(config)

    trained = {}
    for name, model in models.items():
        warnings.filterwarnings("ignore")
        model.fit(X_train, y_train)
        trained[name] = model

    return trained


def evaluate_models(models, X_test, y_test, cv=3):
    """Evaluate trained models."""
    results = []

    for name, model in models.items():
        test_acc = model.score(X_test, y_test)
        cv_scores = cross_val_score(model, X_test, y_test, cv=cv)

        results.append({
            "model": name,
            "test_accuracy": test_acc,
            "cv_accuracy": cv_scores.mean(),
            "cv_std": cv_scores.std()
        })

    return pd.DataFrame(results)


def run_pipeline(data_dir="data/", output_dir="results/", config=None, save_models=True):
    """Run complete training pipeline."""
    if config is None:
        config = ModelConfig(data_dir=data_dir, output_dir=output_dir)

    print("=" * 50)
    print("AF Diagnosis Pipeline v2.0")
    print("=" * 50)

    # 1. Load and merge data
    print("\n[1/5] Loading data...")
    X, y, clinical = merge_datasets(data_dir)
    print("  Loaded %d samples, %d features" % (X.shape[0], X.shape[1]))

    # Add clinical features if available
    if clinical is not None:
        common_idx = X.index.intersection(clinical.index)
        clinical_aligned = clinical.loc[common_idx]
        clinical_aligned = clinical_aligned.fillna(clinical_aligned.median())
        X = pd.concat([X, clinical_aligned], axis=1)
        print("  Added clinical features: %s" % clinical_aligned.columns.tolist())

    # 2. Feature selection
    print("\n[2/5] Selecting features...")
    X_selected, scores, selector, feature_scores = select_features(X, y, k=config.n_features)
    print("  Selected top %d features" % config.n_features)

    # 3. Split data
    print("\n[3/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y
    )
    print("  Train: %d, Test: %d" % (len(X_train), len(X_test)))

    # 4. Train models
    print("\n[4/5] Training models...")
    models = train_models(X_train, y_train, config)

    for name in models:
        print("  Trained %s" % name)

    # 5. Evaluate models
    print("\n[5/5] Evaluating models...")
    results = evaluate_models(models, X_test, y_test, cv=config.cv)

    print("\n" + "=" * 50)
    print("Results")
    print("=" * 50)
    print(results.to_string(index=False))

    # Save results
    print("\nSaving results...")
    os.makedirs(output_dir, exist_ok=True)

    feature_scores.to_csv(output_dir + "/feature_importance.csv", index=False)
    results.to_csv(output_dir + "/model_performance.csv", index=False)

    # Save models if requested
    if save_models:
        from .io import save_model

        for name, model in models.items():
            model_path = output_dir + "/" + name + "_model.pkl"
            metadata = {
                "model_type": name,
                "n_features": config.n_features,
                "feature_names": X_selected.columns.tolist(),
                "cv": config.cv
            }
            save_model(model, model_path, metadata)
            print("  Saved %s model to %s" % (name, model_path))

    print("\nDone!")

    return {
        "models": models,
        "selector": selector,
        "feature_scores": feature_scores,
        "results": results,
        "config": config,
        "feature_names": X_selected.columns.tolist()
    }