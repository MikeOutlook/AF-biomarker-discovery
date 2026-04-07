"""Command-line interface for AF diagnosis model."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import af_diagnosis
from af_diagnosis.config import ModelConfig
from af_diagnosis.pipeline import run_pipeline
from af_diagnosis.evaluation import (
    evaluate_model,
    print_evaluation,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance
)
from af_diagnosis.inference import AFPredictor


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="af-diagnosis",
        description="Atrial Fibrillation Diagnosis Model - End-to-end ML pipeline"
    )
    parser.add_argument(
        "--version", action="version", version="%%(prog)s %s" % af_diagnosis.__version__
    )
    return parser


def add_run_parser(subparsers):
    """Add 'run' subcommand."""
    parser = subparsers.add_parser("run", help="Train models on data")
    parser.add_argument("--data", "-d", default="data/", help="Data directory")
    parser.add_argument("--output", "-o", default="results/", help="Output directory")
    parser.add_argument("--models", "-m", default="lr,rf,svm", help="Models to train")
    parser.add_argument("--n-features", "-k", type=int, default=100, help="Number of features")
    parser.add_argument("--cv", "-cv", type=int, default=3, help="Cross-validation folds")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--no-save", action="store_true", help="Don't save trained models")
    return parser


def add_predict_parser(subparsers):
    """Add 'predict' subcommand."""
    parser = subparsers.add_parser("predict", help="Predict AF status for new samples")
    parser.add_argument("--model", "-m", default="results/rf_model.pkl", help="Model path")
    parser.add_argument("--sample", "-s", required=True, help="Sample CSV or JSON")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Threshold")
    parser.add_argument("--output", "-o", help="Output path for predictions")
    return parser


def add_eval_parser(subparsers):
    """Add 'eval' subcommand."""
    parser = subparsers.add_parser("eval", help="Evaluate model on test data")
    parser.add_argument("--model", "-m", required=True, help="Model path")
    parser.add_argument("--test", "-t", required=True, help="Test data path")
    parser.add_argument("--output", "-o", default="results/", help="Output directory")
    parser.add_argument("--plots", action="store_true", help="Generate plots")
    return parser


def cmd_run(args):
    """Execute 'run' command."""
    models = [m.strip() for m in args.models.split(",")]

    config = ModelConfig(
        data_dir=args.data,
        output_dir=args.output,
        n_features=args.n_features,
        cv=args.cv,
        test_size=args.test_size,
        models=models
    )

    result = run_pipeline(
        data_dir=args.data,
        output_dir=args.output,
        config=config,
        save_models=not args.no_save
    )

    print("\nPipeline completed successfully!")
    return 0


def cmd_predict(args):
    """Execute 'predict' command."""
    if not os.path.exists(args.model):
        print("Error: Model not found: %s" % args.model)
        return 1

    predictor = AFPredictor(args.model, args.threshold)

    if os.path.isfile(args.sample):
        X = pd.read_csv(args.sample, index_col=0)
        predictions, confidence = predictor.predict_with_confidence(X)

        results = pd.DataFrame({
            "prediction": ["AF" if p == 1 else "Sinus Rhythm" for p in predictions],
            "confidence": confidence
        })

        print("\nPredictions:")
        print(results.to_string())

        if args.output:
            results.to_csv(args.output)
            print("\nSaved to %s" % args.output)
    else:
        import json
        try:
            features = json.loads(args.sample)
        except json.JSONDecodeError:
            print("Error: Sample must be CSV path or JSON dictionary")
            return 1

        result = predictor.predict_single(features)
        print("\nPrediction:")
        print("  Diagnosis: %s" % result["prediction"])
        print("  Confidence: %.1f%%" % (result["confidence"] * 100))
        print("  P(AF): %.1f%%" % (result["probability_af"] * 100))
        print("  P(Sinus): %.1f%%" % (result["probability_sr"] * 100))

    return 0


def cmd_eval(args):
    """Execute 'eval' command."""
    if not os.path.exists(args.model):
        print("Error: Model not found: %s" % args.model)
        return 1

    predictor = AFPredictor(args.model)

    if not os.path.exists(args.test):
        print("Error: Test data not found: %s" % args.test)
        return 1

    test_data = pd.read_csv(args.test, index_col=0)

    if "label" in test_data.columns or "diagnosis" in test_data.columns:
        label_col = "label" if "label" in test_data.columns else "diagnosis"
        X = test_data.drop(columns=[label_col])
        y = test_data[label_col].map({"AF": 1, "Sinus Rhythm": 0, 1: 1, 0: 0})
    else:
        print("Warning: No label column found, evaluating on all data")
        X = test_data
        y = None

    if y is not None:
        metrics = evaluate_model(predictor.model, X, y)
        print_evaluation(metrics)

        if args.plots:
            os.makedirs(args.output, exist_ok=True)
            cm = np.array(metrics["confusion_matrix"])
            plot_confusion_matrix(cm, save_path=args.output + "/confusion_matrix.png")

            y_proba = predictor.predict_proba(X)
            if y_proba is not None and y is not None:
                plot_roc_curve(y.values, y_proba[:, 1], save_path=args.output + "/roc_curve.png")

            print("\nPlots saved to %s/" % args.output)
    else:
        predictions, confidence = predictor.predict_with_confidence(X)
        results = pd.DataFrame({
            "prediction": ["AF" if p == 1 else "Sinus Rhythm" for p in predictions],
            "confidence": confidence
        })
        print("\nPredictions:")
        print(results.to_string())

    return 0


def main():
    """Main CLI entry point."""
    parser = create_parser()
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    add_run_parser(subparsers)
    add_predict_parser(subparsers)
    add_eval_parser(subparsers)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "predict":
        return cmd_predict(args)
    elif args.command == "eval":
        return cmd_eval(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())