"""AF Diagnosis Model - End-to-end machine learning pipeline for atrial fibrillation detection."""

__version__ = "2.0.0"
__author__ = "Ziheng Zheng"

from .pipeline import run_pipeline
from .inference import predict, predict_proba, AFPredictor
from .evaluation import evaluate, evaluate_model
from .config import ModelConfig

__all__ = [
    "run_pipeline",
    "predict",
    "predict_proba",
    "AFPredictor",
    "evaluate",
    "evaluate_model",
    "ModelConfig",
]