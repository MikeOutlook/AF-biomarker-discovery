"""Configuration settings for AF diagnosis model."""


class ModelConfig:
    """Model configuration."""

    def __init__(self, data_dir="data/", output_dir="results/",
                 n_features=100, test_size=0.2, random_state=42,
                 cv=3, rf_n_estimators=100, rf_max_depth=None,
                 lr_max_iter=1000, svm_kernel="rbf", models=None):
        if models is None:
            models = ["lr", "rf", "svm"]

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_features = n_features
        self.test_size = test_size
        self.random_state = random_state
        self.cv = cv
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.lr_max_iter = lr_max_iter
        self.svm_kernel = svm_kernel
        self.models = models


class InferenceConfig:
    """Inference configuration."""

    def __init__(self, model_path="model.pkl", threshold=0.5):
        self.model_path = model_path
        self.threshold = threshold


DEFAULT_CONFIG = ModelConfig()