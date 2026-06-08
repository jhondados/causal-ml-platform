"""Causal ML treatment effect estimator."""
from econml.dml import CausalForestDML
from econml.metalearners import XLearner
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import numpy as np
from typing import Dict

class CausalEffectEstimator:
    def __init__(self, method: str = "causal_forest"):
        self.method = method
        if method == "causal_forest":
            self.model = CausalForestDML(
                model_y=GradientBoostingRegressor(), model_t=GradientBoostingClassifier(),
                n_estimators=200, min_samples_leaf=10, random_state=42)
        elif method == "xlearner":
            self.model = XLearner(models=GradientBoostingRegressor())

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "CausalEffectEstimator":
        """Fit causal model. X=features, T=treatment (0/1), Y=outcome."""
        self.model.fit(Y, T, X=X)
        return self

    def estimate_cate(self, X_test: np.ndarray) -> Dict:
        """Estimate Conditional Average Treatment Effect per individual."""
        cate = self.model.effect(X_test)
        return {"cate_mean": float(np.mean(cate)), "cate_std": float(np.std(cate)),
                "cate_p10": float(np.percentile(cate, 10)), "cate_p90": float(np.percentile(cate, 90)),
                "positive_effect_pct": float(np.mean(cate > 0) * 100), "individual_effects": cate.tolist()}

    def uplift_segments(self, X_test: np.ndarray, n_segments: int = 4) -> Dict:
        """Segment population by treatment responsiveness."""
        cate = self.model.effect(X_test)
        percentiles = np.percentile(cate, np.linspace(0, 100, n_segments + 1))
        segments = {}
        for i in range(n_segments):
            mask = (cate >= percentiles[i]) & (cate < percentiles[i+1])
            segments[f"segment_{i+1}"] = {"avg_cate": float(np.mean(cate[mask])), "size": int(mask.sum()),
                "recommendation": "treat" if np.mean(cate[mask]) > 0 else "do_not_treat"}
        return segments
