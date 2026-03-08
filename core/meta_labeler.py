"""
AlphaDesk — Meta-Labeling Module (López de Prado)

Implements the meta-labeling technique from "Advances in Financial Machine
Learning" (Chapter 3). A secondary binary classifier learns to filter false
positives from each base strategy.

How it works:
1. The base strategy provides the direction (BUY/SELL).
2. The meta-label model predicts whether that signal will be profitable (1/0).
3. The predicted probability doubles as a position-size multiplier.

This is distinct from the MLEnsemble, which does general P(profit) prediction
across all strategies. Meta-labeling is per-strategy and trained specifically
on each strategy's own historical signal distribution.

Key design choices:
- Purged K-fold cross-validation to prevent look-ahead bias
- Embargo window (±2 days) around fold boundaries
- Per-strategy model persistence in data/models/meta_{strategy}.pkl
- LightGBM preferred, sklearn GradientBoosting as fallback
"""

import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("alphadesk.meta_labeler")

# ── Optional dependency: LightGBM ──
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold


# ═══════════════════════════════════════════════════════════
#  Feature Engineering
# ═══════════════════════════════════════════════════════════

# Canonical feature list for the meta-labeler.  Order matters — the same
# order must be used at training and inference time.
META_FEATURE_NAMES = [
    # Strategy-specific score (caller fills the one that applies)
    "momentum_score",
    "z_score",
    "factor_score",
    "carry_score",
    # Signal metadata
    "confidence",
    "risk_reward_ratio",
    # Technical indicators
    "rsi",
    "macd",
    "volume_ratio",
    "atr_pct",
    # Regime (numeric-encoded)
    "regime_volatility",
    "regime_trend",
    "regime_liquidity",
    # Time features (cyclical)
    "hour_sin",
    "hour_cos",
]

NUM_FEATURES = len(META_FEATURE_NAMES)

# Regime category → numeric (reuses ml_ensemble conventions)
_VOL_MAP = {"low": 0.0, "medium": 1.0, "high": 2.0, "extreme": 3.0}
_TREND_MAP = {"strong_down": -2.0, "weak_down": -1.0, "ranging": 0.0,
              "weak_up": 1.0, "strong_up": 2.0}
_LIQ_MAP = {"low": 0.0, "normal": 1.0, "high": 2.0}


def build_feature_vector(signal_data: dict) -> np.ndarray:
    """
    Build the fixed-width feature vector expected by the meta-labeler.

    Args:
        signal_data: flat dict that may contain any/all of the keys in
            META_FEATURE_NAMES plus raw regime strings (volatility_regime,
            trend_regime, liquidity_regime) and an ``hour`` integer.

    Returns:
        numpy array of shape (NUM_FEATURES,)
    """
    v = np.zeros(NUM_FEATURES, dtype=np.float64)

    # Strategy scores
    v[0] = float(signal_data.get("momentum_score", 0.0))
    v[1] = float(signal_data.get("z_score", 0.0))
    v[2] = float(signal_data.get("factor_score", 0.0))
    v[3] = float(signal_data.get("carry_score", 0.0))

    # Signal metadata
    v[4] = float(signal_data.get("confidence", 0.5))
    v[5] = float(signal_data.get("risk_reward_ratio", 1.0))

    # Technicals
    v[6] = float(signal_data.get("rsi", 50.0))
    v[7] = float(signal_data.get("macd", 0.0))
    v[8] = float(signal_data.get("volume_ratio", 1.0))
    v[9] = float(signal_data.get("atr_pct", 0.02))

    # Regime (accept either pre-encoded numeric or raw string)
    v[10] = _encode_regime(signal_data, "regime_volatility", "volatility_regime", _VOL_MAP, 1.0)
    v[11] = _encode_regime(signal_data, "regime_trend", "trend_regime", _TREND_MAP, 0.0)
    v[12] = _encode_regime(signal_data, "regime_liquidity", "liquidity_regime", _LIQ_MAP, 1.0)

    # Cyclical hour encoding
    hour = float(signal_data.get("hour", 12))
    v[13] = np.sin(2 * np.pi * hour / 24.0)
    v[14] = np.cos(2 * np.pi * hour / 24.0)

    return v


def _encode_regime(data: dict, numeric_key: str, string_key: str,
                   mapping: dict, default: float) -> float:
    """Resolve a regime field from either its numeric or string form."""
    # Already numeric?
    val = data.get(numeric_key)
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            pass
    # Raw string?
    raw = data.get(string_key)
    if raw is not None and isinstance(raw, str):
        return mapping.get(raw, default)
    return default


# ═══════════════════════════════════════════════════════════
#  Purged K-Fold Cross-Validation
# ═══════════════════════════════════════════════════════════

class PurgedKFold:
    """
    K-Fold with purging and embargo to prevent look-ahead bias.

    For each fold split, samples within ``embargo_td`` of any test-fold
    boundary are removed from the training set.  This prevents information
    leakage when labels are computed over overlapping price windows.

    Reference: López de Prado, *AFML*, §7.4
    """

    def __init__(self, n_splits: int = 5, embargo_days: int = 2):
        self.n_splits = n_splits
        self.embargo_days = embargo_days

    def split(self, timestamps: np.ndarray):
        """
        Yield (train_idx, test_idx) arrays with purged training indices.

        Args:
            timestamps: array of datetime-like values aligned with feature rows
        """
        n = len(timestamps)
        if n < self.n_splits * 2:
            raise ValueError(f"Too few samples ({n}) for {self.n_splits}-fold CV")

        ts = pd.to_datetime(timestamps)
        embargo_td = pd.Timedelta(days=self.embargo_days)

        kf = KFold(n_splits=self.n_splits, shuffle=False)
        for train_idx, test_idx in kf.split(np.arange(n)):
            test_start = ts[test_idx[0]]
            test_end = ts[test_idx[-1]]

            # Purge: remove training samples within embargo of test boundaries
            purge_mask = np.ones(len(train_idx), dtype=bool)
            for i, ti in enumerate(train_idx):
                t = ts[ti]
                if (test_start - embargo_td) <= t <= (test_end + embargo_td):
                    purge_mask[i] = False

            purged_train = train_idx[purge_mask]
            if len(purged_train) == 0:
                logger.warning("Purging removed all training samples in a fold — skipping")
                continue

            yield purged_train, test_idx


# ═══════════════════════════════════════════════════════════
#  MetaLabeler
# ═══════════════════════════════════════════════════════════

class MetaLabeler:
    """
    Per-strategy meta-labeling classifier.

    Usage::

        meta = MetaLabeler(model_dir="data/models")

        # Training (typically called from daily_retrain)
        meta.fit("momentum", X_features, y_outcomes, timestamps=ts)

        # Inference (called in signal loop)
        prob, size_mult = meta.predict("momentum", signal_features)
        if not meta.should_trade("momentum", signal_features, min_prob=0.55):
            continue  # skip this signal

    Models are persisted per-strategy so they survive restarts.
    """

    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache: strategy_name → fitted model
        self._models: Dict[str, object] = {}
        # Metadata per strategy (accuracy, train date, n_samples, etc.)
        self._meta: Dict[str, dict] = {}

        # Load any persisted models
        self._load_all()

    # ── Persistence ──────────────────────────────────────────

    def _model_path(self, strategy_name: str) -> Path:
        return self.model_dir / f"meta_{strategy_name}.pkl"

    def _load_all(self):
        """Load all persisted meta-label models from disk."""
        for path in self.model_dir.glob("meta_*.pkl"):
            strategy = path.stem.replace("meta_", "")
            try:
                with open(path, "rb") as f:
                    payload = pickle.load(f)
                self._models[strategy] = payload["model"]
                self._meta[strategy] = payload.get("meta", {})
                logger.info(
                    f"Loaded meta-labeler for '{strategy}' "
                    f"(trained {self._meta[strategy].get('trained_at', '?')}, "
                    f"accuracy={self._meta[strategy].get('accuracy', 0):.1%})"
                )
            except Exception as e:
                logger.error(f"Failed to load meta-labeler for '{strategy}': {e}")

    def _save(self, strategy_name: str):
        """Persist a single strategy's model to disk."""
        path = self._model_path(strategy_name)
        payload = {
            "model": self._models[strategy_name],
            "meta": self._meta.get(strategy_name, {}),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"Saved meta-labeler for '{strategy_name}' → {path}")

    # ── Training ─────────────────────────────────────────────

    def fit(self, strategy_name: str,
            X_features: np.ndarray,
            y_outcomes: np.ndarray,
            timestamps: Optional[np.ndarray] = None,
            n_splits: int = 5,
            embargo_days: int = 2) -> dict:
        """
        Train a meta-labeling model for a single strategy.

        Args:
            strategy_name: e.g. "momentum", "mean_reversion"
            X_features: array of shape (n_samples, NUM_FEATURES)
            y_outcomes: binary array (1 = signal was profitable, 0 = not)
            timestamps: datetime array for purged CV (optional but recommended)
            n_splits: number of CV folds
            embargo_days: embargo window in days around fold boundaries

        Returns:
            dict with training metrics (accuracy, precision, recall, f1,
            n_samples, n_positive, feature_importances)
        """
        n_samples = len(X_features)
        n_pos = int(y_outcomes.sum())
        logger.info(
            f"Training meta-labeler for '{strategy_name}': "
            f"{n_samples} samples, {n_pos} positive ({n_pos/max(n_samples,1):.1%})"
        )

        if n_samples < 30:
            msg = f"Too few samples ({n_samples}) to train meta-labeler for '{strategy_name}'"
            logger.warning(msg)
            return {"status": "skipped", "reason": msg}

        # ── Cross-validated evaluation ──
        cv_scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}

        if timestamps is not None and len(timestamps) == n_samples:
            cv = PurgedKFold(n_splits=n_splits, embargo_days=embargo_days)
            try:
                folds = list(cv.split(timestamps))
            except ValueError as e:
                logger.warning(f"Purged CV failed ({e}), falling back to standard KFold")
                folds = list(KFold(n_splits=n_splits, shuffle=False).split(X_features))
        else:
            folds = list(KFold(n_splits=n_splits, shuffle=False).split(X_features))

        for train_idx, test_idx in folds:
            X_tr, X_te = X_features[train_idx], X_features[test_idx]
            y_tr, y_te = y_outcomes[train_idx], y_outcomes[test_idx]

            fold_model = self._build_model()
            fold_model.fit(X_tr, y_tr)

            y_pred = fold_model.predict(X_te)
            cv_scores["accuracy"].append(accuracy_score(y_te, y_pred))
            cv_scores["precision"].append(precision_score(y_te, y_pred, zero_division=0))
            cv_scores["recall"].append(recall_score(y_te, y_pred, zero_division=0))
            cv_scores["f1"].append(f1_score(y_te, y_pred, zero_division=0))

        avg_metrics = {k: float(np.mean(v)) for k, v in cv_scores.items()}

        # ── Final model trained on full dataset ──
        final_model = self._build_model()
        final_model.fit(X_features, y_outcomes)

        # Feature importances
        importances = self._get_feature_importances(final_model)

        # Store
        self._models[strategy_name] = final_model
        self._meta[strategy_name] = {
            "trained_at": datetime.utcnow().isoformat(),
            "n_samples": n_samples,
            "n_positive": n_pos,
            "positive_rate": round(n_pos / max(n_samples, 1), 4),
            "cv_folds": len(folds),
            "embargo_days": embargo_days,
            "backend": "lightgbm" if HAS_LGB else "sklearn_gbc",
            **avg_metrics,
        }
        self._save(strategy_name)

        logger.info(
            f"Meta-labeler '{strategy_name}' trained — "
            f"CV accuracy={avg_metrics['accuracy']:.1%}, "
            f"precision={avg_metrics['precision']:.1%}, "
            f"recall={avg_metrics['recall']:.1%}"
        )

        return {
            "status": "trained",
            "strategy": strategy_name,
            **self._meta[strategy_name],
            "feature_importances": importances,
        }

    def _build_model(self):
        """Instantiate the classifier (LightGBM if available, else sklearn)."""
        if HAS_LGB:
            return lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=10,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
            )
        else:
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=10,
                random_state=42,
            )

    @staticmethod
    def _get_feature_importances(model) -> Dict[str, float]:
        """Extract feature importances regardless of backend."""
        try:
            raw = model.feature_importances_
            total = raw.sum() or 1.0
            return {
                META_FEATURE_NAMES[i]: round(float(raw[i] / total), 4)
                for i in range(min(len(raw), NUM_FEATURES))
            }
        except Exception:
            return {}

    # ── Prediction ───────────────────────────────────────────

    def predict(self, strategy_name: str,
                X_features: np.ndarray) -> Tuple[float, float]:
        """
        Predict meta-label for a signal.

        Args:
            strategy_name: strategy that produced the signal
            X_features: feature vector of shape (NUM_FEATURES,) or (1, NUM_FEATURES)

        Returns:
            (probability, side_size):
                probability — P(signal is profitable), in [0, 1]
                side_size   — suggested position-size multiplier (= probability),
                              so higher-confidence signals get larger allocations
        """
        if strategy_name not in self._models:
            # No model trained yet — pass through with neutral probability
            logger.debug(f"No meta-labeler for '{strategy_name}', returning default 0.5")
            return 0.5, 1.0

        X = np.asarray(X_features, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        model = self._models[strategy_name]
        try:
            prob = float(model.predict_proba(X)[0, 1])
        except Exception as e:
            logger.error(f"Meta-label prediction failed for '{strategy_name}': {e}")
            return 0.5, 1.0

        # Position-size multiplier: linearly scale between 0 and 1 based on
        # probability.  Below 0.5 the multiplier is < 1 (reduce size);
        # above 0.5 it is > 1 (up to 1.0 maximum).
        side_size = min(max(prob, 0.0), 1.0)

        return round(prob, 4), round(side_size, 4)

    def should_trade(self, strategy_name: str,
                     signal_features: np.ndarray,
                     min_prob: float = 0.5) -> bool:
        """
        Gate function: should this signal be executed?

        Args:
            strategy_name: strategy that produced the signal
            signal_features: feature vector (raw array or dict-built via
                ``build_feature_vector``)
            min_prob: minimum probability threshold to allow the trade

        Returns:
            True if the meta-labeler approves the trade (or has no model),
            False if it predicts the signal is a false positive.
        """
        prob, _ = self.predict(strategy_name, signal_features)

        approved = prob >= min_prob
        if not approved:
            logger.info(
                f"Meta-label REJECT '{strategy_name}': "
                f"P(profit)={prob:.2%} < threshold {min_prob:.2%}"
            )
        else:
            logger.debug(
                f"Meta-label APPROVE '{strategy_name}': P(profit)={prob:.2%}"
            )
        return approved

    # ── Convenience: build features + predict in one call ────

    def evaluate_signal(self, strategy_name: str,
                        signal_data: dict,
                        min_prob: float = 0.5) -> dict:
        """
        All-in-one helper: build features, predict, and decide.

        Args:
            strategy_name: e.g. "momentum"
            signal_data: flat dict with scores, technicals, regime, hour
            min_prob: gate threshold

        Returns:
            dict with keys: approved, probability, side_size, features
        """
        features = build_feature_vector(signal_data)
        prob, side_size = self.predict(strategy_name, features)
        approved = prob >= min_prob

        if not approved:
            logger.info(
                f"Meta-label REJECT '{strategy_name}': "
                f"P(profit)={prob:.2%} < {min_prob:.2%}"
            )

        return {
            "approved": approved,
            "probability": prob,
            "side_size": side_size,
            "features": features.tolist(),
        }

    # ── Status / Diagnostics ─────────────────────────────────

    def get_status(self) -> dict:
        """Summary of all trained meta-labelers for monitoring."""
        status = {
            "backend": "lightgbm" if HAS_LGB else "sklearn_gbc",
            "strategies": {},
        }
        for name, meta in self._meta.items():
            status["strategies"][name] = {
                "trained_at": meta.get("trained_at"),
                "n_samples": meta.get("n_samples", 0),
                "accuracy": meta.get("accuracy", 0),
                "precision": meta.get("precision", 0),
                "recall": meta.get("recall", 0),
                "f1": meta.get("f1", 0),
                "positive_rate": meta.get("positive_rate", 0),
            }
        return status

    def has_model(self, strategy_name: str) -> bool:
        """Check whether a trained model exists for a given strategy."""
        return strategy_name in self._models
