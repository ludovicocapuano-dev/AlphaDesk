"""
AlphaDesk — Self-Learning ML Ensemble
Inspired by Chris Turner's evolutionary trading bot architecture.

Architecture:
1. Feature pipeline normalizes regime + strategy signals into fixed-width tensors
2. PyTorch feedforward network predicts P(profitable) at 1h horizon
3. Shadow model trains in parallel, promotes to production if better
4. Daily retraining at 03:15 UTC with latest labeled outcomes
5. Drift monitor alerts when feature distributions shift

The ensemble acts as a meta-model: it learns WHEN each strategy works
based on regime fingerprint, signal features, and historical outcomes.
"""

import copy
import hashlib
import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("alphadesk.ml_ensemble")

# ── Try to import PyTorch; fall back to numpy-only mode ──
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed — ML ensemble will use fallback mode")


# ═══════════════════════════════════════════════════════════
#  Feature Pipeline
# ═══════════════════════════════════════════════════════════

class FeaturePipeline:
    """
    Normalizes raw signal + regime data into a fixed-width feature vector.

    Features (in order):
    [0-3]   Strategy signals: momentum_score, mr_zscore, factor_score, fx_carry_score
    [4-7]   Signal metadata: confidence, risk_reward, atr_pct, volume_ratio
    [8-12]  Regime encoding: volatility(0-3), trend(-2 to +2), liquidity(0-2),
            rate(-1 to +1), correlation(0-2)
    [13-17] Technical: rsi_norm, macd_norm, bb_position, momentum_3m, sma_cross
    [18-19] Time features: hour_sin, hour_cos (cyclical encoding)
    [20-22] AFML features: close_ffd (fractional diff), ffd_zscore, cusum_event

    Total: 23 features
    """

    FEATURE_DIM = 23

    # Regime category → numeric encoding
    VOL_MAP = {"low": 0, "medium": 1, "high": 2, "extreme": 3}
    TREND_MAP = {"strong_down": -2, "weak_down": -1, "ranging": 0, "weak_up": 1, "strong_up": 2}
    LIQ_MAP = {"low": 0, "normal": 1, "high": 2}
    RATE_MAP = {"easing": -1, "neutral": 0, "tightening": 1}
    CORR_MAP = {"normal": 0, "elevated": 1, "crisis": 2}

    def __init__(self):
        # Running stats for normalization (Welford's online algorithm)
        self.running_mean = np.zeros(self.FEATURE_DIM)
        self.running_var = np.ones(self.FEATURE_DIM)
        self.count = 0

    def extract_features(self, signal_data: dict, regime_data: dict) -> np.ndarray:
        """
        Extract a 20-dim feature vector from signal + regime data.

        Args:
            signal_data: dict with strategy scores, confidence, technical indicators
            regime_data: dict from RegimeFingerprint.to_dict()

        Returns:
            numpy array of shape (20,)
        """
        features = np.zeros(self.FEATURE_DIM)

        # [0-3] Strategy signals (raw scores, will be normalized)
        features[0] = signal_data.get("momentum_score", 0)
        features[1] = signal_data.get("mr_zscore", 0)
        features[2] = signal_data.get("factor_score", 0)
        features[3] = signal_data.get("fx_carry_score", 0)

        # [4-7] Signal metadata
        features[4] = signal_data.get("confidence", 0.5)
        features[5] = min(signal_data.get("risk_reward", 1.0), 5.0) / 5.0  # Clip & normalize
        features[6] = min(signal_data.get("atr_pct", 0.02), 0.10) / 0.10
        features[7] = min(signal_data.get("volume_ratio", 1.0), 5.0) / 5.0

        # [8-12] Regime encoding
        features[8] = self.VOL_MAP.get(regime_data.get("volatility_regime", "medium"), 1) / 3.0
        features[9] = (self.TREND_MAP.get(regime_data.get("trend_regime", "ranging"), 0) + 2) / 4.0
        features[10] = self.LIQ_MAP.get(regime_data.get("liquidity_regime", "normal"), 1) / 2.0
        features[11] = (self.RATE_MAP.get(regime_data.get("rate_regime", "neutral"), 0) + 1) / 2.0
        features[12] = self.CORR_MAP.get(regime_data.get("correlation_regime", "normal"), 0) / 2.0

        # [13-17] Technical indicators
        features[13] = min(max(signal_data.get("rsi", 50), 0), 100) / 100.0
        features[14] = np.tanh(signal_data.get("macd", 0) * 10)  # Normalize MACD
        features[15] = min(max(signal_data.get("bb_position", 0.5), 0), 1.0)
        features[16] = np.tanh(signal_data.get("momentum_3m", 0))
        features[17] = 1.0 if signal_data.get("sma_cross", False) else 0.0

        # [18-19] Time features (cyclical encoding of hour)
        hour = signal_data.get("hour", 12)
        features[18] = np.sin(2 * np.pi * hour / 24)
        features[19] = np.cos(2 * np.pi * hour / 24)

        # [20-22] AFML features (López de Prado)
        # FFD close: fractionally differentiated price (stationarity + memory)
        features[20] = np.tanh(signal_data.get("close_ffd", 0) * 100)
        # FFD z-score: how far FFD is from its rolling mean
        features[21] = np.tanh(signal_data.get("ffd_zscore", 0))
        # CUSUM event: 1 if recent CUSUM filter triggered, 0 otherwise
        features[22] = 1.0 if signal_data.get("cusum_event", False) else 0.0

        return features

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Z-score normalization using running statistics."""
        if self.count < 100:
            return features  # Not enough data for meaningful normalization
        std = np.sqrt(self.running_var + 1e-8)
        return (features - self.running_mean) / std

    def update_stats(self, features: np.ndarray):
        """Update running mean/variance (Welford's algorithm)."""
        self.count += 1
        delta = features - self.running_mean
        self.running_mean += delta / self.count
        delta2 = features - self.running_mean
        self.running_var += (delta * delta2 - self.running_var) / self.count

    def save(self, path: str):
        """Persist normalization stats."""
        np.savez(path,
                 mean=self.running_mean,
                 var=self.running_var,
                 count=np.array([self.count]))

    def load(self, path: str):
        """Load normalization stats."""
        if os.path.exists(path):
            data = np.load(path)
            self.running_mean = data["mean"]
            self.running_var = data["var"]
            self.count = int(data["count"][0])
            logger.info(f"Loaded feature stats (n={self.count})")


# ═══════════════════════════════════════════════════════════
#  PyTorch Model
# ═══════════════════════════════════════════════════════════

if HAS_TORCH:
    class AlphaNet(nn.Module):
        """
        Feedforward network for trade outcome prediction.

        Architecture: 20 → 64 → 32 → 16 → 1 (sigmoid)
        - Dropout for regularization
        - BatchNorm for training stability
        - LeakyReLU to avoid dead neurons

        Output: P(trade is profitable at 1h horizon)
        """

        def __init__(self, input_dim: int = 23, dropout: float = 0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),

                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),

                nn.Linear(32, 16),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout * 0.5),

                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════
#  Drift Monitor
# ═══════════════════════════════════════════════════════════

class DriftMonitor:
    """
    Monitors feature distribution drift to detect when the model
    is operating outside its training distribution.

    Uses Population Stability Index (PSI) — a standard metric in
    production ML systems for detecting distribution shift.

    PSI > 0.1  → minor drift (log warning)
    PSI > 0.25 → major drift (trigger retrain)
    """

    PSI_WARN = 0.10
    PSI_RETRAIN = 0.25

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.reference_distributions = None  # Set after training

    def set_reference(self, features: np.ndarray):
        """Set reference distribution from training data."""
        self.reference_distributions = []
        for col in range(features.shape[1]):
            hist, bin_edges = np.histogram(features[:, col], bins=self.n_bins)
            hist = hist / hist.sum() + 1e-6  # Avoid zero
            self.reference_distributions.append((hist, bin_edges))
        logger.info(f"Drift reference set from {features.shape[0]} samples")

    def check_drift(self, recent_features: np.ndarray) -> Tuple[float, bool, bool]:
        """
        Check for distribution drift in recent features.

        Returns:
            (psi_score, has_warning, needs_retrain)
        """
        if self.reference_distributions is None or len(recent_features) < 50:
            return 0.0, False, False

        psi_scores = []
        for col in range(min(recent_features.shape[1], len(self.reference_distributions))):
            ref_hist, bin_edges = self.reference_distributions[col]
            # Compute histogram of recent data using same bins
            recent_hist, _ = np.histogram(recent_features[:, col], bins=bin_edges)
            recent_hist = recent_hist / recent_hist.sum() + 1e-6

            # PSI = sum((actual - expected) * ln(actual/expected))
            psi = np.sum((recent_hist - ref_hist) * np.log(recent_hist / ref_hist))
            psi_scores.append(psi)

        avg_psi = np.mean(psi_scores)
        return avg_psi, avg_psi > self.PSI_WARN, avg_psi > self.PSI_RETRAIN

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.reference_distributions, f)

    def load(self, path: str):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.reference_distributions = pickle.load(f)


# ═══════════════════════════════════════════════════════════
#  ML Ensemble (main interface)
# ═══════════════════════════════════════════════════════════

class MLEnsemble:
    """
    Self-learning ML ensemble that enhances base strategy signals.

    Lifecycle:
    1. Cold start: passes through base signals unchanged (no ML override)
    2. Data collection: logs every signal + regime with feature vector
    3. First train: after 200+ labeled outcomes, trains initial model
    4. Production: model predicts P(profitable) for each signal
    5. Daily retrain: shadow model trains, promotes if accuracy improves
    6. Drift monitoring: alerts when feature distributions shift

    The ensemble NEVER overrides a human decision or risk check.
    It only adjusts confidence scores and can veto low-probability trades.
    """

    MIN_TRAINING_SAMPLES = 200
    SHADOW_PROMOTION_THRESHOLD = 0.02  # Shadow must beat production by 2%
    CONFIDENCE_VETO_THRESHOLD = 0.30   # Veto trades below 30% predicted probability

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.feature_pipeline = FeaturePipeline()
        self.drift_monitor = DriftMonitor()

        # Model state
        self.production_model = None
        self.shadow_model = None
        self.model_version = 0
        self.last_train_time = None
        self.training_accuracy = 0.0
        self.is_active = False  # False = cold start (no model yet)

        # Recent predictions for drift monitoring
        self._recent_features = []
        self._recent_predictions = []
        self._recent_actuals = []

        # Load existing model if available
        self._load_state()

    def _load_state(self):
        """Load persisted model and stats."""
        stats_path = self.model_dir / "feature_stats.npz"
        self.feature_pipeline.load(str(stats_path))

        drift_path = self.model_dir / "drift_reference.pkl"
        self.drift_monitor.load(str(drift_path))

        model_path = self.model_dir / "production_model.pt"
        meta_path = self.model_dir / "model_meta.json"

        if HAS_TORCH and model_path.exists() and meta_path.exists():
            try:
                self.production_model = AlphaNet()
                self.production_model.load_state_dict(
                    torch.load(str(model_path), map_location="cpu", weights_only=True)
                )
                self.production_model.eval()

                with open(meta_path) as f:
                    meta = json.load(f)
                self.model_version = meta.get("version", 0)
                self.training_accuracy = meta.get("accuracy", 0)
                self.last_train_time = meta.get("last_train")
                self.is_active = True

                logger.info(
                    f"Loaded production model v{self.model_version} "
                    f"(accuracy: {self.training_accuracy:.1%})"
                )
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.production_model = None
                self.is_active = False

    # ── Prediction ──

    def predict(self, signal_data: dict, regime_data: dict) -> dict:
        """
        Predict trade outcome probability.

        Args:
            signal_data: dict with strategy scores, confidence, technicals
            regime_data: dict from RegimeFingerprint.to_dict()

        Returns:
            dict with:
                ml_probability: P(profitable) [0-1]
                ml_confidence_adj: adjusted confidence score
                ml_veto: True if model recommends NOT trading
                ml_active: whether model is active (vs cold start)
                feature_vector: the extracted features (for logging)
        """
        features = self.feature_pipeline.extract_features(signal_data, regime_data)
        self.feature_pipeline.update_stats(features)
        normalized = self.feature_pipeline.normalize(features)

        result = {
            "ml_probability": 0.5,
            "ml_confidence_adj": signal_data.get("confidence", 0.5),
            "ml_veto": False,
            "ml_active": self.is_active,
            "feature_vector": features.tolist(),
        }

        if not self.is_active or self.production_model is None or not HAS_TORCH:
            # Cold start — pass through unchanged
            return result

        # Run prediction
        try:
            self.production_model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(normalized).unsqueeze(0)
                prob = self.production_model(x).item()

            result["ml_probability"] = round(prob, 4)

            # Adjust confidence: blend base confidence with ML probability
            base_conf = signal_data.get("confidence", 0.5)
            # Weight: 60% base strategy, 40% ML (ML is supplementary, not primary)
            adjusted = 0.6 * base_conf + 0.4 * prob
            result["ml_confidence_adj"] = round(adjusted, 4)

            # Veto if ML is very bearish on this trade
            result["ml_veto"] = prob < self.CONFIDENCE_VETO_THRESHOLD

            # Track for drift monitoring
            self._recent_features.append(normalized)
            if len(self._recent_features) > 500:
                self._recent_features = self._recent_features[-500:]

            if result["ml_veto"]:
                logger.info(
                    f"ML VETO: P(profit)={prob:.2%} < {self.CONFIDENCE_VETO_THRESHOLD:.0%}"
                )

        except Exception as e:
            logger.error(f"ML prediction failed: {e}")

        return result

    # ── Training ──

    def train(self, training_df: pd.DataFrame, force: bool = False) -> dict:
        """
        Train or retrain the ensemble model.

        Args:
            training_df: DataFrame from OutcomeLabeler.get_training_data()
            force: bypass minimum sample check

        Returns:
            dict with training metrics
        """
        if not HAS_TORCH:
            return {"status": "skipped", "reason": "PyTorch not installed"}

        if len(training_df) < self.MIN_TRAINING_SAMPLES and not force:
            return {
                "status": "skipped",
                "reason": f"Need {self.MIN_TRAINING_SAMPLES} samples, have {len(training_df)}",
            }

        logger.info(f"Training ML ensemble on {len(training_df)} samples...")

        try:
            # Prepare features and labels
            X, y = self._prepare_training_data(training_df)
            if X is None:
                return {"status": "error", "reason": "Failed to prepare training data"}

            # Train-validation split (80/20, chronological)
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Train shadow model
            shadow = AlphaNet(input_dim=FeaturePipeline.FEATURE_DIM)
            train_metrics = self._train_model(shadow, X_train, y_train, X_val, y_val)

            # Evaluate shadow vs production
            shadow_acc = train_metrics["val_accuracy"]
            production_acc = self.training_accuracy

            promote = False
            if self.production_model is None:
                # First model — always promote
                promote = True
                logger.info(f"First model trained — accuracy: {shadow_acc:.1%}")
            elif shadow_acc > production_acc + self.SHADOW_PROMOTION_THRESHOLD:
                promote = True
                logger.info(
                    f"Shadow promoted: {shadow_acc:.1%} > "
                    f"{production_acc:.1%} + {self.SHADOW_PROMOTION_THRESHOLD:.0%}"
                )
            else:
                logger.info(
                    f"Shadow NOT promoted: {shadow_acc:.1%} vs "
                    f"production {production_acc:.1%}"
                )

            if promote:
                self.production_model = shadow
                self.model_version += 1
                self.training_accuracy = shadow_acc
                self.last_train_time = datetime.utcnow().isoformat()
                self.is_active = True
                self._save_state(X)

            return {
                "status": "promoted" if promote else "shadow_kept",
                "version": self.model_version,
                "shadow_accuracy": round(shadow_acc, 4),
                "production_accuracy": round(production_acc, 4),
                "samples": len(training_df),
                "train_loss": round(train_metrics["train_loss"], 4),
                "val_loss": round(train_metrics["val_loss"], 4),
                "val_accuracy": round(shadow_acc, 4),
                "epochs": train_metrics["epochs"],
            }

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return {"status": "error", "reason": str(e)}

    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Convert DataFrame to feature matrix and labels."""
        try:
            feature_cols = [c for c in df.columns if c.startswith(("meta_", "regime_", "feat_"))]
            numeric_cols = ["confidence", "risk_reward_ratio",
                           "outcome_15m_pnl", "outcome_1h_pnl",
                           "outcome_4h_pnl", "outcome_24h_pnl"]

            # Build feature matrix from available columns
            all_features = []
            for _, row in df.iterrows():
                feat = np.zeros(FeaturePipeline.FEATURE_DIM)

                # Fill from available data
                feat[4] = row.get("confidence", 0.5) or 0.5
                rr = row.get("risk_reward_ratio", 1.0) or 1.0
                feat[5] = min(rr, 5.0) / 5.0

                # Regime features
                for fc in feature_cols:
                    if fc.startswith("regime_volatility"):
                        feat[8] = FeaturePipeline.VOL_MAP.get(row.get(fc, "medium"), 1) / 3.0
                    elif fc.startswith("regime_trend"):
                        feat[9] = (FeaturePipeline.TREND_MAP.get(row.get(fc, "ranging"), 0) + 2) / 4.0
                    elif fc.startswith("regime_liquidity"):
                        feat[10] = FeaturePipeline.LIQ_MAP.get(row.get(fc, "normal"), 1) / 2.0
                    elif fc.startswith("regime_rate"):
                        feat[11] = (FeaturePipeline.RATE_MAP.get(row.get(fc, "neutral"), 0) + 1) / 2.0
                    elif fc.startswith("regime_correlation"):
                        feat[12] = FeaturePipeline.CORR_MAP.get(row.get(fc, "normal"), 0) / 2.0

                # Technical features from feat_ columns
                for fc in feature_cols:
                    if fc == "feat_rsi":
                        feat[13] = min(max(float(row.get(fc, 50) or 50), 0), 100) / 100.0
                    elif fc == "feat_macd":
                        feat[14] = np.tanh(float(row.get(fc, 0) or 0) * 10)
                    elif fc == "feat_bb_position":
                        feat[15] = min(max(float(row.get(fc, 0.5) or 0.5), 0), 1.0)
                    elif fc == "feat_momentum_3m":
                        feat[16] = np.tanh(float(row.get(fc, 0) or 0))

                all_features.append(feat)

            X = np.array(all_features, dtype=np.float32)

            # Labels: 1 if 1h outcome was profitable, 0 otherwise
            y = (df["label"].fillna(0).astype(float) > 0).astype(np.float32).values

            # Update normalization stats
            for feat in X:
                self.feature_pipeline.update_stats(feat)

            # Normalize
            X_normalized = np.array([self.feature_pipeline.normalize(f) for f in X])

            return X_normalized, y

        except Exception as e:
            logger.error(f"Data preparation failed: {e}", exc_info=True)
            return None, None

    def _train_model(self, model: 'AlphaNet',
                     X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     max_epochs: int = 100, patience: int = 10,
                     lr: float = 0.001, batch_size: int = 64) -> dict:
        """
        Train a PyTorch model with early stopping.
        """
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        actual_epochs = 0

        for epoch in range(max_epochs):
            # ── Train ──
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # ── Validate ──
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
                val_acc = ((val_pred > 0.5).float() == y_val_t).float().mean().item()

            scheduler.step(val_loss)
            actual_epochs = epoch + 1

            # Early stopping
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.1%}"
                )

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final validation metrics
        model.eval()
        with torch.no_grad():
            final_pred = model(X_val_t)
            final_loss = criterion(final_pred, y_val_t).item()
            final_acc = ((final_pred > 0.5).float() == y_val_t).float().mean().item()

        return {
            "train_loss": train_loss,
            "val_loss": final_loss,
            "val_accuracy": final_acc,
            "epochs": actual_epochs,
        }

    def _save_state(self, X_train: np.ndarray = None):
        """Persist model, stats, and drift reference."""
        if self.production_model is not None and HAS_TORCH:
            torch.save(
                self.production_model.state_dict(),
                str(self.model_dir / "production_model.pt"),
            )

        meta = {
            "version": self.model_version,
            "accuracy": self.training_accuracy,
            "last_train": self.last_train_time,
            "feature_dim": FeaturePipeline.FEATURE_DIM,
            "min_samples": self.MIN_TRAINING_SAMPLES,
        }
        with open(self.model_dir / "model_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        self.feature_pipeline.save(str(self.model_dir / "feature_stats.npz"))

        if X_train is not None:
            self.drift_monitor.set_reference(X_train)
            self.drift_monitor.save(str(self.model_dir / "drift_reference.pkl"))

        logger.info(f"Model state saved (v{self.model_version})")

    # ── Drift Monitoring ──

    def check_drift(self) -> dict:
        """
        Check for feature distribution drift.

        Returns:
            dict with psi_score, warning, needs_retrain
        """
        if not self._recent_features:
            return {"psi": 0, "warning": False, "needs_retrain": False}

        recent = np.array(self._recent_features[-200:])
        psi, warning, retrain = self.drift_monitor.check_drift(recent)

        if warning:
            logger.warning(f"Feature drift detected: PSI={psi:.4f}")
        if retrain:
            logger.warning(f"MAJOR drift — retrain recommended: PSI={psi:.4f}")

        return {
            "psi": round(psi, 4),
            "warning": warning,
            "needs_retrain": retrain,
            "samples_monitored": len(self._recent_features),
        }

    # ── Status ──

    def get_status(self) -> dict:
        """Get ensemble status for monitoring/Telegram."""
        drift = self.check_drift()
        return {
            "active": self.is_active,
            "model_version": self.model_version,
            "accuracy": round(self.training_accuracy, 4),
            "last_train": self.last_train_time,
            "has_torch": HAS_TORCH,
            "predictions_made": len(self._recent_predictions),
            "drift_psi": drift["psi"],
            "drift_warning": drift["warning"],
        }
