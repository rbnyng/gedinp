"""
Baseline models for GEDI biomass prediction.

Implements:
- Random Forest
- XGBoost
- Inverse Distance Weighting (IDW)
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


class RandomForestBaseline:
    """
    Random Forest baseline for biomass prediction.

    Uses flattened embeddings + coordinates as features to predict log(AGBD).
    Supports quantile regression for uncertainty estimation.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_jobs: int = -1,
        random_state: int = 42,
        quantiles: bool = False
    ):
        """
        Initialize Random Forest model.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random seed
            quantiles: If True, store all tree predictions for quantile estimation
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.quantiles = quantiles

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state
        )

    def _prepare_features(
        self,
        coords: np.ndarray,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Prepare features for Random Forest.

        Args:
            coords: (N, 2) array of [lon, lat]
            embeddings: (N, H, W, C) array of embedding patches

        Returns:
            (N, 2 + H*W*C) flattened feature array
        """
        # Flatten embeddings
        n_samples = embeddings.shape[0]
        embeddings_flat = embeddings.reshape(n_samples, -1)

        # Concatenate coords + embeddings
        features = np.concatenate([coords, embeddings_flat], axis=1)
        return features

    def fit(
        self,
        coords: np.ndarray,
        embeddings: np.ndarray,
        agbd: np.ndarray
    ):
        """
        Train the Random Forest model.

        Args:
            coords: (N, 2) training coordinates
            embeddings: (N, H, W, C) training embeddings
            agbd: (N,) training AGBD values (already log-transformed)
        """
        X = self._prepare_features(coords, embeddings)
        self.model.fit(X, agbd)

    def predict(
        self,
        coords: np.ndarray,
        embeddings: np.ndarray,
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict AGBD values.

        Args:
            coords: (N, 2) query coordinates
            embeddings: (N, H, W, C) query embeddings
            return_std: If True, return standard deviation estimates

        Returns:
            predictions: (N,) predicted AGBD values
            std: (N,) prediction std (from tree variance) if return_std=True, else None
        """
        X = self._prepare_features(coords, embeddings)
        predictions = self.model.predict(X)

        if return_std:
            # Get predictions from all trees
            tree_predictions = np.array([
                tree.predict(X) for tree in self.model.estimators_
            ])  # (n_trees, N)

            # Standard deviation across trees
            std = np.std(tree_predictions, axis=0)
            return predictions, std
        else:
            return predictions, None


class XGBoostBaseline:
    """
    XGBoost baseline for biomass prediction.

    Uses flattened embeddings + coordinates as features to predict log(AGBD).
    Supports quantile regression for uncertainty estimation.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_jobs: int = -1,
        random_state: int = 42,
        quantile_alpha: float = 0.95
    ):
        """
        Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of features
            n_jobs: Number of parallel jobs
            random_state: Random seed
            quantile_alpha: Alpha for quantile regression (default: 0.95 for 95% interval)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.quantile_alpha = quantile_alpha

        # Main model (mean prediction)
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_jobs=n_jobs,
            random_state=random_state,
            objective='reg:squarederror'
        )

        # Quantile models for uncertainty (upper and lower bounds)
        self.model_upper = None
        self.model_lower = None

    def _prepare_features(
        self,
        coords: np.ndarray,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Prepare features for XGBoost.

        Args:
            coords: (N, 2) array of [lon, lat]
            embeddings: (N, H, W, C) array of embedding patches

        Returns:
            (N, 2 + H*W*C) flattened feature array
        """
        # Flatten embeddings
        n_samples = embeddings.shape[0]
        embeddings_flat = embeddings.reshape(n_samples, -1)

        # Concatenate coords + embeddings
        features = np.concatenate([coords, embeddings_flat], axis=1)
        return features

    def fit(
        self,
        coords: np.ndarray,
        embeddings: np.ndarray,
        agbd: np.ndarray,
        fit_quantiles: bool = True
    ):
        """
        Train the XGBoost model.

        Args:
            coords: (N, 2) training coordinates
            embeddings: (N, H, W, C) training embeddings
            agbd: (N,) training AGBD values (already log-transformed)
            fit_quantiles: If True, also fit quantile regression models for uncertainty
        """
        X = self._prepare_features(coords, embeddings)

        # Fit main model
        self.model.fit(X, agbd)

        # Fit quantile models for uncertainty estimation
        if fit_quantiles:
            self.model_upper = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                objective='reg:quantileerror',
                quantile_alpha=self.quantile_alpha
            )

            self.model_lower = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                objective='reg:quantileerror',
                quantile_alpha=1.0 - self.quantile_alpha
            )

            self.model_upper.fit(X, agbd)
            self.model_lower.fit(X, agbd)

    def predict(
        self,
        coords: np.ndarray,
        embeddings: np.ndarray,
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict AGBD values.

        Args:
            coords: (N, 2) query coordinates
            embeddings: (N, H, W, C) query embeddings
            return_std: If True, return std estimates from quantiles

        Returns:
            predictions: (N,) predicted AGBD values
            std: (N,) prediction std (from quantile range) if return_std=True, else None
        """
        X = self._prepare_features(coords, embeddings)
        predictions = self.model.predict(X)

        if return_std and self.model_upper is not None and self.model_lower is not None:
            # Predict upper and lower quantiles
            upper = self.model_upper.predict(X)
            lower = self.model_lower.predict(X)

            # Approximate std from quantile range
            # For 95% interval, std ≈ (upper - lower) / (2 * 1.96)
            std = (upper - lower) / (2 * 1.96)
            return predictions, std
        else:
            return predictions, None


class IDWBaseline:
    """
    Inverse Distance Weighting (IDW) baseline.

    Pure spatial interpolation that ignores embeddings - only uses coordinates
    and AGBD values. This baseline shows the value-add of satellite embeddings.
    """

    def __init__(
        self,
        power: float = 2.0,
        n_neighbors: int = 10,
        epsilon: float = 1e-10
    ):
        """
        Initialize IDW model.

        Args:
            power: Power parameter for distance weighting (higher = more local)
            n_neighbors: Number of nearest neighbors to use
            epsilon: Small constant to avoid division by zero
        """
        self.power = power
        self.n_neighbors = n_neighbors
        self.epsilon = epsilon

        # Store training data
        self.train_coords = None
        self.train_agbd = None

    def fit(
        self,
        coords: np.ndarray,
        embeddings: np.ndarray,  # Ignored for IDW
        agbd: np.ndarray
    ):
        """
        Store training data for IDW interpolation.

        Args:
            coords: (N, 2) training coordinates
            embeddings: (N, H, W, C) training embeddings (IGNORED)
            agbd: (N,) training AGBD values (already log-transformed)
        """
        self.train_coords = coords
        self.train_agbd = agbd

    def predict(
        self,
        coords: np.ndarray,
        embeddings: np.ndarray,  # Ignored for IDW
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict AGBD using inverse distance weighting.

        Args:
            coords: (N, 2) query coordinates
            embeddings: (N, H, W, C) query embeddings (IGNORED)
            return_std: If True, return std estimates

        Returns:
            predictions: (N,) predicted AGBD values
            std: (N,) prediction std (from weighted variance) if return_std=True, else None
        """
        n_query = coords.shape[0]
        predictions = np.zeros(n_query)
        stds = np.zeros(n_query) if return_std else None

        for i in range(n_query):
            query_coord = coords[i:i+1]

            # Compute distances to all training points
            distances = np.sqrt(
                np.sum((self.train_coords - query_coord) ** 2, axis=1)
            )

            # Get k nearest neighbors
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_distances = distances[nearest_indices]
            nearest_agbd = self.train_agbd[nearest_indices]

            # Compute inverse distance weights
            # Add epsilon to avoid division by zero
            weights = 1.0 / (nearest_distances ** self.power + self.epsilon)
            weights = weights / weights.sum()  # Normalize

            # Weighted prediction
            predictions[i] = np.sum(weights * nearest_agbd)

            # Weighted standard deviation
            if return_std:
                weighted_mean = predictions[i]
                weighted_var = np.sum(weights * (nearest_agbd - weighted_mean) ** 2)
                stds[i] = np.sqrt(weighted_var)

        return predictions, stds
