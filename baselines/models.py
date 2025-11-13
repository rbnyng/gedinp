"""
Baseline models for GEDI biomass prediction.

Implements:
- Random Forest
- XGBoost
- Linear Regression
- Inverse Distance Weighting (IDW)
- MLP with MC Dropout
- Ensemble MLP
"""

import numpy as np
from typing import Tuple, Optional, List
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize_scalar
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


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

        # Temperature scaling for uncertainty calibration
        self.temperature = 1.0

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

        # Temperature scaling for uncertainty calibration
        self.temperature = 1.0

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


class MLPNet(nn.Module):
    """
    Simple MLP architecture for biomass prediction.

    Used by both MLPBaseline (with MC Dropout) and EnsembleMLPBaseline.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        dropout_rate: float = 0.0
    ):
        """
        Initialize MLP network.

        Args:
            input_dim: Input feature dimension (coords + flattened embeddings)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate (0.0 = no dropout)
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer (predict mean)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, input_dim) input features

        Returns:
            (batch, 1) predictions
        """
        return self.network(x)


class MLPBaseline:
    """
    MLP with Monte Carlo Dropout for biomass prediction.

    Uses flattened embeddings + coordinates as features to predict log(AGBD).
    Uncertainty is estimated via MC Dropout at test time.
    """

    def __init__(
        self,
        hidden_dims: List[int] = [512, 256, 128],
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        n_epochs: int = 100,
        mc_samples: int = 100,
        random_state: int = 42,
        device: Optional[str] = None
    ):
        """
        Initialize MLP with MC Dropout.

        Args:
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for MC Dropout
            learning_rate: Learning rate for AdamW optimizer
            weight_decay: L2 regularization strength
            batch_size: Batch size for training
            n_epochs: Number of training epochs
            mc_samples: Number of MC samples for uncertainty estimation
            random_state: Random seed
            device: Device to use ('cuda' or 'cpu', None=auto)
        """
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.mc_samples = mc_samples
        self.random_state = random_state

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def _prepare_features(
        self,
        coords: np.ndarray,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Prepare features for MLP.

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
        val_coords: Optional[np.ndarray] = None,
        val_embeddings: Optional[np.ndarray] = None,
        val_agbd: Optional[np.ndarray] = None,
        verbose: bool = True
    ):
        """
        Train the MLP model.

        Args:
            coords: (N, 2) training coordinates
            embeddings: (N, H, W, C) training embeddings
            agbd: (N,) training AGBD values (already log-transformed)
            val_coords: Optional validation coordinates
            val_embeddings: Optional validation embeddings
            val_agbd: Optional validation AGBD values
            verbose: Print training progress
        """
        # Prepare features
        X_train = self._prepare_features(coords, embeddings)
        y_train = agbd

        # Create dataset and dataloader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(1)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Initialize model
        input_dim = X_train.shape[1]
        self.model = MLPNet(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        # Optimizer and loss
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()

        # Training loop
        self.model.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * X_batch.size(0)

            epoch_loss /= len(train_dataset)

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {epoch_loss:.6f}")

    def predict(
        self,
        coords: np.ndarray,
        embeddings: np.ndarray,
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict AGBD using MC Dropout.

        Args:
            coords: (N, 2) query coordinates
            embeddings: (N, H, W, C) query embeddings
            return_std: If True, return std estimates via MC Dropout

        Returns:
            predictions: (N,) predicted AGBD values
            std: (N,) prediction std (from MC samples) if return_std=True, else None
        """
        X = self._prepare_features(coords, embeddings)
        X_tensor = torch.FloatTensor(X).to(self.device)

        if return_std:
            # MC Dropout: multiple forward passes with dropout enabled
            self.model.train()  # Keep dropout active
            mc_predictions = []

            with torch.no_grad():
                for _ in range(self.mc_samples):
                    pred = self.model(X_tensor)
                    mc_predictions.append(pred.cpu().numpy())

            mc_predictions = np.array(mc_predictions)  # (mc_samples, N, 1)
            mc_predictions = mc_predictions.squeeze(-1)  # (mc_samples, N)

            # Mean and std across MC samples
            predictions = np.mean(mc_predictions, axis=0)
            stds = np.std(mc_predictions, axis=0)

            return predictions, stds
        else:
            # Single forward pass
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor).cpu().numpy().squeeze(-1)
            return predictions, None


class EnsembleMLPBaseline:
    """
    Ensemble of MLPs for biomass prediction.

    Trains K independent MLPs with different random initializations.
    Uncertainty is estimated from the variance across ensemble members.
    """

    def __init__(
        self,
        n_models: int = 3,
        hidden_dims: List[int] = [512, 256, 128],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        n_epochs: int = 100,
        bootstrap: bool = True,
        random_state: int = 42,
        device: Optional[str] = None
    ):
        """
        Initialize ensemble of MLPs.

        Args:
            n_models: Number of models in ensemble
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for AdamW optimizer
            weight_decay: L2 regularization strength (higher than MC Dropout since no dropout)
            batch_size: Batch size for training
            n_epochs: Number of training epochs
            bootstrap: If True, use bootstrap sampling for each ensemble member
            random_state: Base random seed
            device: Device to use ('cuda' or 'cpu', None=auto)
        """
        self.n_models = n_models
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.bootstrap = bootstrap
        self.random_state = random_state

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.models = []

    def _prepare_features(
        self,
        coords: np.ndarray,
        embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Prepare features for MLP.

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
        val_coords: Optional[np.ndarray] = None,
        val_embeddings: Optional[np.ndarray] = None,
        val_agbd: Optional[np.ndarray] = None,
        verbose: bool = True
    ):
        """
        Train ensemble of MLP models.

        Args:
            coords: (N, 2) training coordinates
            embeddings: (N, H, W, C) training embeddings
            agbd: (N,) training AGBD values (already log-transformed)
            val_coords: Optional validation coordinates
            val_embeddings: Optional validation embeddings
            val_agbd: Optional validation AGBD values
            verbose: Print training progress
        """
        # Prepare features
        X_train = self._prepare_features(coords, embeddings)
        y_train = agbd
        n_samples = X_train.shape[0]

        # Train each model with different random seed
        for i in range(self.n_models):
            if verbose:
                print(f"\nTraining ensemble member {i+1}/{self.n_models}...")

            # Set different random seed for each model
            model_seed = self.random_state + i
            torch.manual_seed(model_seed)
            np.random.seed(model_seed)

            # Bootstrap sampling for diversity (if enabled)
            if self.bootstrap:
                # Sample with replacement
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_train_bootstrap = X_train[bootstrap_indices]
                y_train_bootstrap = y_train[bootstrap_indices]
                if verbose:
                    unique_samples = len(np.unique(bootstrap_indices))
                    print(f"  Bootstrap: {unique_samples}/{n_samples} unique samples")
            else:
                X_train_bootstrap = X_train
                y_train_bootstrap = y_train

            # Create dataset and dataloader
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_bootstrap),
                torch.FloatTensor(y_train_bootstrap).unsqueeze(1)
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )

            # Initialize model (no dropout for ensemble)
            input_dim = X_train.shape[1]
            model = MLPNet(
                input_dim=input_dim,
                hidden_dims=self.hidden_dims,
                dropout_rate=0.0  # No dropout for ensemble
            ).to(self.device)

            # Optimizer and loss (with L2 regularization via weight_decay)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            criterion = nn.MSELoss()

            # Training loop
            model.train()
            for epoch in range(self.n_epochs):
                epoch_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    optimizer.zero_grad()
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item() * X_batch.size(0)

                epoch_loss /= len(train_dataset)

                if verbose and (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{self.n_epochs}, Loss: {epoch_loss:.6f}")

            self.models.append(model)

    def predict(
        self,
        coords: np.ndarray,
        embeddings: np.ndarray,
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict AGBD using ensemble.

        Args:
            coords: (N, 2) query coordinates
            embeddings: (N, H, W, C) query embeddings
            return_std: If True, return std estimates from ensemble variance

        Returns:
            predictions: (N,) predicted AGBD values
            std: (N,) prediction std (from ensemble) if return_std=True, else None
        """
        X = self._prepare_features(coords, embeddings)
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Get predictions from all ensemble members
        ensemble_predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(X_tensor)
                ensemble_predictions.append(pred.cpu().numpy())

        ensemble_predictions = np.array(ensemble_predictions)  # (n_models, N, 1)
        ensemble_predictions = ensemble_predictions.squeeze(-1)  # (n_models, N)

        # Mean and std across ensemble
        predictions = np.mean(ensemble_predictions, axis=0)

        if return_std:
            stds = np.std(ensemble_predictions, axis=0)
            return predictions, stds
        else:
            return predictions, None
