# Built-in
import os
import logging
import warnings
from datetime import datetime, timedelta

# Typing
from typing import Any, Dict, List, Optional, Tuple

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Progress bar
from tqdm import tqdm

# Machine Learning & Preprocessing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import (
    cross_val_score, 
    cross_validate,
    RandomizedSearchCV, 
    KFold, 
    TimeSeriesSplit, 
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel

# XGBoost & Optuna
import xgboost as xgb
import optuna

# WRDS Database
import wrds

# Deep Learning (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Statistics & Tests
from scipy import stats
from scipy.stats import qmc
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

# Suppress warnings
warnings.filterwarnings('ignore')

#Deep Learning Model
#-------------------------------------
class DeepLearningRegressor(nn.Module):
    """
    Deep neural network for regression tasks using PyTorch.

    This flexible regressor allows customization of:
    - The number and size of hidden layers
    - Dropout regularization
    - Activation functions (ReLU, Tanh, LeakyReLU, ELU)
    - Normalization (LayerNorm instead of BatchNorm for stability on small batches)

    Args:
        input_dim (int): Number of input features.
        hidden_layers (list of int): List with the size of each hidden layer.
        dropout_rate (float): Dropout probability (default 0.2).
        activation (str): Activation function, one of ['relu', 'tanh', 'leaky_relu', 'elu'].

    Raises:
        ValueError: If input_dim is not a positive integer.
    """

    def __init__(self, input_dim, hidden_layers, dropout_rate=0.2, activation='relu'):
        """
        Initialize the DeepLearningRegressor.

        - Sets up the network architecture according to specified parameters.
        - Applies input validation.
        - Constructs the sequence of layers, including Linear, LayerNorm, activation, and Dropout.
        - Final output layer maps to a single value (regression).
        - Initializes weights with Xavier/Glorot initialization for better convergence.
        """
        super(DeepLearningRegressor, self).__init__()
        
        # Safety check for input dimensions
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim doit être un entier positif, reçu: {input_dim}")
        
        # Set activation function based on argument
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()  # Default to ReLU
        
        # Build network layers
        layers = []
        prev_size = int(input_dim)
        for hidden_size in hidden_layers:
            hidden_size = int(hidden_size)
            layers.extend([
                nn.Linear(prev_size, hidden_size),    # Fully connected layer
                nn.LayerNorm(hidden_size),            # Layer normalization
                self.activation,                      # Non-linearity
                nn.Dropout(dropout_rate)              # Dropout for regularization
            ])
            prev_size = hidden_size
        
        # Output layer (regression, single output)
        layers.append(nn.Linear(prev_size, 1))
        
        # Combine layers into a sequential network
        self.network = nn.Sequential(*layers)
        
        # Weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights of all Linear layers using Xavier initialization.
        Biases are initialized to zero.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        return self.network(x)

class AdaptiveFeatureSelector:
    """
    Adaptive feature selector using Lasso (and related) strategies.

    Supports standard Lasso, ElasticNet, and two-step adaptive Lasso feature selection.
    Includes stability selection with memory and thresholding for more robust results
    over multiple selection rounds (such as in time-series or cross-validation).
    
    Parameters
    ----------
    method : str
        Feature selection method: 'lasso', 'elastic_net', or 'adaptive_lasso'
    alpha_range : tuple
        Range of regularization strengths (alphas) to search
    max_features : int or None
        Maximum number of features to select (keep)
    min_features : int
        Minimum number of features to keep
    stability_threshold : float
        Minimum stability score to retain a feature over time
    memory_factor : float
        Controls exponential smoothing of stability scores (between 0 and 1)
    """

    def __init__(
        self, 
        method='lasso', 
        alpha_range=(1e-4, 1e-1),
        max_features=None,
        min_features=5,
        stability_threshold=0.7,
        memory_factor=0.3
    ):
        """
        Initialize selector with the specified method and parameters.
        """
        self.method = method
        self.alpha_range = alpha_range
        self.max_features = max_features
        self.min_features = min_features
        self.stability_threshold = stability_threshold
        self.memory_factor = memory_factor
        
        # Feature selection history and stability scores (used for stability selection)
        self.feature_history = []
        self.stability_scores = {}

    def _get_selector_model(self, X, y):
        """
        Internal. Returns the configured feature selection model,
        fitted as necessary (for adaptive Lasso).
        """
        if self.method == 'lasso':
            # LassoCV for Lasso-based feature selection
            alphas = np.logspace(np.log10(self.alpha_range[0]), 
                                 np.log10(self.alpha_range[1]), 50)
            return LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=2000)
        
        elif self.method == 'elastic_net':
            # ElasticNetCV for ElasticNet-based feature selection
            alphas = np.logspace(np.log10(self.alpha_range[0]), 
                                 np.log10(self.alpha_range[1]), 20)
            l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
            return ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, 
                                cv=5, random_state=42, max_iter=2000)
        
        elif self.method == 'adaptive_lasso':
            """
            Adaptive Lasso: fit a Ridge to get weights, then fit Lasso with inverse weights.
            """
            from sklearn.linear_model import RidgeCV
            ridge = RidgeCV(cv=5)
            ridge.fit(X, y)
            weights = 1 / (np.abs(ridge.coef_) + 1e-8)
            X_weighted = X / weights
            alphas = np.logspace(np.log10(self.alpha_range[0]), 
                                 np.log10(self.alpha_range[1]), 50)
            lasso = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=2000)
            lasso.fit(X_weighted, y)
            # Adjust coefficients back to original scale
            lasso.coef_ = lasso.coef_ / weights
            return lasso
        else:
            raise ValueError(f"Method {self.method} is not supported.")

    def select_features(self, X, y, feature_names):
        """
        Selects features using the chosen method.
        
        Returns:
        --------
        selected_features : list of str
            Names of selected features
        alpha : float
            Regularization strength chosen by CV
        selected_coefs : array
            Nonzero coefficients of selected features
        """
        # Get and fit the selector model
        selector_model = self._get_selector_model(X, y)
        if self.method != 'adaptive_lasso':
            selector_model.fit(X, y)
        
        # Features with nonzero (abs > 1e-8) coefficients are selected
        selected_mask = np.abs(selector_model.coef_) > 1e-8
        
        # Limit number of features if max_features is set
        if self.max_features is not None and np.sum(selected_mask) > self.max_features:
            coef_abs = np.abs(selector_model.coef_)
            top_indices = np.argsort(coef_abs)[-self.max_features:]
            selected_mask = np.zeros_like(selected_mask, dtype=bool)
            selected_mask[top_indices] = True
        
        # Ensure at least min_features are selected
        if np.sum(selected_mask) < self.min_features:
            coef_abs = np.abs(selector_model.coef_)
            top_indices = np.argsort(coef_abs)[-self.min_features:]
            selected_mask = np.zeros_like(selected_mask, dtype=bool)
            selected_mask[top_indices] = True
        
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        
        # Update stability scores and apply stability-based selection if applicable
        self._update_stability_scores(selected_features, feature_names)
        if len(self.feature_history) > 0:
            selected_features = self._apply_stability_filter(selected_features, feature_names)
        
        # Record selection history
        self.feature_history.append(selected_features)
        
        return selected_features, selector_model.alpha_, selector_model.coef_[selected_mask]

    def _update_stability_scores(self, selected_features, all_features):
        """
        Update stability scores for each feature (exponentially weighted).
        """
        for feature in all_features:
            if feature not in self.stability_scores:
                self.stability_scores[feature] = 0.0
            # Decay score over time
            self.stability_scores[feature] *= (1 - self.memory_factor)
            # Boost score if feature was selected
            if feature in selected_features:
                self.stability_scores[feature] += self.memory_factor

    def _apply_stability_filter(self, current_selection, all_features):
        """
        Apply the stability threshold to combine currently selected features
        and historically stable features.
        """
        # Select features with high stability
        stable_features = [
            f for f in all_features 
            if self.stability_scores.get(f, 0) >= self.stability_threshold
        ]
        # Combine and deduplicate
        combined_features = list(set(current_selection + stable_features))
        # Limit if needed
        if self.max_features is not None and len(combined_features) > self.max_features:
            combined_features.sort(key=lambda x: self.stability_scores.get(x, 0), reverse=True)
            combined_features = combined_features[:self.max_features]
        return combined_features

    def get_feature_stability_report(self):
        """
        Return a pandas DataFrame summarizing the stability score of each feature.

        Returns:
        --------
        pd.DataFrame with columns ['feature', 'stability_score'], sorted by stability.
        """
        if not self.stability_scores:
            return pd.DataFrame()
        stability_df = pd.DataFrame([
            {'feature': feature, 'stability_score': score}
            for feature, score in self.stability_scores.items()
        ]).sort_values('stability_score', ascending=False)
        return stability_df

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience, device):
    """
    Trains a PyTorch model with early stopping and robust error handling.

    This function performs supervised training for the provided model using the specified
    data loaders, optimizer, and loss criterion. Includes:
      - Early stopping based on validation loss
      - Support for gradient clipping
      - Scheduler step on validation loss (if provided)
      - Device handling (CPU/GPU)
      - Batch-level and epoch-level error catching for robustness

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to be trained.
    train_loader : DataLoader
        PyTorch DataLoader for training data.
    val_loader : DataLoader
        PyTorch DataLoader for validation data.
    criterion : torch.nn.Module
        Loss function (e.g., nn.MSELoss()).
    optimizer : torch.optim.Optimizer
        Optimization algorithm (e.g., Adam, SGD).
    scheduler : torch.optim.lr_scheduler._LRScheduler or None
        Learning rate scheduler to call at end of each epoch (can be None).
    epochs : int
        Maximum number of training epochs.
    patience : int
        Number of epochs with no improvement to wait before stopping (early stopping).
    device : torch.device
        Device to run training on ('cpu' or 'cuda').

    Returns
    -------
    dict
        {
            'final_epoch': int,          # Number of epochs completed
            'best_val_loss': float,      # Best validation loss achieved
            'train_losses': list,        # List of train losses per epoch
            'val_losses': list           # List of val losses per epoch
        }
    """
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    # --- Defensive: Make sure there is data to train on ---
    if len(train_loader) == 0 or len(val_loader) == 0:
        return {
            'final_epoch': 0,
            'best_val_loss': float('inf'),
            'train_losses': [],
            'val_losses': []
        }

    # --- Training loop with error handling ---
    for epoch in range(epochs):
        try:
            # ----- TRAINING PHASE -----
            model.train()
            train_loss = 0.0
            train_count = 0

            for batch_X, batch_y in train_loader:
                try:
                    # Move data to device (CPU/GPU)
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    # Defensive: Skip empty batches
                    if batch_X.numel() == 0 or batch_y.numel() == 0:
                        continue

                    optimizer.zero_grad()
                    outputs = model(batch_X)

                    # Squeeze for shape compatibility, handle different shape scenarios
                    outputs = outputs.squeeze()
                    if outputs.dim() == 0 and batch_y.dim() == 0:
                        # Both scalars: unsqueeze for loss
                        loss = criterion(outputs.unsqueeze(0), batch_y.unsqueeze(0))
                    elif outputs.dim() == 0:
                        # Outputs scalar, batch_y vector
                        outputs = outputs.repeat(batch_y.size(0))
                        loss = criterion(outputs, batch_y)
                    elif batch_y.dim() == 0:
                        # batch_y scalar, outputs vector
                        batch_y = batch_y.repeat(outputs.size(0))
                        loss = criterion(outputs, batch_y)
                    else:
                        loss = criterion(outputs, batch_y)

                    # Backpropagation
                    loss.backward()
                    # Gradient clipping to avoid exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_loss += loss.item()
                    train_count += 1

                except Exception as e:
                    print(f"Erreur dans batch d'entraînement: {e}")
                    continue

            # No successful batches, break training
            if train_count == 0:
                break

            # ----- VALIDATION PHASE -----
            model.eval()
            val_loss = 0.0
            val_count = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    try:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        if batch_X.numel() == 0 or batch_y.numel() == 0:
                            continue

                        outputs = model(batch_X)
                        outputs = outputs.squeeze()

                        # Shape compatibility (same as training)
                        if outputs.dim() == 0 and batch_y.dim() == 0:
                            loss = criterion(outputs.unsqueeze(0), batch_y.unsqueeze(0))
                        elif outputs.dim() == 0:
                            outputs = outputs.repeat(batch_y.size(0))
                            loss = criterion(outputs, batch_y)
                        elif batch_y.dim() == 0:
                            batch_y = batch_y.repeat(outputs.size(0))
                            loss = criterion(outputs, batch_y)
                        else:
                            loss = criterion(outputs, batch_y)

                        val_loss += loss.item()
                        val_count += 1

                    except Exception as e:
                        print(f"Erreur dans batch de validation: {e}")
                        continue

            if val_count == 0:
                break

            # --- Logging/accumulation ---
            train_loss /= train_count
            val_loss /= val_count
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Step scheduler on validation loss, if present
            if scheduler:
                scheduler.step(val_loss)

            # --- Early Stopping Logic ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        except Exception as e:
            print(f"Erreur dans époque {epoch}: {e}")
            break

    # --- Restore best model state before returning ---
    try:
        if 'best_model_state' in locals():
            model.load_state_dict(best_model_state)
    except Exception:
        pass

    return {
        'final_epoch': epoch + 1,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

def sliding_window_dl_prediction_with_lasso(
    df: pd.DataFrame,
    date_column: str = 'date',
    target_column: str = 'target_ret',
    feature_columns: list = None,
    train_years: int = 5,
    test_years: int = 1,
    tune_frequency: int = 4,
    n_trials: int = 20,
    early_stopping_rounds: int = 10,
    # Feature selection parameters
    feature_selection_method: str = 'lasso',
    feature_selection_frequency: int = 1,
    max_features: int = None,
    min_features: int = 5,
    lasso_alpha_range: Tuple[float, float] = (1e-4, 1e1),
    stability_threshold: float = 0.7,
    memory_factor: float = 0.3,
    **base_dl_params
) -> Dict[str, Any]:
    """
    Run a sliding-window deep learning pipeline with adaptive Lasso feature selection.

    The process uses:
      - A sliding window through time to simulate a walk-forward validation (train/test split over time).
      - Feature selection in each window using Lasso/ElasticNet/adaptive Lasso.
      - Optionally, Bayesian hyperparameter optimization with Optuna for the neural network.
      - Deep learning training via PyTorch, with early stopping and robust error handling.
      - Metrics and results stored for each window and summarized at the end.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (must have time, target, and feature columns).
    date_column : str
        Name of the time/date column.
    target_column : str
        Name of the regression target column.
    feature_columns : list or None
        Which features to use (if None, use all except date/target).
    train_years : int
        Number of years for training window.
    test_years : int
        Number of years for test window.
    tune_frequency : int
        How often to re-tune neural net hyperparameters (in window steps).
    n_trials : int
        Number of Optuna trials for Bayesian hyperparameter search.
    early_stopping_rounds : int
        Number of tuning rounds without improvement before Optuna stops.
    feature_selection_method : str
        Which feature selection strategy to use.
    feature_selection_frequency : int
        How often to re-select features (in window steps).
    max_features : int or None
        Max number of features to select.
    min_features : int
        Minimum number of features to keep.
    lasso_alpha_range : tuple(float, float)
        Alpha parameter range for Lasso-based feature selection.
    stability_threshold : float
        Feature stability threshold for stability selection.
    memory_factor : float
        Exponential smoothing factor for stability calculation.
    **base_dl_params : dict
        Any additional neural net params (overrides defaults).

    Returns
    -------
    Dict
        Dictionary containing:
            - predictions
            - actual_values
            - metrics_by_window
            - feature_importance_evolution
            - feature_selection_evolution
            - best_params_evolution
            - dates
            - training_history
            - overall_metrics
            - final_feature_stability
    """

    # --- Device configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device: {device}")

    # --- Data preparation ---
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df_sorted = df.sort_values(by=date_column)

    # Determine features to use
    if feature_columns is None:
        exclude_cols = [target_column, date_column]
        feature_columns = [col for col in df_sorted.columns if col not in exclude_cols]

    # --- Feature Selector Initialization ---
    feature_selector = AdaptiveFeatureSelector(
        method=feature_selection_method,
        alpha_range=lasso_alpha_range,
        max_features=max_features,
        min_features=min_features,
        stability_threshold=stability_threshold,
        memory_factor=memory_factor
    )

    # --- Deep Learning Parameter Defaults ---
    base_params = {
        'hidden_layers': [1024, 512, 256, 128],
        'dropout_rate': 0.3,
        'learning_rate': 0.0001,
        'batch_size': 256,
        'epochs': 500,
        'activation': 'relu',
        'optimizer': 'adam',
        'patience': 15,
        'weight_decay': 1e-4
    }
    base_params.update(base_dl_params)

    # --- Results Storage ---
    results = {
        'predictions': [],
        'actual_values': [],
        'metrics_by_window': [],
        'feature_importance_evolution': [],
        'feature_selection_evolution': [],
        'best_params_evolution': [],
        'dates': [],
        'training_history': []
    }

    # --- Sliding Window Setup ---
    min_date = df_sorted[date_column].min()
    max_date = df_sorted[date_column].max()
    current_start = min_date
    window_count = 0
    best_params = base_params.copy()
    current_selected_features = feature_columns.copy()

    print(f"\n=== DEEP LEARNING (PyTorch) avec PRÉSÉLECTION LASSO ===")
    print(f"Période totale: {min_date.strftime('%Y-%m')} à {max_date.strftime('%Y-%m')}")
    print(f"Fenêtre d'entraînement: {train_years} ans")
    print(f"Fenêtre de test: {test_years} ans")
    print(f"Méthode de sélection: {feature_selection_method}")
    print(f"Sélection des features tous les {feature_selection_frequency} pas")
    print(f"Tuning bayésien tous les {tune_frequency} pas\n")

    while True:
        """
        Main sliding window loop.
        For each window:
        - Split data into train/test according to rolling dates.
        - Optionally re-select features and/or re-tune neural net hyperparameters.
        - Train final model and collect predictions/metrics.
        """
        train_end = current_start + pd.DateOffset(years=train_years)
        test_start = train_end
        test_end = test_start + pd.DateOffset(years=test_years)

        if test_end > max_date:
            break

        # Split data into train and test windows
        train_mask = (df_sorted[date_column] >= current_start) & (df_sorted[date_column] < train_end)
        test_mask = (df_sorted[date_column] >= test_start) & (df_sorted[date_column] < test_end)
        train_data = df_sorted[train_mask]
        test_data = df_sorted[test_mask]

        # Not enough data? Move forward one year.
        if len(train_data) < 100 or len(test_data) < 10:
            current_start += pd.DateOffset(years=1)
            continue

        # --- PHASE 1: FEATURE SELECTION ---
        if window_count % feature_selection_frequency == 0:
            print(f"Fenêtre {window_count + 1}: Sélection des features avec {feature_selection_method}...")

            # Prepare data for feature selection
            X_selection = train_data[feature_columns].fillna(train_data[feature_columns].median())
            y_selection = train_data[target_column]
            scaler_selection = StandardScaler()
            X_selection_scaled = scaler_selection.fit_transform(X_selection)

            # Select features
            selected_features, lasso_alpha, lasso_coefs = feature_selector.select_features(
                X_selection_scaled, y_selection.values, feature_columns
            )
            current_selected_features = selected_features

            print(f"Features sélectionnées: {len(selected_features)}/{len(feature_columns)}")
            print(f"Alpha Lasso optimal: {lasso_alpha:.6f}")
            print(f"Top 5 features: {selected_features[:5]}")

        # Prepare final train and test sets with selected features
        X_train = train_data[current_selected_features].fillna(train_data[current_selected_features].median())
        y_train = train_data[target_column]
        X_test = test_data[current_selected_features].fillna(X_train.median())
        y_test = test_data[target_column]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- PHASE 2: HYPERPARAMETER TUNING ---
        if window_count % tune_frequency == 0:
            """
            Bayesian hyperparameter tuning (with Optuna) on training set using
            rolling time series cross-validation.
            """
            print(f"Fenêtre {window_count + 1}: Hyperparameter tuning bayésien en cours...")

            def objective(trial):
                """
                Defines the Optuna optimization objective for NN hyperparams.
                """
                # Sample NN layer sizes and parameters
                n_layers = trial.suggest_categorical('n_layers', [3, 4, 5])

                # COUCHES BEAUCOUP PLUS LARGES pour vos 867k échantillons
                layer1 = trial.suggest_categorical('layer1', [512, 1024, 1536, 2048])      # 64-512 → 512-2048
                layer2 = trial.suggest_categorical('layer2', [256, 512, 768, 1024])        # 32-256 → 256-1024  
                layer3 = trial.suggest_categorical('layer3', [128, 256, 384, 512])         # 16-128 → 128-512
                layer4 = trial.suggest_categorical('layer4', [64, 128, 192, 256])          # 8-64 → 64-256
                layer5 = trial.suggest_categorical('layer5', [32, 64, 96, 128])            # 8-32 → 32-128

                if n_layers == 3:
                    layer_sizes = [layer1, layer2, layer3]
                elif n_layers == 4:
                    layer_sizes = [layer1, layer2, layer3, layer4]
                else:
                    layer_sizes = [layer1, layer2, layer3, layer4, layer5]

                params = {
                    'hidden_layers': layer_sizes,
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.05, 0.3),              
                    'learning_rate': trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True),  
                    'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024, 2048]), 
                    'activation': trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu']),
                    'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop']),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),    
                    'epochs': 50,
                    'patience': 8
                }

                # --- TimeSeriesSplit for validation ---
                tscv = TimeSeriesSplit(n_splits=3)
                val_scores = []

                for train_idx, valid_idx in tscv.split(X_train_scaled):
                    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[valid_idx]
                    y_tr, y_val = y_train.iloc[train_idx].values, y_train.iloc[valid_idx].values
                    X_tr = np.array(X_tr, dtype=np.float32)
                    X_val = np.array(X_val, dtype=np.float32)
                    y_tr = np.array(y_tr, dtype=np.float32).reshape(-1)
                    y_val = np.array(y_val, dtype=np.float32).reshape(-1)

                    # Defensive: if too little data, penalize trial
                    if len(y_tr) < 5 or len(y_val) < 2:
                        val_scores.append(1e6)
                        continue

                    try:
                        # Setup PyTorch datasets/loaders
                        train_dataset = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
                        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
                        effective_batch_size = min(params['batch_size'], len(train_dataset) // 2)
                        train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=effective_batch_size)

                        # Build NN
                        model = DeepLearningRegressor(
                            input_dim=int(X_tr.shape[1]),
                            hidden_layers=[int(x) for x in params['hidden_layers']],
                            dropout_rate=params['dropout_rate'],
                            activation=params['activation']
                        )

                        # Optimizer
                        if params['optimizer'] == 'adam':
                            optimizer = optim.Adam(model.parameters(),
                                                  lr=params['learning_rate'],
                                                  weight_decay=params['weight_decay'])
                        elif params['optimizer'] == 'adamw':
                            optimizer = optim.AdamW(model.parameters(),
                                                   lr=params['learning_rate'],
                                                   weight_decay=params['weight_decay'])
                        else:
                            optimizer = optim.RMSprop(model.parameters(),
                                                     lr=params['learning_rate'],
                                                     weight_decay=params['weight_decay'])

                        criterion = nn.MSELoss()
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

                        # Train NN
                        training_info = train_model(
                            model, train_loader, val_loader, criterion, optimizer, scheduler,
                            params['epochs'], params['patience'], device
                        )
                        val_scores.append(training_info['best_val_loss'])

                    except Exception as e:
                        print(f"Erreur dans validation croisée: {e}")
                        val_scores.append(1e6)

                return np.mean(val_scores)

            # --- Optuna study and early stopping ---
            study = optuna.create_study(direction='minimize')
            best_score = np.inf
            no_improve_rounds = 0

            def callback(study, trial):
                nonlocal best_score, no_improve_rounds
                if study.best_value + 1e-8 < best_score:
                    best_score = study.best_value
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
                if no_improve_rounds >= early_stopping_rounds:
                    study.stop()

            study.optimize(objective, n_trials=n_trials, callbacks=[callback])

            # --- Store new best NN hyperparameters ---
            best_trial_params = study.best_trial.params
            n_layers = best_trial_params['n_layers']
            if n_layers == 3:
                hidden_layers = [best_trial_params['layer1'],
                                 best_trial_params['layer2'],
                                 best_trial_params['layer3']]
            elif n_layers == 4:
                hidden_layers = [best_trial_params['layer1'],
                                 best_trial_params['layer2'],
                                 best_trial_params['layer3'],
                                 best_trial_params['layer4']]
            else:
                hidden_layers = [best_trial_params['layer1'],
                                 best_trial_params['layer2'],
                                 best_trial_params['layer3'],
                                 best_trial_params['layer4'],
                                 best_trial_params['layer5']]

            best_params = {
                'hidden_layers': hidden_layers,
                'dropout_rate': best_trial_params['dropout_rate'],
                'learning_rate': best_trial_params['learning_rate'],
                'batch_size': best_trial_params['batch_size'],
                'activation': best_trial_params['activation'],
                'optimizer': best_trial_params['optimizer'],
                'weight_decay': best_trial_params['weight_decay'],
                'epochs': 200,
                'patience': 20
            }
            print(f"New Best Parameters: {best_params}")

        # --- PHASE 3: FINAL MODEL TRAINING ---
        try:
            print(f"Dimensions: X_train={X_train_scaled.shape}, y_train={y_train.shape}")

            # Convert to numpy arrays for PyTorch
            X_train_array = np.array(X_train_scaled, dtype=np.float32)
            y_train_array = np.array(y_train.values, dtype=np.float32).reshape(-1)
            X_test_array = np.array(X_test_scaled, dtype=np.float32)
            y_test_array = np.array(y_test.values, dtype=np.float32).reshape(-1)
            if len(y_train_array) < 10:
                raise ValueError("not enough DATA")

            # Datasets & loaders
            train_dataset = TensorDataset(torch.FloatTensor(X_train_array), torch.FloatTensor(y_train_array))
            test_dataset = TensorDataset(torch.FloatTensor(X_test_array), torch.FloatTensor(y_test_array))

            train_size = max(int(0.9 * len(train_dataset)), len(train_dataset) - 10)
            val_size = len(train_dataset) - train_size
            if val_size < 5:
                train_subset = train_dataset
                val_subset = train_dataset
            else:
                train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

            effective_batch_size = min(best_params['batch_size'], len(train_subset) // 4, 4096)
            train_loader = DataLoader(train_subset, batch_size=effective_batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=effective_batch_size)
            test_loader = DataLoader(test_dataset, batch_size=effective_batch_size)

            # Build and train final model
            final_model = DeepLearningRegressor(
                input_dim=int(X_train_array.shape[1]),
                hidden_layers=[int(x) for x in best_params['hidden_layers']],
                dropout_rate=best_params['dropout_rate'],
                activation=best_params['activation']
            )

            if best_params['optimizer'] == 'adam':
                optimizer = optim.Adam(final_model.parameters(),
                                      lr=best_params['learning_rate'],
                                      weight_decay=best_params['weight_decay'])
            elif best_params['optimizer'] == 'adamw':
                optimizer = optim.AdamW(final_model.parameters(),
                                       lr=best_params['learning_rate'],
                                       weight_decay=best_params['weight_decay'])
            else:
                optimizer = optim.RMSprop(final_model.parameters(),
                                         lr=best_params['learning_rate'],
                                         weight_decay=best_params['weight_decay'])

            criterion = nn.MSELoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

            training_info = train_model(
                final_model, train_loader, val_loader, criterion, optimizer, scheduler,
                best_params['epochs'], best_params['patience'], device
            )

            # --- Predict on test set ---
            final_model.eval()
            test_predictions = []
            with torch.no_grad():
                for batch_idx, (batch_X, _) in enumerate(test_loader):
                    try:
                        batch_X = batch_X.to(device)
                        outputs = final_model(batch_X)
                        outputs_np = outputs.squeeze().cpu().numpy()
                        if outputs_np.ndim == 0:
                            test_predictions.append(outputs_np.item())
                        elif outputs_np.ndim == 1:
                            test_predictions.extend(outputs_np.tolist())
                        else:
                            test_predictions.extend(outputs_np.flatten().tolist())
                    except Exception as e:
                        print(f"Error in batch prediction {batch_idx}: {e}")
                        batch_size = len(batch_X)
                        test_predictions.extend([y_train.mean()] * batch_size)
            # Adjust prediction count if needed
            if len(test_predictions) != len(y_test):
                print(f"Ajustement: {len(test_predictions)} prédictions pour {len(y_test)} échantillons")
                if len(test_predictions) > len(y_test):
                    test_predictions = test_predictions[:len(y_test)]
                else:
                    while len(test_predictions) < len(y_test):
                        test_predictions.append(y_train.mean())
            test_pred = np.array(test_predictions, dtype=np.float32)

        except Exception as e:
            print(f"Erreur lors de l'entraînement: {e}")
            print(f"Type d'erreur: {type(e)}")
            import traceback
            traceback.print_exc()
            # Fallback: mean prediction
            test_pred = np.full(len(y_test), y_train.mean())
            training_info = {'final_epoch': 0, 'best_val_loss': float('inf')}

        # --- Store window metrics and information ---
        window_metrics = {
            'window': window_count + 1,
            'train_period': f"{current_start.strftime('%Y-%m')} à {train_end.strftime('%Y-%m')}",
            'test_period': f"{test_start.strftime('%Y-%m')} à {test_end.strftime('%Y-%m')}",
            'mse': mean_squared_error(y_test, test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'mae': mean_absolute_error(y_test, test_pred),
            'r2': r2_score(y_test, test_pred),
            'n_train': len(train_data),
            'n_test': len(test_data),
            'n_features_selected': len(current_selected_features),
            'final_epoch': training_info['final_epoch'],
            'best_val_loss': training_info['best_val_loss']
        }

        # Permutation feature importance (optional)
        try:
            feature_importance = calculate_pytorch_feature_importance(
                final_model, X_test_scaled, y_test.values, device
            )
        except:
            feature_importance = np.ones(len(current_selected_features)) / len(current_selected_features)
        feature_imp = pd.DataFrame({
            'feature': current_selected_features,
            'importance': feature_importance,
            'window': window_count + 1
        }).sort_values('importance', ascending=False)

        feature_selection_info = {
            'window': window_count + 1,
            'selected_features': current_selected_features.copy(),
            'n_selected': len(current_selected_features),
            'n_total': len(feature_columns),
            'selection_ratio': len(current_selected_features) / len(feature_columns)
        }

        if window_count > 0:
            stability_report = feature_selector.get_feature_stability_report()
            feature_selection_info['stability_report'] = stability_report

        # Store all results for the current window
        results['predictions'].extend(test_pred)
        results['actual_values'].extend(y_test.values)
        results['metrics_by_window'].append(window_metrics)
        results['feature_importance_evolution'].append(feature_imp)
        results['feature_selection_evolution'].append(feature_selection_info)
        results['best_params_evolution'].append({
            'window': window_count + 1,
            'params': best_params.copy()
        })
        results['dates'].extend(test_data[date_column].tolist())
        results['training_history'].append(training_info)

        print(f"Fenêtre {window_count + 1}: R² = {window_metrics['r2']:.4f}, "
              f"RMSE = {window_metrics['rmse']:.6f}, "
              f"Features = {len(current_selected_features)}, "
              f"Epochs = {training_info['final_epoch']}")

        current_start += pd.DateOffset(years=1)
        window_count += 1

    # --- Global Metrics and Reporting ---
    overall_metrics = {
        'overall_r2': r2_score(results['actual_values'], results['predictions']),
        'overall_rmse': np.sqrt(mean_squared_error(results['actual_values'], results['predictions'])),
        'overall_mae': mean_absolute_error(results['actual_values'], results['predictions']),
        'n_windows': window_count,
        'avg_window_r2': np.mean([m['r2'] for m in results['metrics_by_window']]),
        'avg_epochs': np.mean([m['final_epoch'] for m in results['metrics_by_window']]),
        'avg_val_loss': np.mean([m['best_val_loss'] for m in results['metrics_by_window'] if m['best_val_loss'] != float('inf')]),
        'avg_features_selected': np.mean([m['n_features_selected'] for m in results['metrics_by_window']]),
        'feature_reduction_ratio': 1 - (np.mean([m['n_features_selected'] for m in results['metrics_by_window']]) / len(feature_columns))
    }
    results['overall_metrics'] = overall_metrics
    results['model_type'] = 'PyTorch_DeepLearning_with_Lasso_Feature_Selection'
    results['feature_selector'] = feature_selector

    # Add final feature stability report
    final_stability_report = feature_selector.get_feature_stability_report()
    results['final_feature_stability'] = final_stability_report

    print(f"\n=== RÉSULTATS GLOBAUX DEEP LEARNING avec SÉLECTION LASSO ===")
    print(f"R² global: {overall_metrics['overall_r2']:.4f}")
    print(f"RMSE global: {overall_metrics['overall_rmse']:.6f}")
    print(f"R² moyen par fenêtre: {overall_metrics['avg_window_r2']:.4f}")
    print(f"Nombre de fenêtres: {overall_metrics['n_windows']}")
    print(f"Époques moyennes: {overall_metrics['avg_epochs']:.1f}")
    print(f"Features moyennes sélectionnées: {overall_metrics['avg_features_selected']:.1f}/{len(feature_columns)}")
    print(f"Réduction de features: {overall_metrics['feature_reduction_ratio']:.1%}")

    # Display the 10 most stable features
    if not final_stability_report.empty:
        print(f"\n=== TOP 10 FEATURES LES PLUS STABLES ===")
        for idx, row in final_stability_report.head(10).iterrows():
            print(f"{row['feature']:20}: {row['stability_score']:.3f}")

    return results

def calculate_pytorch_feature_importance(model, X_test, y_test, device):
    """
    Compute permutation-based feature importance for a PyTorch model.

    For each feature in the test set, this function permutes its values and measures the
    resulting increase in prediction error (MSE). Features whose permutation most
    increases error are deemed most important.

    Parameters
    ----------
    model : torch.nn.Module
        The trained PyTorch model for prediction.
    X_test : np.ndarray
        The test features, shape (n_samples, n_features).
    y_test : np.ndarray or pd.Series
        The true target values for the test set.
    device : torch.device
        Device to run model on ('cpu' or 'cuda').

    Returns
    -------
    importances : np.ndarray
        Array of normalized importances for each feature (sums to 1).
    """

    # Set model to evaluation mode (disables dropout, batchnorm update, etc.)
    model.eval()

    # --- Compute baseline prediction and error on unshuffled data ---
    X_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        baseline_pred = model(X_tensor).squeeze().cpu().numpy()
    baseline_score = mean_squared_error(y_test, baseline_pred)

    importances = []

    # --- For each feature, permute its values and measure error increase ---
    for i in range(X_test.shape[1]):
        """
        For each feature index i:
        - Shuffle the values for column i in X_test
        - Predict using the model on this permuted set
        - Compute new prediction error (MSE)
        - Importance is the increase in error from baseline
        """
        X_permuted = X_test.copy()
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
        X_permuted_tensor = torch.FloatTensor(X_permuted).to(device)

        with torch.no_grad():
            permuted_pred = model(X_permuted_tensor).squeeze().cpu().numpy()
        permuted_score = mean_squared_error(y_test, permuted_pred)

        # Feature importance is (increase in error). Negative values clipped to zero.
        importance = permuted_score - baseline_score
        importances.append(max(0, importance))

    # --- Normalize importances to sum to 1 ---
    importances = np.array(importances)
    if importances.sum() > 0:
        importances = importances / importances.sum()

    return importances

def analyze_feature_selection_results(results):
    """
    Detailed analysis of feature selection evolution across sliding windows.

    Examines:
      - The distribution (min/max/mean/median) of the number of features selected per window.
      - The most frequently selected features over all windows.
      - The correlation between the number of selected features and model R² performance.

    Parameters
    ----------
    results : dict
        Output dictionary from sliding window process. Must contain 'feature_selection_evolution'
        (list of per-window feature selection info) and 'metrics_by_window' (with R² scores).

    Returns
    -------
    dict
        {
            'n_features_stats': {
                'min': int,
                'max': int,
                'mean': float,
                'median': float
            },
            'feature_frequency': collections.Counter,
            'correlation_features_performance': float
        }
    """

    print("\n=== ANALYSE DÉTAILLÉE DE LA SÉLECTION DES FEATURES ===")
    
    # --- 1. Evolution of the number of features per window ---
    n_features_by_window = [info['n_selected'] for info in results['feature_selection_evolution']]
    
    print(f"Nombre de features par fenêtre:")
    print(f"  Minimum: {min(n_features_by_window)}")
    print(f"  Maximum: {max(n_features_by_window)}")
    print(f"  Moyenne: {np.mean(n_features_by_window):.1f}")
    print(f"  Médiane: {np.median(n_features_by_window):.1f}")
    
    # --- 2. Features most frequently selected across all windows ---
    all_selected_features = []
    for info in results['feature_selection_evolution']:
        all_selected_features.extend(info['selected_features'])
    
    from collections import Counter
    feature_counts = Counter(all_selected_features)
    n_windows = len(results['feature_selection_evolution'])
    
    print(f"\n=== TOP 15 FEATURES LES PLUS SÉLECTIONNÉES ===")
    for feature, count in feature_counts.most_common(15):
        percentage = (count / n_windows) * 100
        print(f"{feature:25}: {count:2d}/{n_windows} ({percentage:5.1f}%)")
    
    # --- 3. Correlation between feature count and R² performance ---
    r2_scores = [m['r2'] for m in results['metrics_by_window']]
    correlation = np.corrcoef(n_features_by_window, r2_scores)[0, 1]
    
    print(f"\nCorrélation nombre de features vs R²: {correlation:.3f}")
    
    return {
        'n_features_stats': {
            'min': min(n_features_by_window),
            'max': max(n_features_by_window),
            'mean': np.mean(n_features_by_window),
            'median': np.median(n_features_by_window)
        },
        'feature_frequency': feature_counts,
        'correlation_features_performance': correlation
    }
#-------------------------------------

#Random Forest and XgBoost Model
#-------------------------------------
def sliding_window_r_prediction(
    df: pd.DataFrame,
    date_column: str = 'date',
    target_column: str = 'target_ret',
    feature_columns: list = None,
    train_years: int = 5,
    test_years: int = 1,
    tune_frequency: int = 4,
    n_trials: int = 20,
    early_stopping_rounds: int = 5,
    **base_rf_params
) -> Dict[str, Any]:
    """
    Sliding window Random Forest regression with Bayesian hyperparameter tuning (Optuna) and early stopping.

    The process:
      - Uses a walk-forward windowing over the time series to train/test on moving periods.
      - Runs Bayesian hyperparameter optimization (with Optuna) at a given frequency, using early stopping to speed up.
      - Trains a Random Forest on each window and stores predictions, metrics, and feature importances.
      - Aggregates overall and per-window metrics.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe (must include date, target, and feature columns).
    date_column : str
        Name of the time/date column.
    target_column : str
        Name of the target variable.
    feature_columns : list or None
        List of feature column names to use (if None, uses all except date/target/other excludes).
    train_years : int
        Number of years in each training window.
    test_years : int
        Number of years in each test window.
    tune_frequency : int
        How often to re-run hyperparameter tuning (every N windows).
    n_trials : int
        Number of Optuna trials for hyperparameter optimization.
    early_stopping_rounds : int
        Early stopping patience for Optuna.
    **base_rf_params : dict
        Additional parameters to override RandomForest defaults.

    Returns
    -------
    dict
        {
            'predictions': list,
            'actual_values': list,
            'metrics_by_window': list,
            'feature_importance_evolution': list,
            'best_params_evolution': list,
            'dates': list,
            'overall_metrics': dict,
            'model_type': str
        }
    """

    """ --- DATA PREPARATION AND SETUP --- """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df_sorted = df.sort_values(by=date_column)

    # Determine usable features
    if feature_columns is None:
        exclude_cols = [target_column, date_column]
        feature_columns = [col for col in df_sorted.columns if col not in exclude_cols]

    # Default Random Forest parameters (can be overridden)
    base_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
    base_params.update(base_rf_params)

    # Initialize result storage
    results = {
        'predictions': [],
        'actual_values': [],
        'metrics_by_window': [],
        'feature_importance_evolution': [],
        'best_params_evolution': [],
        'dates': []
    }

    min_date = df_sorted[date_column].min()
    max_date = df_sorted[date_column].max()
    current_start = min_date
    window_count = 0
    best_params = base_params.copy()

    print(f"=== RANDOM FOREST - BAYESIAN TUNING ===")
    print(f"Période totale: {min_date.strftime('%Y-%m')} à {max_date.strftime('%Y-%m')}")
    print(f"Fenêtre d'entraînement: {train_years} ans")
    print(f"Fenêtre de test: {test_years} ans")
    print(f"Tuning bayésien tous les {tune_frequency} pas\n")

    """ --- MAIN SLIDING WINDOW LOOP --- """
    while True:
        # Define window periods
        train_end = current_start + pd.DateOffset(years=train_years)
        test_start = train_end
        test_end = test_start + pd.DateOffset(years=test_years)

        if test_end > max_date:
            break

        # Mask data for train/test splits
        train_mask = (df_sorted[date_column] >= current_start) & (df_sorted[date_column] < train_end)
        test_mask = (df_sorted[date_column] >= test_start) & (df_sorted[date_column] < test_end)
        train_data = df_sorted[train_mask]
        test_data = df_sorted[test_mask]

        # Skip windows with not enough data
        if len(train_data) < 100 or len(test_data) < 10:
            current_start += pd.DateOffset(years=1)
            continue

        X_train = train_data[feature_columns].fillna(train_data[feature_columns].median())
        y_train = train_data[target_column]
        X_test = test_data[feature_columns].fillna(X_train.median())
        y_test = test_data[target_column]

        """ --- BAYESIAN HYPERPARAMETER TUNING (Optuna) --- """
        if window_count % tune_frequency == 0:
            print(f"Fenêtre {window_count + 1}: Hyperparameter tuning bayésien en cours...")

            def objective(trial):
                """
                Optuna objective for Random Forest hyperparameters.
                """
                params = {
                    'n_estimators': trial.suggest_categorical('n_estimators', list(range(200, 250, 10))),
                    'max_depth': trial.suggest_categorical('max_depth', [8, 10, 12]),
                    'min_samples_split': trial.suggest_categorical('min_samples_split', [5, 6, 7, 8, 9, 10]),
                    'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [2, 4, 6]),
                    'max_features': trial.suggest_categorical('max_features', [0.6, 0.7, 0.8, 0.9, 1.0]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True]),
                    'random_state': 42,
                    'n_jobs': -1
                }
                model = RandomForestRegressor(**params)
                tscv = TimeSeriesSplit(n_splits=3)
                scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
                return -scores.mean()

            # Early stopping: custom callback for Optuna
            study = optuna.create_study(direction='minimize')
            best_score = np.inf
            no_improve_rounds = 0

            def callback(study, trial):
                nonlocal best_score, no_improve_rounds
                if study.best_value + 1e-8 < best_score:
                    best_score = study.best_value
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
                if no_improve_rounds >= early_stopping_rounds:
                    study.stop()

            study.optimize(objective, n_trials=n_trials, callbacks=[callback])
            best_params = study.best_trial.params
            best_params['random_state'] = 42
            best_params['n_jobs'] = -1
            print(f"Nouveaux meilleurs paramètres: {best_params}")

        """ --- TRAIN FINAL RANDOM FOREST MODEL FOR THIS WINDOW --- """
        model = RandomForestRegressor(**best_params)
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)

        """ --- STORE METRICS, FEATURE IMPORTANCE, ETC FOR THIS WINDOW --- """
        window_metrics = {
            'window': window_count + 1,
            'train_period': f"{current_start.strftime('%Y-%m')} à {train_end.strftime('%Y-%m')}",
            'test_period': f"{test_start.strftime('%Y-%m')} à {test_end.strftime('%Y-%m')}",
            'mse': mean_squared_error(y_test, test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'mae': mean_absolute_error(y_test, test_pred),
            'r2': r2_score(y_test, test_pred),
            'n_train': len(train_data),
            'n_test': len(test_data)
        }

        feature_imp = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_,
            'window': window_count + 1
        }).sort_values('importance', ascending=False)

        results['predictions'].extend(test_pred)
        results['actual_values'].extend(y_test.values)
        results['metrics_by_window'].append(window_metrics)
        results['feature_importance_evolution'].append(feature_imp)
        results['best_params_evolution'].append({
            'window': window_count + 1,
            'params': best_params.copy()
        })
        results['dates'].extend(test_data[date_column].tolist())

        print(f"Fenêtre {window_count + 1}: R² = {window_metrics['r2']:.4f}, RMSE = {window_metrics['rmse']:.6f}")

        current_start += pd.DateOffset(years=1)
        window_count += 1

    """ --- GLOBAL METRICS AND FINAL REPORTING --- """
    overall_metrics = {
        'overall_r2': r2_score(results['actual_values'], results['predictions']),
        'overall_rmse': np.sqrt(mean_squared_error(results['actual_values'], results['predictions'])),
        'overall_mae': mean_absolute_error(results['actual_values'], results['predictions']),
        'n_windows': window_count,
        'avg_window_r2': np.mean([m['r2'] for m in results['metrics_by_window']])
    }

    results['overall_metrics'] = overall_metrics
    results['model_type'] = 'RandomForest'

    print(f"\n=== RÉSULTATS GLOBAUX RANDOM FOREST ===")
    print(f"R² global: {overall_metrics['overall_r2']:.4f}")
    print(f"RMSE global: {overall_metrics['overall_rmse']:.6f}")
    print(f"R² moyen par fenêtre: {overall_metrics['avg_window_r2']:.4f}")
    print(f"Nombre de fenêtres: {overall_metrics['n_windows']}")

    return results

def sliding_window_xgb_prediction(
    df: pd.DataFrame,
    date_column: str = 'date',
    target_column: str = 'target_ret',
    feature_columns: list = None,
    train_years: int = 5,
    test_years: int = 1,
    tune_frequency: int = 4,
    n_trials: int = 20,
    early_stopping_rounds: int = 10,  # Not used in XGB fit, only for Optuna stopping
    **base_xgb_params
) -> Dict[str, Any]:
    """
    Sliding window XGBoost regression with Bayesian hyperparameter optimization (Optuna).

    For each time window:
      - Uses walk-forward validation to simulate real-time forecasting.
      - Performs Bayesian hyperparameter search (Optuna) every N windows, with early stopping.
      - Fits an XGBoost regressor and stores predictions, window metrics, and feature importances.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe including date, target, and feature columns.
    date_column : str
        Name of the date column.
    target_column : str
        Name of the regression target column.
    feature_columns : list or None
        List of feature column names to use (all except target/date/Ticker if None).
    train_years : int
        Number of years for training set in each window.
    test_years : int
        Number of years for test set in each window.
    tune_frequency : int
        How often to re-run Optuna hyperparameter search (every N windows).
    n_trials : int
        Number of Optuna trials for each hyperparameter search.
    early_stopping_rounds : int
        Patience for Optuna study early stopping (NOT used in XGB fit directly).
    **base_xgb_params : dict
        Additional or overriding keyword arguments for XGBoost.

    Returns
    -------
    dict
        Dictionary containing:
            - predictions
            - actual_values
            - metrics_by_window
            - feature_importance_evolution
            - best_params_evolution
            - dates
            - overall_metrics
            - model_type
    """

    """ --- DATA PREPARATION AND SETUP --- """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df_sorted = df.sort_values(by=date_column)
    
    if feature_columns is None:
        exclude_cols = [target_column, date_column]
        feature_columns = [col for col in df_sorted.columns if col not in exclude_cols]
    
    base_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    base_params.update(base_xgb_params)
    
    results = {
        'predictions': [],
        'actual_values': [],
        'metrics_by_window': [],
        'feature_importance_evolution': [],
        'best_params_evolution': [],
        'dates': []
    }
    
    min_date = df_sorted[date_column].min()
    max_date = df_sorted[date_column].max()
    current_start = min_date
    window_count = 0
    best_params = base_params.copy()
    
    print(f"\n=== XGBOOST - BAYESIAN TUNING ===")
    print(f"Période totale: {min_date.strftime('%Y-%m')} à {max_date.strftime('%Y-%m')}")
    print(f"Fenêtre d'entraînement: {train_years} ans")
    print(f"Fenêtre de test: {test_years} ans")
    print(f"Tuning bayésien tous les {tune_frequency} pas\n")
    
    """ --- MAIN SLIDING WINDOW LOOP --- """
    while True:
        # Window definition
        train_end = current_start + pd.DateOffset(years=train_years)
        test_start = train_end
        test_end = test_start + pd.DateOffset(years=test_years)
        if test_end > max_date:
            break
            
        train_mask = (df_sorted[date_column] >= current_start) & (df_sorted[date_column] < train_end)
        test_mask = (df_sorted[date_column] >= test_start) & (df_sorted[date_column] < test_end)
        train_data = df_sorted[train_mask]
        test_data = df_sorted[test_mask]
        if len(train_data) < 100 or len(test_data) < 10:
            current_start += pd.DateOffset(years=1)
            continue
            
        X_train = train_data[feature_columns].fillna(train_data[feature_columns].median())
        y_train = train_data[target_column]
        X_test = test_data[feature_columns].fillna(X_train.median())
        y_test = test_data[target_column]
        
        """ --- HYPERPARAMETER TUNING (Optuna, every N windows) --- """
        if window_count % tune_frequency == 0:
            print(f"Fenêtre {window_count + 1}: Hyperparameter tuning bayésien en cours...")

            def objective(trial):
                """
                Optuna objective for XGBoost hyperparameters.
                """
                params = {
                    'n_estimators': trial.suggest_categorical('n_estimators', list(range(180, 250, 10))),
                    'max_depth': trial.suggest_categorical('max_depth', [3, 4, 5, 6]),
                    'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.07, 0.1]),
                    'subsample': trial.suggest_categorical('subsample', [0.6, 0.7, 0.8, 1.0]),
                    'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.6, 0.7, 0.8, 1.0]),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1.0),
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbosity': 0
                }
                tscv = TimeSeriesSplit(n_splits=3)
                val_scores = []
                for train_idx, valid_idx in tscv.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
                    model = xgb.XGBRegressor(**params)
                    # No XGB early stopping for simplicity
                    model.fit(X_tr, y_tr, verbose=False)
                    preds = model.predict(X_val)
                    val_scores.append(mean_squared_error(y_val, preds))
                return np.mean(val_scores)
            
            # Optuna study with early stopping
            study = optuna.create_study(direction='minimize')
            best_score = np.inf
            no_improve_rounds = 0
            def callback(study, trial):
                nonlocal best_score, no_improve_rounds
                if study.best_value + 1e-8 < best_score:
                    best_score = study.best_value
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
                if no_improve_rounds >= early_stopping_rounds:
                    study.stop()
            study.optimize(objective, n_trials=n_trials, callbacks=[callback])
            best_params = study.best_trial.params
            best_params['random_state'] = 42
            best_params['n_jobs'] = -1
            best_params['verbosity'] = 0
            print(f"Nouveaux meilleurs paramètres: {best_params}")
        
        """ --- TRAIN FINAL XGBOOST MODEL AND STORE RESULTS FOR THIS WINDOW --- """
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train, verbose=False)
        test_pred = model.predict(X_test)
        
        window_metrics = {
            'window': window_count + 1,
            'train_period': f"{current_start.strftime('%Y-%m')} à {train_end.strftime('%Y-%m')}",
            'test_period': f"{test_start.strftime('%Y-%m')} à {test_end.strftime('%Y-%m')}",
            'mse': mean_squared_error(y_test, test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'mae': mean_absolute_error(y_test, test_pred),
            'r2': r2_score(y_test, test_pred),
            'n_train': len(train_data),
            'n_test': len(test_data)
        }
        
        feature_imp = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_,
            'window': window_count + 1
        }).sort_values('importance', ascending=False)
        
        results['predictions'].extend(test_pred)
        results['actual_values'].extend(y_test.values)
        results['metrics_by_window'].append(window_metrics)
        results['feature_importance_evolution'].append(feature_imp)
        results['best_params_evolution'].append({
            'window': window_count + 1,
            'params': best_params.copy()
        })
        results['dates'].extend(test_data[date_column].tolist())
        
        print(f"Fenêtre {window_count + 1}: R² = {window_metrics['r2']:.4f}, RMSE = {window_metrics['rmse']:.6f}")
        
        current_start += pd.DateOffset(years=1)
        window_count += 1
    
    """ --- GLOBAL METRICS AND FINAL REPORTING --- """
    overall_metrics = {
        'overall_r2': r2_score(results['actual_values'], results['predictions']),
        'overall_rmse': np.sqrt(mean_squared_error(results['actual_values'], results['predictions'])),
        'overall_mae': mean_absolute_error(results['actual_values'], results['predictions']),
        'n_windows': window_count,
        'avg_window_r2': np.mean([m['r2'] for m in results['metrics_by_window']])
    }
    
    results['overall_metrics'] = overall_metrics
    results['model_type'] = 'XGBoost'
    
    print(f"\n=== RÉSULTATS GLOBAUX XGBOOST ===")
    print(f"R² global: {overall_metrics['overall_r2']:.4f}")
    print(f"RMSE global: {overall_metrics['overall_rmse']:.6f}")
    print(f"R² moyen par fenêtre: {overall_metrics['avg_window_r2']:.4f}")
    print(f"Nombre de fenêtres: {overall_metrics['n_windows']}")
    
    return results

def compare_sliding_window_models(rf_results: Dict, xgb_results: Dict) -> pd.DataFrame:
    """
    Compare les performances des deux modèles avec fenêtre glissante
    
    Parameters:
    -----------
    rf_results : dict
        Résultats du Random Forest
    xgb_results : dict
        Résultats du XGBoost
    
    Returns:
    --------
    pd.DataFrame avec comparaison des métriques
    """
    
    comparison = pd.DataFrame({
        'Random_Forest': [
            rf_results['overall_metrics']['overall_r2'],
            rf_results['overall_metrics']['overall_rmse'],
            rf_results['overall_metrics']['overall_mae'],
            rf_results['overall_metrics']['avg_window_r2'],
            rf_results['overall_metrics']['n_windows']
        ],
        'XGBoost': [
            xgb_results['overall_metrics']['overall_r2'],
            xgb_results['overall_metrics']['overall_rmse'],
            xgb_results['overall_metrics']['overall_mae'],
            xgb_results['overall_metrics']['avg_window_r2'],
            xgb_results['overall_metrics']['n_windows']
        ]
    }, index=['R² Global', 'RMSE Global', 'MAE Global', 'R² Moyen par Fenêtre', 'Nombre de Fenêtres'])
    
    print("\n=== COMPARAISON DES MODÈLES - FENÊTRE GLISSANTE ===")
    print(comparison)
    
    # Comparaison fenêtre par fenêtre
    window_comparison = []
    min_windows = min(len(rf_results['metrics_by_window']), len(xgb_results['metrics_by_window']))
    
    for i in range(min_windows):
        rf_metrics = rf_results['metrics_by_window'][i]
        xgb_metrics = xgb_results['metrics_by_window'][i]
        
        window_comparison.append({
            'Window': i + 1,
            'RF_R2': rf_metrics['r2'],
            'XGB_R2': xgb_metrics['r2'],
            'RF_RMSE': rf_metrics['rmse'],
            'XGB_RMSE': xgb_metrics['rmse'],
            'Period': rf_metrics['test_period']
        })
    
    window_df = pd.DataFrame(window_comparison)
    
    print(f"\n=== COMPARAISON PAR FENÊTRE ===")
    print(window_df[['Window', 'RF_R2', 'XGB_R2', 'RF_RMSE', 'XGB_RMSE']])
    
    return comparison, window_df

def plot_sliding_window_results(rf_results: Dict, xgb_results: Dict):
    """
    Fonction pour visualiser les résultats (nécessite matplotlib)
    """
    try:
        import matplotlib.pyplot as plt
        
        # Graphique des R² par fenêtre
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: R² par fenêtre
        plt.subplot(2, 2, 1)
        rf_r2 = [m['r2'] for m in rf_results['metrics_by_window']]
        xgb_r2 = [m['r2'] for m in xgb_results['metrics_by_window']]
        windows = range(1, len(rf_r2) + 1)
        
        plt.plot(windows, rf_r2, 'o-', label='Random Forest', linewidth=2)
        plt.plot(windows, xgb_r2, 's-', label='XGBoost', linewidth=2)
        plt.xlabel('Fenêtre')
        plt.ylabel('R²')
        plt.title('R² par Fenêtre Glissante')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: RMSE par fenêtre
        plt.subplot(2, 2, 2)
        rf_rmse = [m['rmse'] for m in rf_results['metrics_by_window']]
        xgb_rmse = [m['rmse'] for m in xgb_results['metrics_by_window']]
        
        plt.plot(windows, rf_rmse, 'o-', label='Random Forest', linewidth=2)
        plt.plot(windows, xgb_rmse, 's-', label='XGBoost', linewidth=2)
        plt.xlabel('Fenêtre')
        plt.ylabel('RMSE')
        plt.title('RMSE par Fenêtre Glissante')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Prédictions vs Réalité - RF
        plt.subplot(2, 2, 3)
        plt.scatter(rf_results['actual_values'], rf_results['predictions'], alpha=0.5)
        plt.plot([min(rf_results['actual_values']), max(rf_results['actual_values'])], 
                [min(rf_results['actual_values']), max(rf_results['actual_values'])], 'r--')
        plt.xlabel('Valeurs Réelles')
        plt.ylabel('Prédictions')
        plt.title('Random Forest: Prédictions vs Réalité')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Prédictions vs Réalité - XGB
        plt.subplot(2, 2, 4)
        plt.scatter(xgb_results['actual_values'], xgb_results['predictions'], alpha=0.5)
        plt.plot([min(xgb_results['actual_values']), max(xgb_results['actual_values'])], 
                [min(xgb_results['actual_values']), max(xgb_results['actual_values'])], 'r--')
        plt.xlabel('Valeurs Réelles')
        plt.ylabel('Prédictions')
        plt.title('XGBoost: Prédictions vs Réalité')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib non disponible pour les graphiques")
#-------------------------------------

#Trading Strategy
#-------------------------------------
def implement_long_short_strategy(
    results: Dict[str, Any],
    df: pd.DataFrame,
    date_column: str = 'date',
    target_column: str = 'target_ret',
    ticker_column: str = 'Ticker',
    percentile_threshold: float = 0.1,
    rebalance_frequency: str = 'yearly'  # 'yearly', 'quarterly', 'monthly'
) -> Dict[str, Any]:
    """
    Implémente une stratégie long-short basée sur les prédictions du modèle
    
    Parameters:
    -----------
    results : dict
        Résultats du modèle (RF ou XGBoost) contenant predictions, dates, etc.
    df : pd.DataFrame
        DataFrame original avec les données
    percentile_threshold : float
        Seuil pour définir les top/bottom stocks (0.1 = 10%)
    rebalance_frequency : str
        Fréquence de rebalancement du portefeuille
        
    Returns:
    --------
    Dict contenant les résultats de la stratégie
    """
    
    # Créer un DataFrame avec prédictions, dates et valeurs réelles
    strategy_df = pd.DataFrame({
        'date': results['dates'],
        'prediction': results['predictions'],
        'actual_return': results['actual_values']
    })
    
    # Fusionner avec le DataFrame original pour récupérer les tickers
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    
    # Merger sur date et target pour récupérer les tickers
    strategy_df = pd.merge(
        strategy_df, 
        df_copy[[date_column, target_column, ticker_column]],
        left_on=['date', 'actual_return'],
        right_on=[date_column, target_column],
        how='left'
    )
    
    # Nettoyer et trier
    strategy_df = strategy_df.dropna().reset_index(drop=True)
    strategy_df = strategy_df.sort_values(['date', 'prediction']).reset_index(drop=True)
    
    # Grouper par période de rebalancement
    if rebalance_frequency == 'yearly':
        strategy_df['period'] = strategy_df['date'].dt.year
    elif rebalance_frequency == 'quarterly':
        strategy_df['period'] = strategy_df['date'].dt.year.astype(str) + '-Q' + strategy_df['date'].dt.quarter.astype(str)
    else:  # monthly
        strategy_df['period'] = strategy_df['date'].dt.to_period('M')
    
    portfolio_results = []
    
    print(f"=== STRATÉGIE LONG-SHORT ({percentile_threshold*100:.0f}% - {percentile_threshold*100:.0f}%) ===")
    print(f"Rebalancement: {rebalance_frequency}")
    print(f"Périodes analysées: {strategy_df['period'].nunique()}")
    
    for period in strategy_df['period'].unique():
        period_data = strategy_df[strategy_df['period'] == period].copy()
        
        if len(period_data) < 20:  # Minimum de stocks pour la stratégie
            continue
            
        # Calculer les seuils pour long et short
        n_stocks = len(period_data)
        n_long = max(1, int(n_stocks * percentile_threshold))
        n_short = max(1, int(n_stocks * percentile_threshold))
        
        # Trier par prédiction (du plus élevé au plus bas)
        period_data_sorted = period_data.sort_values('prediction', ascending=False)
        
        # Sélectionner les positions
        long_positions = period_data_sorted.head(n_long).copy()
        short_positions = period_data_sorted.tail(n_short).copy()
        
        # Calculer les rendements de la stratégie
        long_return = long_positions['actual_return'].mean()
        short_return = short_positions['actual_return'].mean()
        
        # Rendement de la stratégie (long - short)
        strategy_return = long_return - short_return
        
        # Rendement du marché (moyenne de tous les stocks)
        market_return = period_data['actual_return'].mean()
        
        # Alpha (rendement de la stratégie - rendement du marché)
        alpha = strategy_return - market_return
        
        period_result = {
            'period': period,
            'n_stocks_total': n_stocks,
            'n_long': n_long,
            'n_short': n_short,
            'long_return': long_return,
            'short_return': short_return,
            'strategy_return': strategy_return,
            'market_return': market_return,
            'alpha': alpha,
            'long_tickers': long_positions[ticker_column].tolist(),
            'short_tickers': short_positions[ticker_column].tolist(),
            'long_predictions': long_positions['prediction'].tolist(),
            'short_predictions': short_positions['prediction'].tolist(),
            'dates': period_data['date'].tolist()
        }
        
        portfolio_results.append(period_result)
        
        print(f"Période {period}: Strategy Return = {strategy_return:.4f}, "
              f"Long = {long_return:.4f}, Short = {short_return:.4f}, "
              f"Alpha = {alpha:.4f}")
    
    # Calculer les métriques globales
    strategy_returns = [r['strategy_return'] for r in portfolio_results]
    alphas = [r['alpha'] for r in portfolio_results]
    
    overall_metrics = {
        'avg_strategy_return': np.mean(strategy_returns),
        'total_cumulative_return': np.prod([1 + r for r in strategy_returns]) - 1,
        'volatility': np.std(strategy_returns),
        'sharpe_ratio': np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0,
        'avg_alpha': np.mean(alphas),
        'win_rate': len([r for r in strategy_returns if r > 0]) / len(strategy_returns),
        'max_return': max(strategy_returns),
        'min_return': min(strategy_returns),
        'n_periods': len(portfolio_results)
    }
    
    results_dict = {
        'portfolio_results': portfolio_results,
        'overall_metrics': overall_metrics,
        'strategy_config': {
            'percentile_threshold': percentile_threshold,
            'rebalance_frequency': rebalance_frequency,
            'model_type': results.get('model_type', 'Unknown')
        }
    }
    
    print(f"\n=== RÉSULTATS GLOBAUX DE LA STRATÉGIE ===")
    print(f"Rendement moyen par période: {overall_metrics['avg_strategy_return']:.4f}")
    print(f"Rendement cumulé total: {overall_metrics['total_cumulative_return']:.4f}")
    print(f"Volatilité: {overall_metrics['volatility']:.4f}")
    print(f"Ratio de Sharpe: {overall_metrics['sharpe_ratio']:.4f}")
    print(f"Alpha moyen: {overall_metrics['avg_alpha']:.4f}")
    print(f"Taux de réussite: {overall_metrics['win_rate']:.2%}")
    
    return results_dict

def analyze_strategy_performance(strategy_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Analyse détaillée des performances de la stratégie
    """
    portfolio_results = strategy_results['portfolio_results']
    
    # Créer un DataFrame avec tous les résultats
    analysis_df = pd.DataFrame(portfolio_results)
    
    # Statistiques détaillées
    stats = {
        'Metric': [
            'Rendement Moyen Stratégie (%)',
            'Rendement Cumulé Total (%)',
            'Volatilité (%)',
            'Ratio de Sharpe',
            'Alpha Moyen (%)',
            'Taux de Réussite (%)',
            'Meilleur Rendement (%)',
            'Pire Rendement (%)',
            'Nombre de Périodes'
        ],
        'Value': [
            f"{strategy_results['overall_metrics']['avg_strategy_return']*100:.2f}",
            f"{strategy_results['overall_metrics']['total_cumulative_return']*100:.2f}",
            f"{strategy_results['overall_metrics']['volatility']*100:.2f}",
            f"{strategy_results['overall_metrics']['sharpe_ratio']:.3f}",
            f"{strategy_results['overall_metrics']['avg_alpha']*100:.2f}",
            f"{strategy_results['overall_metrics']['win_rate']*100:.1f}",
            f"{strategy_results['overall_metrics']['max_return']*100:.2f}",
            f"{strategy_results['overall_metrics']['min_return']*100:.2f}",
            f"{strategy_results['overall_metrics']['n_periods']}"
        ]
    }
    
    stats_df = pd.DataFrame(stats)
    print("\n=== ANALYSE DÉTAILLÉE DES PERFORMANCES ===")
    print(stats_df.to_string(index=False))
    
    return analysis_df

def plot_strategy_results(strategy_results: Dict[str, Any]):
    """
    Visualise les résultats de la stratégie
    """
    try:
        portfolio_results = strategy_results['portfolio_results']
        
        # Extraire les données pour les graphiques
        periods = [str(r['period']) for r in portfolio_results]
        strategy_returns = [r['strategy_return'] for r in portfolio_results]
        long_returns = [r['long_return'] for r in portfolio_results]
        short_returns = [r['short_return'] for r in portfolio_results]
        alphas = [r['alpha'] for r in portfolio_results]
        
        # Calculer les rendements cumulés
        cumulative_strategy = np.cumprod([1 + r for r in strategy_returns])
        
        plt.figure(figsize=(16, 12))
        
        # Graphique 1: Rendements par période
        plt.subplot(2, 3, 1)
        plt.plot(periods, strategy_returns, 'o-', label='Stratégie Long-Short', linewidth=2, markersize=6)
        plt.plot(periods, long_returns, 's--', label='Long seulement', alpha=0.7)
        plt.plot(periods, short_returns, '^--', label='Short seulement', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Rendements par Période')
        plt.ylabel('Rendement')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Graphique 2: Performance cumulée
        plt.subplot(2, 3, 2)
        plt.plot(periods, cumulative_strategy, 'o-', linewidth=2, markersize=6, color='green')
        plt.title('Performance Cumulée de la Stratégie')
        plt.ylabel('Valeur du Portefeuille (base 1)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Graphique 3: Distribution des rendements
        plt.subplot(2, 3, 3)
        plt.hist(strategy_returns, bins=min(15, len(strategy_returns)//2), alpha=0.7, edgecolor='black')
        plt.axvline(x=np.mean(strategy_returns), color='red', linestyle='--', 
                   label=f'Moyenne: {np.mean(strategy_returns):.3f}')
        plt.title('Distribution des Rendements')
        plt.xlabel('Rendement de la Stratégie')
        plt.ylabel('Fréquence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graphique 4: Alpha par période
        plt.subplot(2, 3, 4)
        colors = ['green' if alpha > 0 else 'red' for alpha in alphas]
        plt.bar(periods, alphas, color=colors, alpha=0.7, edgecolor='black')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.title('Alpha par Période')
        plt.ylabel('Alpha')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Graphique 5: Corrélation Long vs Short
        plt.subplot(2, 3, 5)
        plt.scatter(long_returns, short_returns, alpha=0.7, s=60)
        plt.xlabel('Rendement Long')
        plt.ylabel('Rendement Short')
        plt.title('Corrélation Long vs Short')
        
        # Ajouter une ligne de tendance
        z = np.polyfit(long_returns, short_returns, 1)
        p = np.poly1d(z)
        plt.plot(long_returns, p(long_returns), "r--", alpha=0.8)
        correlation = np.corrcoef(long_returns, short_returns)[0,1]
        plt.text(0.05, 0.95, f'Corrélation: {correlation:.3f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.grid(True, alpha=0.3)
        
        # Graphique 6: Métriques de risque
        plt.subplot(2, 3, 6)
        metrics = ['Rendement\nMoyen', 'Volatilité', 'Ratio Sharpe', 'Win Rate']
        values = [
            np.mean(strategy_returns) * 100,
            np.std(strategy_returns) * 100,
            strategy_results['overall_metrics']['sharpe_ratio'],
            strategy_results['overall_metrics']['win_rate'] * 100
        ]
        
        bars = plt.bar(metrics, values, color=['blue', 'orange', 'green', 'purple'], alpha=0.7)
        plt.title('Métriques Clés (%)')
        plt.ylabel('Valeur')
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Erreur lors de la création des graphiques: {e}")

def compare_strategies(rf_strategy: Dict, xgb_strategy: Dict) -> pd.DataFrame:
    """
    Compare les performances des stratégies basées sur RF et XGBoost
    """
    comparison = pd.DataFrame({
        'Random_Forest_Strategy': [
            rf_strategy['overall_metrics']['avg_strategy_return'],
            rf_strategy['overall_metrics']['total_cumulative_return'],
            rf_strategy['overall_metrics']['volatility'],
            rf_strategy['overall_metrics']['sharpe_ratio'],
            rf_strategy['overall_metrics']['avg_alpha'],
            rf_strategy['overall_metrics']['win_rate']
        ],
        'XGBoost_Strategy': [
            xgb_strategy['overall_metrics']['avg_strategy_return'],
            xgb_strategy['overall_metrics']['total_cumulative_return'],
            xgb_strategy['overall_metrics']['volatility'],
            xgb_strategy['overall_metrics']['sharpe_ratio'],
            xgb_strategy['overall_metrics']['avg_alpha'],
            xgb_strategy['overall_metrics']['win_rate']
        ]
    }, index=[
        'Rendement Moyen par Période',
        'Rendement Cumulé Total',
        'Volatilité',
        'Ratio de Sharpe',
        'Alpha Moyen',
        'Taux de Réussite'
    ])
    
    print("\n=== COMPARAISON DES STRATÉGIES RF vs XGBoost ===")
    print(comparison)
    
    return comparison

#--------------------------------------