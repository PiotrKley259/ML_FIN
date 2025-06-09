# Built-in
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
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
        Feature selection method: 'lasso', 'elastic_net', 'adaptive_lasso', or 'random_forest'
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
                                np.log10(self.alpha_range[1]), 500)
            return LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=2000)
        elif self.method == 'elastic_net':
            # ElasticNetCV for ElasticNet-based feature selection
            alphas = np.logspace(np.log10(self.alpha_range[0]),
                                np.log10(self.alpha_range[1]), 200)
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
                                np.log10(self.alpha_range[1]), 500)
            lasso = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=2000)
            lasso.fit(X_weighted, y)
            # Adjust coefficients back to original scale
            lasso.coef_ = lasso.coef_ / weights
            return lasso
        elif self.method == 'random_forest':
            # Random Forest for feature importance-based selection
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.feature_selection import SelectFromModel
            
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            # Use SelectFromModel to select features based on importance
            selector = SelectFromModel(rf, threshold='median')
            return selector
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
            Regularization strength chosen by CV (None for random_forest)
        selected_coefs : array
            Nonzero coefficients of selected features (feature importances for random_forest)
        """
        # Get and fit the selector model
        selector_model = self._get_selector_model(X, y)
        if self.method != 'adaptive_lasso':
            selector_model.fit(X, y)
        
        # Handle different types of selectors
        if self.method == 'random_forest':
            # For Random Forest, use the transform method to get selected features
            selected_mask = selector_model.get_support()
            alpha_value = None  # Random Forest doesn't have alpha
            # Get feature importances for selected features
            feature_importances = selector_model.estimator_.feature_importances_
            selected_coefs = feature_importances[selected_mask]
        else:
            # For regularization methods (Lasso, ElasticNet, Adaptive Lasso)
            # Features with nonzero (abs > 1e-8) coefficients are selected
            selected_mask = np.abs(selector_model.coef_) > 1e-8
            alpha_value = selector_model.alpha_
            selected_coefs = selector_model.coef_[selected_mask]
        
        # Limit number of features if max_features is set
        if self.max_features is not None and np.sum(selected_mask) > self.max_features:
            if self.method == 'random_forest':
                # For Random Forest, sort by feature importance
                feature_importances = selector_model.estimator_.feature_importances_
                top_indices = np.argsort(feature_importances)[-self.max_features:]
            else:
                # For regularization methods, sort by coefficient magnitude
                coef_abs = np.abs(selector_model.coef_)
                top_indices = np.argsort(coef_abs)[-self.max_features:]
            
            selected_mask = np.zeros_like(selected_mask, dtype=bool)
            selected_mask[top_indices] = True
            
            # Update selected_coefs accordingly
            if self.method == 'random_forest':
                selected_coefs = feature_importances[selected_mask]
            else:
                selected_coefs = selector_model.coef_[selected_mask]
        
        # Ensure at least min_features are selected
        if np.sum(selected_mask) < self.min_features:
            if self.method == 'random_forest':
                # For Random Forest, sort by feature importance
                feature_importances = selector_model.estimator_.feature_importances_
                top_indices = np.argsort(feature_importances)[-self.min_features:]
            else:
                # For regularization methods, sort by coefficient magnitude
                coef_abs = np.abs(selector_model.coef_)
                top_indices = np.argsort(coef_abs)[-self.min_features:]
            
            selected_mask = np.zeros_like(selected_mask, dtype=bool)
            selected_mask[top_indices] = True
            
            # Update selected_coefs accordingly
            if self.method == 'random_forest':
                selected_coefs = feature_importances[selected_mask]
            else:
                selected_coefs = selector_model.coef_[selected_mask]
        
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        
        # Update stability scores and apply stability-based selection if applicable
        self._update_stability_scores(selected_features, feature_names)
        if len(self.feature_history) > 0:
            selected_features = self._apply_stability_filter(selected_features, feature_names)
        
        # Record selection history
        self.feature_history.append(selected_features)
        
        return selected_features, alpha_value, selected_coefs

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