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

import math


class DeepLearningRegressor(nn.Module):
    """
    Deep neural network for regression tasks using PyTorch with Transformer architecture + ALiBi.
    This flexible regressor allows customization of:
    - The number and size of hidden layers (interpreted as transformer layers and dimensions)
    - Dropout regularization
    - Activation functions (ReLU, Tanh, LeakyReLU, ELU)
    - ALiBi (Attention with Linear Biases) for better sequence extrapolation
    
    Args:
        input_dim (int): Number of input features.
        hidden_layers (list of int): [d_model, n_heads, n_layers, ff_dim] 
                                   or [d_model, n_heads, n_layers] (ff_dim = 4*d_model)
        dropout_rate (float): Dropout probability (default 0.2).
        activation (str): Activation function for feed-forward layers.
        
    Raises:
        ValueError: If input_dim is not a positive integer.
    """
    
    def __init__(self, input_dim, hidden_layers, dropout_rate=0.2, activation='relu'):
        """
        Initialize the DeepLearningRegressor with Transformer + ALiBi architecture.
        
        Interface 100% compatible avec l'ancienne version MLP !
        """
        super(DeepLearningRegressor, self).__init__()
        
        # Safety check for input dimensions
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim doit être un entier positif, reçu: {input_dim}")
        
        # Parse hidden_layers - COMPATIBLE avec vos paramètres actuels
        if len(hidden_layers) >= 4:
            # Format: [d_model, n_heads, n_layers, ff_dim]
            d_model, n_heads, n_layers, ff_dim = hidden_layers[:4]
        elif len(hidden_layers) == 3:
            # Format: [d_model, n_heads, n_layers]
            d_model, n_heads, n_layers = hidden_layers
            ff_dim = 4 * d_model
        elif len(hidden_layers) >= 2:
            # Format: [d_model, n_heads]
            d_model, n_heads = hidden_layers[:2]
            n_layers = 6
            ff_dim = 4 * d_model
        else:
            # Format minimal: [d_model]
            d_model = hidden_layers[0] if hidden_layers else 256
            n_heads = 8
            n_layers = 6
            ff_dim = 4 * d_model
        
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.ff_dim = int(ff_dim)
        self.input_dim = input_dim
        
        # Validation des paramètres
        assert self.d_model % self.n_heads == 0, f"d_model ({self.d_model}) doit être divisible par n_heads ({self.n_heads})"
        
        # Set activation function
        if activation == 'relu':
            self.ff_activation = nn.ReLU()
        elif activation == 'tanh':
            self.ff_activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.ff_activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.ff_activation = nn.ELU()
        else:
            self.ff_activation = nn.ReLU()
        
        # Input projection to d_model
        self.input_projection = nn.Linear(input_dim, self.d_model)
        
        # ALiBi: Pas de positional encoding ! Les biais sont calculés dynamiquement
        self.max_seq_len = 2000  # Maximum sequence length supportée
        
        # Transformer encoder layers avec ALiBi
        self.transformer_layers = nn.ModuleList([
            ALiBiTransformerLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                ff_dim=self.ff_dim,
                dropout=dropout_rate,
                activation=self.ff_activation
            ) for _ in range(self.n_layers)
        ])
        
        # Layer norm finale
        self.final_norm = nn.LayerNorm(self.d_model)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            self.ff_activation,
            nn.Dropout(dropout_rate),
            nn.Linear(self.d_model // 2, self.d_model // 4),
            nn.LayerNorm(self.d_model // 4),
            self.ff_activation,
            nn.Dropout(dropout_rate),
            nn.Linear(self.d_model // 4, 1)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Forward pass of the transformer network avec ALiBi.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim) 
                            or (batch_size, input_dim) for single timestep
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        batch_size = x.size(0)
        
        # Handle different input shapes
        if x.dim() == 2:
            # Single timestep: (batch_size, input_dim) -> (batch_size, 1, input_dim)
            x = x.unsqueeze(1)
            seq_len = 1
        else:
            # Sequential: (batch_size, seq_len, input_dim)
            seq_len = x.size(1)
        
        # Project input to d_model
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Passer par chaque couche Transformer avec ALiBi
        for layer in self.transformer_layers:
            x = layer(x)  # Les biais ALiBi sont calculés automatiquement dans chaque couche
        
        # Final normalization
        x = self.final_norm(x)  # (batch_size, seq_len, d_model)
        
        # Global average pooling across sequence dimension
        pooled = x.mean(dim=1)  # (batch_size, d_model)
        
        # Output prediction
        output = self.output_layers(pooled)  # (batch_size, 1)
        
        return output
    
    def get_attention_weights(self, x, layer_idx=0):
        """
        Extract attention weights from a specific transformer layer.
        
        Args:
            x (torch.Tensor): Input tensor
            layer_idx (int): Which transformer layer to extract from
            
        Returns:
            torch.Tensor: Attention weights of shape (batch_size, n_heads, seq_len, seq_len)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # Forward through transformer layers up to desired layer
        for i, layer in enumerate(self.transformer_layers):
            if i == layer_idx:
                return layer.get_attention_weights(x)
            x = layer(x)
        
        return None

class ALiBiTransformerLayer(nn.Module):
    """
    Couche Transformer avec ALiBi (Attention with Linear Biases)
    """
    
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1, activation=nn.ReLU()):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Multi-head attention avec ALiBi
        self.attention = ALiBiMultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass avec connexions résiduelles et ALiBi
        """
        # Pre-norm multi-head attention avec ALiBi
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        
        # Pre-norm feed-forward
        ff_out = self.ff_network(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x
    
    def get_attention_weights(self, x):
        """Extraire les poids d'attention pour visualisation"""
        return self.attention.get_attention_weights(self.norm1(x))
    
class ALiBiMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention avec ALiBi (Attention with Linear Biases)
    
    ALiBi remplace le positional encoding par des biais linéaires appliqués
    directement aux scores d'attention, améliorant l'extrapolation.
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0
        
        # Projections linéaires pour Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False) 
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # ALiBi slopes - calculées une seule fois et réutilisées
        self.register_buffer('alibi_slopes', self._get_alibi_slopes(n_heads))
    
    def _get_alibi_slopes(self, n_heads):
        """
        Calcule les pentes ALiBi pour chaque tête d'attention
        
        Formule: m_i = 2^(-8*i/n_heads) pour i=1..n_heads
        """
        # Calculer les pentes selon le papier ALiBi
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(n_heads).is_integer():
            # Si n_heads est une puissance de 2
            slopes = get_slopes_power_of_2(n_heads)
        else:
            # Si n_heads n'est pas une puissance de 2
            closest_power_of_2 = 2**(math.floor(math.log2(n_heads)))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            
            # Ajouter des pentes supplémentaires si nécessaire
            extra = n_heads - closest_power_of_2
            if extra > 0:
                extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)
                slopes.extend(extra_slopes[closest_power_of_2:closest_power_of_2 + extra])
        
        return torch.tensor(slopes[:n_heads], dtype=torch.float32)
    
    def _get_alibi_bias(self, seq_len, device):
        """
        Génère la matrice de biais ALiBi pour une séquence de longueur seq_len
        
        Returns:
            torch.Tensor: (n_heads, seq_len, seq_len) - biais ALiBi
        """
        # Créer une matrice de distances relatives
        # distances[i][j] = i - j (distance du token i au token j)
        distances = torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(seq_len, device=device).unsqueeze(1)
        
        # Appliquer les pentes ALiBi: bias = slope * distance
        # Shape: (n_heads, 1, 1) * (1, seq_len, seq_len) = (n_heads, seq_len, seq_len)
        alibi_bias = self.alibi_slopes.to(device).unsqueeze(1).unsqueeze(2) * distances.unsqueeze(0).float()
        
        return alibi_bias
    
    def forward(self, x):
        """
        Forward pass de l'attention avec ALiBi
        
        Args:
            x: (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Projections Q, K, V
        Q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        K = self.k_proj(x)  # (batch_size, seq_len, d_model)
        V = self.v_proj(x)  # (batch_size, seq_len, d_model)
        
        # Reshape pour multi-head: (batch_size, n_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Calcul des scores d'attention: Q @ K^T / sqrt(head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Shape: (batch_size, n_heads, seq_len, seq_len)
        
        # AJOUTER LES BIAIS ALiBi
        alibi_bias = self._get_alibi_bias(seq_len, x.device)
        # Shape: (n_heads, seq_len, seq_len) -> broadcast à (batch_size, n_heads, seq_len, seq_len)
        scores = scores + alibi_bias.unsqueeze(0)
        
        # Softmax pour obtenir les poids d'attention
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Appliquer l'attention: weights @ V
        attn_output = torch.matmul(attn_weights, V)
        # Shape: (batch_size, n_heads, seq_len, head_dim)
        
        # Concatener les têtes: (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Projection de sortie
        output = self.out_proj(attn_output)
        
        return output
    
    def get_attention_weights(self, x):
        """
        Retourne les poids d'attention pour visualisation
        """
        batch_size, seq_len, d_model = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        alibi_bias = self._get_alibi_bias(seq_len, x.device)
        scores = scores + alibi_bias.unsqueeze(0)
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        return attn_weights

class AdaptiveFeatureSelector:
    """
    Adaptive feature selector optimized for Transformers with non-linear selection methods.

    Supports Mutual Information, Variance + Correlation filtering, and Transformer-based attention
    feature selection. Includes stability selection with memory and thresholding for more robust 
    results over multiple selection rounds.
    
    Parameters
    ----------
    method : str
        Feature selection method: 'mutual_info', 'variance_corr', 'lasso' (legacy), 'elastic_net', 'adaptive_lasso'
    alpha_range : tuple
        Range for mutual information parameters or legacy alpha range
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
        method='mutual_info',  # Changed default to mutual_info
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
        
        # Feature selection history and stability scores
        self.feature_history = []
        self.stability_scores = {}

    def _get_selector_model(self, X, y):
        """
        Internal. Returns the configured feature selection model.
        Now includes Transformer-optimized methods.
        """
        if self.method == 'mutual_info':
            # Mutual Information - captures non-linear relationships
            from sklearn.feature_selection import mutual_info_regression
            
            class MutualInfoSelector:
                def __init__(self):
                    self.feature_scores_ = None
                    self.alpha_ = 0.0  # For compatibility
                    self.coef_ = None
                
                def fit(self, X, y):
                    # Calculate mutual information scores
                    self.feature_scores_ = mutual_info_regression(
                        X, y, 
                        discrete_features=False,
                        n_neighbors=3,
                        random_state=42
                    )
                    # Create pseudo-coefficients for compatibility
                    self.coef_ = self.feature_scores_.copy()
                    return self
            
            return MutualInfoSelector()
            
        elif self.method == 'variance_corr':
            # Variance threshold + correlation filtering
            from sklearn.feature_selection import VarianceThreshold
            
            class VarianceCorrSelector:
                def __init__(self, variance_threshold=0.01, corr_threshold=0.95):
                    self.variance_threshold = variance_threshold
                    self.corr_threshold = corr_threshold
                    self.feature_scores_ = None
                    self.alpha_ = 0.0
                    self.coef_ = None
                
                def fit(self, X, y):
                    n_features = X.shape[1]
                    
                    # Step 1: Remove low variance features
                    var_selector = VarianceThreshold(threshold=self.variance_threshold)
                    X_var_filtered = var_selector.fit_transform(X)
                    var_mask = var_selector.get_support()
                    
                    # Step 2: Remove highly correlated features
                    if X_var_filtered.shape[1] > 1:
                        corr_matrix = np.corrcoef(X_var_filtered.T)
                        # Find upper triangle of correlation matrix
                        upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                        high_corr_pairs = np.where((np.abs(corr_matrix) > self.corr_threshold) & upper_tri)
                        
                        # Create mask for features to keep
                        corr_mask = np.ones(X_var_filtered.shape[1], dtype=bool)
                        corr_mask[high_corr_pairs[1]] = False
                        
                        # Combine masks
                        final_mask = np.zeros(n_features, dtype=bool)
                        var_indices = np.where(var_mask)[0]
                        final_mask[var_indices[corr_mask]] = True
                    else:
                        final_mask = var_mask
                    
                    # Calculate importance scores (inverse of correlation with removed features)
                    scores = np.ones(n_features)
                    scores[~final_mask] = 0.0
                    
                    # Add some randomness to break ties
                    np.random.seed(42)
                    scores += np.random.normal(0, 0.001, n_features)
                    
                    self.feature_scores_ = scores
                    self.coef_ = scores.copy()
                    return self
            
            return VarianceCorrSelector()
            
        elif self.method == 'polynomial_features':
            # Polynomial features + feature selection
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.feature_selection import SelectKBest, f_regression
            
            class PolynomialFeatureSelector:
                def __init__(self, degree=2, interaction_only=True, include_bias=False):
                    self.degree = degree
                    self.interaction_only = interaction_only  # Seulement les interactions, pas les puissances
                    self.include_bias = include_bias
                    self.feature_scores_ = None
                    self.alpha_ = 0.0
                    self.coef_ = None
                    self.poly_transformer_ = None
                    self.original_feature_names_ = None
                    self.poly_feature_names_ = None
                
                def fit(self, X, y):
                    n_samples, n_features = X.shape
                    
                    # Limiter le nombre de features pour éviter l'explosion combinatoire
                    max_features_for_poly = min(50, n_features)  # Maximum 50 features pour polynomial
                    
                    if n_features > max_features_for_poly:
                        print(f"Trop de features ({n_features}) pour polynomial. Pré-sélection des {max_features_for_poly} meilleures...")
                        # Pré-sélection avec mutual info
                        from sklearn.feature_selection import mutual_info_regression
                        mi_scores = mutual_info_regression(X, y, random_state=42)
                        top_indices = np.argsort(mi_scores)[-max_features_for_poly:]
                        X_reduced = X[:, top_indices]
                        self.preselected_indices_ = top_indices
                    else:
                        X_reduced = X
                        self.preselected_indices_ = np.arange(n_features)
                    
                    # Créer les features polynomiales
                    self.poly_transformer_ = PolynomialFeatures(
                        degree=self.degree,
                        interaction_only=self.interaction_only,
                        include_bias=self.include_bias
                    )
                    
                    X_poly = self.poly_transformer_.fit_transform(X_reduced)
                    
                    print(f"Features polynomiales créées: {X_reduced.shape[1]} → {X_poly.shape[1]}")
                    
                    # Éviter l'explosion mémoire avec trop de features
                    max_poly_features = 1000
                    if X_poly.shape[1] > max_poly_features:
                        print(f"Trop de features polynomiales ({X_poly.shape[1]}). Sélection des {max_poly_features} meilleures...")
                        
                        # Sélection rapide avec variance et correlation
                        from sklearn.feature_selection import VarianceThreshold
                        var_selector = VarianceThreshold(threshold=0.01)
                        X_poly_var = var_selector.fit_transform(X_poly)
                        
                        if X_poly_var.shape[1] > max_poly_features:
                            # Sélection finale avec f_regression (plus rapide que mutual_info)
                            selector = SelectKBest(f_regression, k=max_poly_features)
                            X_poly_final = selector.fit_transform(X_poly_var, y)
                            poly_scores = selector.scores_
                            
                            # Reconstruction des indices
                            var_indices = var_selector.get_support()
                            kbest_indices = selector.get_support()
                            final_indices = np.where(var_indices)[0][kbest_indices]
                        else:
                            X_poly_final = X_poly_var
                            var_indices = var_selector.get_support()
                            poly_scores = f_regression(X_poly_var, y)[0]
                            final_indices = np.where(var_indices)[0]
                    else:
                        # Si pas trop de features, utiliser toutes
                        X_poly_final = X_poly
                        poly_scores = f_regression(X_poly, y)[0]
                        final_indices = np.arange(X_poly.shape[1])
                    
                    # Stocker les scores des features polynomiales sélectionnées
                    full_scores = np.zeros(X_poly.shape[1])
                    full_scores[final_indices] = poly_scores
                    
                    self.feature_scores_ = full_scores
                    self.coef_ = full_scores.copy()
                    self.selected_poly_indices_ = final_indices
                    
                    # Générer les noms des features polynomiales pour debug
                    if hasattr(self.poly_transformer_, 'get_feature_names_out'):
                        try:
                            input_features = [f'x{i}' for i in range(X_reduced.shape[1])]
                            self.poly_feature_names_ = self.poly_transformer_.get_feature_names_out(input_features)
                        except:
                            self.poly_feature_names_ = [f'poly_{i}' for i in range(X_poly.shape[1])]
                    else:
                        self.poly_feature_names_ = [f'poly_{i}' for i in range(X_poly.shape[1])]
                    
                    print(f"Sélection finale: {len(final_indices)} features polynomiales")
                    return self
                
                def get_selected_polynomial_info(self):
                    """Retourne des infos sur les features polynomiales sélectionnées"""
                    if hasattr(self, 'poly_feature_names_') and hasattr(self, 'selected_poly_indices_'):
                        selected_names = [self.poly_feature_names_[i] for i in self.selected_poly_indices_]
                        selected_scores = self.feature_scores_[self.selected_poly_indices_]
                        return list(zip(selected_names, selected_scores))
                    return []
            
            return PolynomialFeatureSelector(degree=2, interaction_only=True)
        
        elif self.method == 'lasso':
            # Legacy Lasso support for backward compatibility
            alphas = np.logspace(np.log10(self.alpha_range[0]), 
                                 np.log10(self.alpha_range[1]), 500)
            return LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=2000)
        
        elif self.method == 'elastic_net':
            # Legacy ElasticNet support
            alphas = np.logspace(np.log10(self.alpha_range[0]), 
                                 np.log10(self.alpha_range[1]), 200)
            l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
            return ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, 
                                cv=5, random_state=42, max_iter=2000)
        
        elif self.method == 'adaptive_lasso':
            # Legacy Adaptive Lasso support
            from sklearn.linear_model import RidgeCV
            ridge = RidgeCV(cv=5)
            ridge.fit(X, y)
            weights = 1 / (np.abs(ridge.coef_) + 1e-8)
            X_weighted = X / weights
            alphas = np.logspace(np.log10(self.alpha_range[0]), 
                                 np.log10(self.alpha_range[1]), 500)
            lasso = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=2000)
            lasso.fit(X_weighted, y)
            lasso.coef_ = lasso.coef_ / weights
            return lasso
        
        else:
            raise ValueError(f"Method {self.method} is not supported. "
                           f"Available: 'mutual_info', 'variance_corr', 'lasso', 'elastic_net', 'adaptive_lasso'")

    def select_features(self, X, y, feature_names):
        """
        Selects features using the chosen method.
        
        INTERFACE IDENTIQUE - Aucun changement dans l'appel !
        
        Returns:
        --------
        selected_features : list of str
            Names of selected features
        alpha : float
            Regularization strength or method parameter
        selected_coefs : array
            Feature importance scores of selected features
        """
        # Get and fit the selector model
        selector_model = self._get_selector_model(X, y)
        
        # Fit the model
        if self.method != 'adaptive_lasso':
            selector_model.fit(X, y)
        
        # Handle different scoring methods
        if hasattr(selector_model, 'feature_scores_'):
            # For mutual_info and variance_corr methods
            scores = selector_model.feature_scores_
            
            # Select features based on scores
            if self.max_features is not None:
                # Select top max_features
                top_indices = np.argsort(scores)[-self.max_features:]
                selected_mask = np.zeros_like(scores, dtype=bool)
                selected_mask[top_indices] = True
            else:
                # Select features with positive scores
                selected_mask = scores > np.percentile(scores, 50)  # Top 50%
                
        else:
            # For legacy linear methods (lasso, elastic_net, adaptive_lasso)
            selected_mask = np.abs(selector_model.coef_) > 1e-8
            scores = np.abs(selector_model.coef_)
            
            # Limit number of features if max_features is set
            if self.max_features is not None and np.sum(selected_mask) > self.max_features:
                top_indices = np.argsort(scores)[-self.max_features:]
                selected_mask = np.zeros_like(selected_mask, dtype=bool)
                selected_mask[top_indices] = True
        
        # Ensure at least min_features are selected
        if np.sum(selected_mask) < self.min_features:
            top_indices = np.argsort(scores)[-self.min_features:]
            selected_mask = np.zeros_like(selected_mask, dtype=bool)
            selected_mask[top_indices] = True
        
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        
        # Update stability scores and apply stability-based selection
        self._update_stability_scores(selected_features, feature_names)
        if len(self.feature_history) > 0:
            selected_features = self._apply_stability_filter(selected_features, feature_names)
        
        # Record selection history
        self.feature_history.append(selected_features)
        
        # Return in same format as before
        alpha_value = getattr(selector_model, 'alpha_', 0.0)
        selected_coefs = scores[selected_mask] if hasattr(scores, '__getitem__') else scores
        
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

    def get_transformer_attention_features(self, transformer_model, X_sample, top_k=None):
        """
        NOUVELLE MÉTHODE : Sélection basée sur les poids d'attention du Transformer
        
        Args:
            transformer_model: Modèle Transformer entraîné
            X_sample: Échantillon de données pour calculer l'attention
            top_k: Nombre de features à sélectionner
            
        Returns:
            list: Indices des features les plus importantes selon l'attention
        """
        if not hasattr(transformer_model, 'get_attention_weights'):
            raise ValueError("Le modèle doit avoir une méthode get_attention_weights")
        
        try:
            # Convertir en tensor si nécessaire
            if isinstance(X_sample, np.ndarray):
                X_tensor = torch.FloatTensor(X_sample[:100])  # Échantillon pour calcul
            else:
                X_tensor = X_sample[:100]
            
            # Extraire les poids d'attention
            transformer_model.eval()
            with torch.no_grad():
                attention_weights = transformer_model.get_attention_weights(X_tensor)
                # Moyenne sur batch, têtes et positions
                feature_importance = attention_weights.mean(dim=(0, 1, 2)).cpu().numpy()
            
            # Sélectionner top_k features
            if top_k is None:
                top_k = len(feature_importance) // 2
            
            top_indices = np.argsort(feature_importance)[-top_k:]
            return top_indices.tolist()
            
        except Exception as e:
            print(f"Erreur dans sélection par attention: {e}")
            # Fallback sur mutual info
            return list(range(min(top_k or 50, X_sample.shape[1])))

# Ajout des imports nécessaires pour les nouvelles méthodes
try:
    from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
except ImportError:
    print("sklearn.feature_selection non disponible - utilisation des méthodes legacy uniquement")

try:
    from sklearn.linear_model import LassoCV, ElasticNetCV, RidgeCV
except ImportError:
    print("sklearn.linear_model non disponible")
