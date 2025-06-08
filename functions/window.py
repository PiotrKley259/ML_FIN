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
from scipy.stats import mstats
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

from functions.class_dl import (
    DeepLearningRegressor,
    AdaptiveFeatureSelector
    
)


# Suppress warnings
warnings.filterwarnings('ignore')

#Deep Learning Model
#-------------------------------------

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
    
    enable_bayesian_tuning: bool = True,
    
    # Feature selection parameters
    feature_selection_method: str = 'lasso',
    feature_selection_frequency: int = 1,
    max_features: int = None,
    min_features: int = 5,
    lasso_alpha_range: Tuple[float, float] = (1e-4, 1e1),
    stability_threshold: float = 0.7,
    memory_factor: float = 0.3,
    include_date_as_feature : bool = True,
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
        #Convertir la date en feature numérique si demandé
    if include_date_as_feature:

        #Année décimale (ex: 2023.5 pour mi-2023)
        df['decimal_year'] = (df[date_column].dt.year + 
                             df[date_column].dt.dayofyear / 365.25)
        
        print(f"Features date créées: date_timestamp, days_since_start, decimal_year")
    
    df_sorted = df.sort_values(by=date_column)

    #Exclure seulement la colonne date originale (string/datetime)
    # mais garder les features numériques créées
    if feature_columns is None:
        if include_date_as_feature:
            # Exclure seulement la date originale, garder les features numériques
            exclude_cols = [target_column, date_column]
        else:
            # Comportement original
            exclude_cols = [target_column, date_column]
            
        feature_columns = [col for col in df_sorted.columns if col not in exclude_cols]
        
        if include_date_as_feature:
            date_features = [col for col in feature_columns if 'decimal_' in col]
            print(f"Features date incluses dans le modèle: {date_features}")

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
        'dropout_rate': 0.3, #nombre de neurones désactivés à chaque passe
        'learning_rate': 0.001,
        'batch_size': 1024,
        'epochs': 300,
        'activation': 'relu',
        'optimizer': 'adam',
        'patience': 20, #nombre d'epochs à attendre sans amélioration
        'weight_decay': 1e-4 #taux de régularization appliqué au poids
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
            print(f"Fenêtre {window_count + 1}: Préparation des données pour sélection des features...")
    
            # Prepare data for feature selection
            X_selection = train_data[feature_columns].fillna(train_data[feature_columns].median())
            y_selection = train_data[target_column]
    
            # === ÉTAPE 1 : WINSORISATION SÉLECTIVE (UNE SEULE FOIS) ===
            print(f"Winsorisation sélective des données (1%-99% percentiles)...")
            
            # Identifier les colonnes à exclure de la winsorisation ET de la normalisation
            protected_features = [col for col in X_selection.columns if 
                                'decimal_year' in col or 'timestamp' in col or 'days_since' in col or 
                                '_mask' in col or '_flag' in col or 'stock_idx' in col]
            
            # Features régulières (winsorisées ET normalisées)
            regular_features = [col for col in X_selection.columns if col not in protected_features]
            
            print(f"Features protégées (non winsorisées): {protected_features}")
            print(f"Features régulières (winsorisées + normalisées): {len(regular_features)}")
            
            # === WINSORISATION UNIQUE + SAUVEGARDE DES LIMITES ===
            X_selection_processed = X_selection.copy()
            
            # Calculer et sauvegarder les limites AVANT winsorisation pour réutilisation
            global_winsorization_limits = {}
            if regular_features:
                # Sauvegarder les percentiles originaux pour réutilisation
                global_winsorization_limits['train_percentiles_1'] = X_selection[regular_features].quantile(0.01)
                global_winsorization_limits['train_percentiles_99'] = X_selection[regular_features].quantile(0.99)
                global_winsorization_limits['regular_features'] = regular_features
                
                # Appliquer la winsorisation UNE SEULE FOIS
                X_selection_processed[regular_features] = pd.DataFrame(
                    mstats.winsorize(X_selection[regular_features].values, limits=[0.01, 0.01], axis=0),
                    columns=regular_features,
                    index=X_selection.index
                )
                
                print(f"Limites de winsorisation sauvegardées pour {len(regular_features)} features")
            else:
                global_winsorization_limits['regular_features'] = []
            
            # Winsoriser la target UNE SEULE FOIS
            y_selection_processed = pd.Series(
                mstats.winsorize(y_selection.values, limits=[0.01, 0.01]),
                index=y_selection.index
            )
            
            # Sauvegarder les limites de la target
            global_winsorization_limits['y_train_p1'] = np.percentile(y_selection.values, 1)
            global_winsorization_limits['y_train_p99'] = np.percentile(y_selection.values, 99)
    
            # === ÉTAPE 2 : SÉLECTION DE FEATURES AVEC LASSO ===
            print(f"Sélection des features avec {feature_selection_method}...")
            
            # Normalisation SEULEMENT pour les features régulières (pour LASSO)
            if regular_features:
                scaler_selection = StandardScaler()
                X_regular_scaled = scaler_selection.fit_transform(X_selection_processed[regular_features])
                
                # Créer une liste de features pour LASSO (exclure features protégées)
                lasso_feature_columns = regular_features
                
                # Appliquer LASSO seulement sur les features régulières normalisées
                selected_features_lasso, lasso_alpha, lasso_coefs = feature_selector.select_features(
                    X_regular_scaled, y_selection_processed.values, lasso_feature_columns
                )
            else:
                selected_features_lasso = []
                lasso_alpha = 0.0
            
            # === EXCLUSION DE STOCK_IDX DES FEATURES FINALES ===
            # Ajouter automatiquement les features protégées SAUF stock_idx
            protected_features_for_model = [col for col in protected_features if 'stock_idx' not in col]
            current_selected_features = selected_features_lasso + protected_features_for_model
            
            print(f"Features sélectionnées par LASSO: {len(selected_features_lasso)}")
            print(f"Features protégées ajoutées (SANS stock_idx): {len(protected_features_for_model)}")
            print(f"Total features finales: {len(current_selected_features)}/{len(feature_columns)}")
            print(f"stock_idx EXCLU du modèle (gardé seulement pour identification)")
            if lasso_alpha > 0:
                print(f"Alpha optimal: {lasso_alpha:.6f}")
            if current_selected_features:
                print(f"Top 5 features: {current_selected_features[:5]}")

            # === ÉTAPE 3 : DONNÉES TRAIN FINALES (déjà winsorisées) ===
            # Utiliser les données déjà winsorisées et sélectionnées
            X_train_final = X_selection_processed[current_selected_features]
            y_train_final = y_selection_processed
            
            # Mettre à jour les limites pour les features sélectionnées seulement
            winsorization_limits = {}
            regular_features_selected = [col for col in current_selected_features if col not in protected_features_for_model]
            
            if regular_features_selected:
                # Utiliser les limites des features sélectionnées seulement
                winsorization_limits['train_percentiles_1'] = global_winsorization_limits['train_percentiles_1'][regular_features_selected]
                winsorization_limits['train_percentiles_99'] = global_winsorization_limits['train_percentiles_99'][regular_features_selected]
                winsorization_limits['regular_features'] = regular_features_selected
            else:
                winsorization_limits['regular_features'] = []
            
            winsorization_limits['y_train_p1'] = global_winsorization_limits['y_train_p1']
            winsorization_limits['y_train_p99'] = global_winsorization_limits['y_train_p99']
            
        else:
            # === CAS SANS NOUVELLE SÉLECTION : APPLIQUER LES MÊMES LIMITES ===
            print(f"Fenêtre {window_count + 1}: Réutilisation des features ET limites de winsorisation...")
            
            # Préparer les données avec les mêmes features sélectionnées
            X_train_raw = train_data[current_selected_features].fillna(train_data[current_selected_features].median())
            y_train_raw = train_data[target_column]
            
            # Identifier les features protégées dans la sélection actuelle (SANS stock_idx)
            protected_features_current = [col for col in current_selected_features if 
                                        'decimal_year' in col or 'timestamp' in col or 'days_since' in col or
                                        '_mask' in col or '_flag' in col]
            regular_features_current = [col for col in current_selected_features if col not in protected_features_current]
            
            # === APPLIQUER LES LIMITES SAUVEGARDÉES (PAS DE NOUVELLE WINSORISATION) ===
            X_train_final = X_train_raw.copy()
            
            if regular_features_current and winsorization_limits['regular_features']:
                print(f"Application des limites de winsorisation sauvegardées sur {len(regular_features_current)} features")
                
                # Appliquer les MÊMES limites que lors de la sélection (pas de nouvelle winsorisation)
                for col in regular_features_current:
                    if col in winsorization_limits['regular_features']:
                        X_train_final[col] = np.clip(
                            X_train_raw[col].values,
                            winsorization_limits['train_percentiles_1'][col],
                            winsorization_limits['train_percentiles_99'][col]
                        )
            
            # Appliquer les mêmes limites à la target
            y_train_final = pd.Series(
                np.clip(y_train_raw.values, 
                       winsorization_limits['y_train_p1'], 
                       winsorization_limits['y_train_p99']),
                index=y_train_raw.index
            )
            
            print(f"Limites appliquées sans nouvelle winsorisation")

        # === DONNÉES TEST FINALES (application des limites du train) ===
        # Préparer les données de test SANS re-winsoriser, juste appliquer les limites
        X_test_raw = test_data[current_selected_features].fillna(X_train_final.median())
        X_test_final = X_test_raw.copy()
        
        # Appliquer SEULEMENT les limites des données train aux features régulières
        if winsorization_limits['regular_features']:
            for col in winsorization_limits['regular_features']:
                X_test_final[col] = np.clip(
                    X_test_final[col].values,
                    winsorization_limits['train_percentiles_1'][col],
                    winsorization_limits['train_percentiles_99'][col]
                )
        
        # Pour y_test, appliquer les limites du train
        y_test_final = np.clip(
            test_data[target_column].values, 
            winsorization_limits['y_train_p1'], 
            winsorization_limits['y_train_p99']
        )

        # === NORMALISATION FINALE SÉLECTIVE (SANS stock_idx) ===
        # Identifier les features à normaliser et celles à garder intactes (SANS stock_idx)
        features_to_normalize = [col for col in current_selected_features if 
                               '_mask' not in col and '_flag' not in col]
        features_to_keep_raw = [col for col in current_selected_features if col not in features_to_normalize]
        
        print(f"Features à normaliser: {len(features_to_normalize)}")
        print(f"Features gardées brutes: {features_to_keep_raw}")
        
        # Normaliser seulement les features appropriées
        if features_to_normalize:
            scaler = StandardScaler()
            X_train_normalized = scaler.fit_transform(X_train_final[features_to_normalize])
            X_test_normalized = scaler.transform(X_test_final[features_to_normalize])
            
            if features_to_keep_raw:
                # Reconstruire les DataFrames complets (sans stock_idx)
                X_train_scaled = np.column_stack([
                    X_train_normalized,  # Features normalisées
                    X_train_final[features_to_keep_raw].values  # Features brutes (masks, decimal_year)
                ])
                
                X_test_scaled = np.column_stack([
                    X_test_normalized,   # Features normalisées  
                    X_test_final[features_to_keep_raw].values   # Features brutes
                ])
                
                # Créer la liste des noms de colonnes dans le bon ordre (sans stock_idx)
                final_feature_names = features_to_normalize + features_to_keep_raw
            else:
                # Seulement features normalisées
                X_train_scaled = X_train_normalized
                X_test_scaled = X_test_normalized
                final_feature_names = features_to_normalize
                
        else:
            # Si aucune feature à normaliser, garder seulement les features brutes (sans stock_idx)
            if features_to_keep_raw:
                X_train_scaled = X_train_final[features_to_keep_raw].values
                X_test_scaled = X_test_final[features_to_keep_raw].values
                final_feature_names = features_to_keep_raw
            else:
                # Cas extrême: aucune feature utilisable
                raise ValueError("Aucune feature utilisable après exclusion de stock_idx!")
        
        # Variables finales pour la suite du code
        y_train = pd.Series(y_train_final, index=X_train_final.index)
        y_test = pd.Series(y_test_final, index=X_test_final.index)

        print(f"Données préparées. Features: {X_train_scaled.shape[1]}, "
              f"Échantillons train: {len(X_train_scaled)}, test: {len(X_test_scaled)}")
        print(f"Features finales: {final_feature_names}")
        
        # Vérification finale que stock_idx est bien absent
        if any('stock_idx' in col for col in final_feature_names):
            print("ERREUR: stock_idx encore présent dans les features finales!")
            raise ValueError("stock_idx ne devrait pas être dans les features d'entraînement!")
        else:
            print("stock_idx correctement exclu des features d'entraînement")
            
        # === RÉSUMÉ DE LA WINSORISATION ===
        if window_count % feature_selection_frequency == 0:
            print(f"WINSORISATION: Nouvelle winsorisation appliquée")
        else:
            print(f"WINSORISATION: Limites précédentes réutilisées (pas de nouvelle winsorisation)")

        
        # --- PHASE 3: HYPERPARAMETER TUNING ---
        if enable_bayesian_tuning and (window_count % tune_frequency == 0):
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
        elif not enable_bayesian_tuning and window_count == 0:
            # Message informatif seulement à la première fenêtre
            print("Tuning bayésien désactivé - utilisation des paramètres par défaut Deep Learning")
            # best_params garde ses valeurs par défaut

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
            train_predictions = []
            with torch.no_grad():
                # Créer un loader pour les données d'entraînement (sans shuffle)
                
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
                
                train_eval_dataset = TensorDataset(torch.FloatTensor(X_train_array), torch.FloatTensor(y_train_array))
                train_eval_loader = DataLoader(train_eval_dataset, batch_size=effective_batch_size, shuffle=False)

                for batch_idx, (batch_X, _) in enumerate(train_eval_loader):
                    try:
                        batch_X = batch_X.to(device)
                        outputs = final_model(batch_X)
                        outputs_np = outputs.squeeze().cpu().numpy()
                        if outputs_np.ndim == 0:
                            train_predictions.append(outputs_np.item())
                        elif outputs_np.ndim == 1:
                            train_predictions.extend(outputs_np.tolist())
                        else:
                            train_predictions.extend(outputs_np.flatten().tolist())
                    except Exception as e:
                        print(f"Error in train prediction batch {batch_idx}: {e}")
                        batch_size = len(batch_X)
                        train_predictions.extend([y_train.mean()] * batch_size)
                        
          
            if len(train_predictions) != len(y_train):
                print(f"Ajustement train: {len(train_predictions)} prédictions pour {len(y_train)} échantillons")
                if len(train_predictions) > len(y_train):
                    train_predictions = train_predictions[:len(y_train)]
                else:
                    while len(train_predictions) < len(y_train):
                        train_predictions.append(y_train.mean())

                        
                        
            # Adjust prediction count if needed
            if len(test_predictions) != len(y_test):
                print(f"Ajustement: {len(test_predictions)} prédictions pour {len(y_test)} échantillons")
                if len(test_predictions) > len(y_test):
                    test_predictions = test_predictions[:len(y_test)]
                else:
                    while len(test_predictions) < len(y_test):
                        test_predictions.append(y_train.mean())
                        
            train_pred = np.array(train_predictions, dtype=np.float32)
            test_pred = np.array(test_predictions, dtype=np.float32)
            
            train_r2 = r2_score(y_train, train_pred)
            train_mse = mean_squared_error(y_train, train_pred)
            train_rmse = np.sqrt(train_mse)

        except Exception as e:
            print(f"Erreur lors de l'entraînement: {e}")
            print(f"Type d'erreur: {type(e)}")
            import traceback
            traceback.print_exc()
            # Fallback: mean prediction
            test_pred = np.full(len(y_test), y_train.mean())
            train_pred = np.full(len(y_train), y_train.mean())  
            train_r2 = float('nan')                             
            train_mse = float('nan')                            
            train_rmse = float('nan')                           
            training_info = {'final_epoch': 0, 'best_val_loss': float('inf')}

        # --- Store window metrics and information ---
        window_metrics = {
            'window': window_count + 1,
            'train_period': f"{current_start.strftime('%Y-%m')} à {train_end.strftime('%Y-%m')}",
            'test_period': f"{test_start.strftime('%Y-%m')} à {test_end.strftime('%Y-%m')}",
            
            # Test metrics
            'test_mse': mean_squared_error(y_test, test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'test_r2': r2_score(y_test, test_pred),
    
            # Train metrics (NOUVEAU)
            'train_mse': train_mse,
            'train_rmse': train_rmse,
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_r2': train_r2,
    
            # Overfitting indicator (NOUVEAU)
            'overfitting_ratio': train_r2 - r2_score(y_test, test_pred),
            
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

        print(f"Fenêtre {window_count + 1}: "
                f"R² train = {train_r2:.4f}, "
                f"R² test = {window_metrics['test_r2']:.4f}, "
                f"Overfitting = {window_metrics['overfitting_ratio']:.4f}, "
                f"RMSE = {window_metrics['test_rmse']:.6f}, "
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
    print(f"RMSE global: {overall_metrics
          
          
          
          
          ['overall_rmse']:.6f}")
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
    include_date_as_feature: bool = True,
    enable_bayesian_tuning: bool = False,
    **base_rf_params
) -> Dict[str, Any]:
    """
    Sliding window Random Forest regression with optional Bayesian hyperparameter tuning (Optuna) and early stopping.

    The process:
      - Uses a walk-forward windowing over the time series to train/test on moving periods.
      - Optionally runs Bayesian hyperparameter optimization (with Optuna) at a given frequency, using early stopping to speed up.
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
    include_date_as_feature : bool
        Whether to include date-based features.
    enable_bayesian_tuning : bool
        Whether to enable Bayesian hyperparameter tuning with Optuna.
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
        # Convertir la date en feature numérique si demandé
    if include_date_as_feature:
        
        #  Année décimale (ex: 2023.5 pour mi-2023)
        df['decimal_year'] = (df[date_column].dt.year + 
                             df[date_column].dt.dayofyear / 365.25)
        
        print(f"Features date créées: date_timestamp, days_since_start, decimal_year")
    
    df_sorted = df.sort_values(by=date_column)

    #Exclure seulement la colonne date originale (string/datetime)
    # mais garder les features numériques créées
    if feature_columns is None:
        if include_date_as_feature:
            # Exclure seulement la date originale, garder les features numériques
            exclude_cols = [target_column, date_column]
        else:
            # Comportement original
            exclude_cols = [target_column, date_column]
            
        feature_columns = [col for col in df_sorted.columns if col not in exclude_cols]
        
        if include_date_as_feature:
            date_features = [col for col in feature_columns if 'decimal_' in col]
            print(f"Features date incluses dans le modèle: {date_features}")

    # Default Random Forest parameters (can be overridden)
    base_params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'max_features' : 0.3,
        'bootstrap': True,
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

    print(f"=== RANDOM FOREST - {'BAYESIAN TUNING' if enable_bayesian_tuning else 'FIXED PARAMS'} ===")
    print(f"Période totale: {min_date.strftime('%Y-%m')} à {max_date.strftime('%Y-%m')}")
    print(f"Fenêtre d'entraînement: {train_years} ans")
    print(f"Fenêtre de test: {test_years} ans")
    if enable_bayesian_tuning:
        print(f"Tuning bayésien tous les {tune_frequency} pas")
    else:
        print(f"Utilisation des paramètres fixes: {best_params}")
    print()

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
        if enable_bayesian_tuning and window_count % tune_frequency == 0:
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
    early_stopping_rounds: int = 10,# Not used in XGB fit, only for Optuna stopping
    include_date_as_feature: bool = True,
    enable_bayesian_tuning: bool = False,
    **base_xgb_params
) -> Dict[str, Any]:
    """
    Sliding window XGBoost regression with optional Bayesian hyperparameter optimization (Optuna).

    For each time window:
      - Uses walk-forward validation to simulate real-time forecasting.
      - Optionally performs Bayesian hyperparameter search (Optuna) every N windows, with early stopping.
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
    include_date_as_feature : bool
        Whether to include date-based features.
    enable_bayesian_tuning : bool
        Whether to enable Bayesian hyperparameter tuning with Optuna.
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
        #Convertir la date en feature numérique si demandé
    if include_date_as_feature:
        # Année décimale (ex: 2023.5 pour mi-2023)
        df['decimal_year'] = (df[date_column].dt.year + 
                             df[date_column].dt.dayofyear / 365.25)
        
        print(f"Features date créées: date_timestamp, days_since_start, decimal_year")
    
    df_sorted = df.sort_values(by=date_column)

    #Exclure seulement la colonne date originale (string/datetime)
    # mais garder les features numériques créées
    if feature_columns is None:
        if include_date_as_feature:
            # Exclure seulement la date originale, garder les features numériques
            exclude_cols = [target_column, date_column]
        else:
            # Comportement original
            exclude_cols = [target_column, date_column]
            
        feature_columns = [col for col in df_sorted.columns if col not in exclude_cols]
        
        if include_date_as_feature:
            date_features = [col for col in feature_columns if 'decimal_' in col]
            print(f"Features date incluses dans le modèle: {date_features}")

    
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
    
    print(f"\n=== XGBOOST - {'BAYESIAN TUNING' if enable_bayesian_tuning else 'FIXED PARAMS'} ===")
    print(f"Période totale: {min_date.strftime('%Y-%m')} à {max_date.strftime('%Y-%m')}")
    print(f"Fenêtre d'entraînement: {train_years} ans")
    print(f"Fenêtre de test: {test_years} ans")
    if enable_bayesian_tuning:
        print(f"Tuning bayésien tous les {tune_frequency} pas")
    else:
        print(f"Utilisation des paramètres fixes: {best_params}")
    print()
    
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
        if enable_bayesian_tuning and window_count % tune_frequency == 0:
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

#-------------------