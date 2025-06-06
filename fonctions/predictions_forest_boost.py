from typing import Any, Dict, List, Optional, Tuple
import os
import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import (
    cross_val_score, 
    RandomizedSearchCV, 
    KFold, 
    TimeSeriesSplit, 
    train_test_split, 
    cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

import xgboost as xgb
import optuna
import wrds

from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from scipy import stats
from scipy.stats import qmc
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox


def sliding_window_r_prediction(
    df: pd.DataFrame,
    date_column: str = 'date',
    target_column: str = 'MthRet',
    feature_columns: list = None,
    train_years: int = 5,
    test_years: int = 1,
    tune_frequency: int = 4,
    n_trials: int = 20,
    early_stopping_rounds: int = 5,
    **base_rf_params
) -> Dict[str, Any]:
    """
    Random Forest avec fenêtre glissante, tuning bayésien Optuna et early stopping sur le tuning.
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df_sorted = df.sort_values(by=date_column)

    if feature_columns is None:
        exclude_cols = [target_column, 'date', 'MthRet', 'Ticker', 'sprtrn']
        feature_columns = [col for col in df_sorted.columns if col not in exclude_cols]

    base_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
    base_params.update(base_rf_params)

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

    while True:
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

        # BAYESIAN HYPERPARAMETER TUNING with EARLY STOPPING
        if window_count % tune_frequency == 0:
            print(f"Fenêtre {window_count + 1}: Hyperparameter tuning bayésien en cours...")

            # Objectif pour Optuna
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_categorical('n_estimators', list(range(200,250,10))),
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

            study = optuna.create_study(direction='minimize')
            # Early stopping custom: stop if no improvement for 'early_stopping_rounds'
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

        # Train avec les meilleurs params trouvés
        model = RandomForestRegressor(**best_params)
        model.fit(X_train, y_train)
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
    target_column: str = 'MthRet',
    feature_columns: list = None,
    train_years: int = 5,
    test_years: int = 1,
    tune_frequency: int = 4,
    n_trials: int = 20,
    early_stopping_rounds: int = 10,  # Paramètre gardé pour compatibilité mais pas utilisé
    **base_xgb_params
) -> Dict[str, Any]:
    """
    XGBoost avec fenêtre glissante et hyperparameter tuning bayésien (Optuna).
    """
    # Préparation des données
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df_sorted = df.sort_values(by=date_column)
    
    # Définir les features
    if feature_columns is None:
        exclude_cols = [target_column, 'date', 'MthRet', 'Ticker', 'sprtrn']
        feature_columns = [col for col in df_sorted.columns if col not in exclude_cols]
    
    # Paramètres de base pour XGBoost
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
    
    while True:
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
        
        # HYPERPARAMETER TUNING BAYESIEN (Optuna)
        if window_count % tune_frequency == 0:
            print(f"Fenêtre {window_count + 1}: Hyperparameter tuning bayésien en cours...")

            def objective(trial):
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
                    
                    # Fit simple sans early stopping
                    model.fit(X_tr, y_tr, verbose=False)
                    preds = model.predict(X_val)
                    val_scores.append(mean_squared_error(y_val, preds))
                return np.mean(val_scores)
            
            study = optuna.create_study(direction='minimize')
            # Early stopping custom pour Optuna
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
        
        # Entraîner le modèle final
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


#APPEL
    """
rf_results = sliding_window_r_prediction(
    df=merged,
    date_column='date',
    target_column='MthRet',
    train_years=10,
    test_years=1,
    tune_frequency=2  # Tuning tous les pas j'ai remarqué que chque fois que je tune les performances grandissent 
)

xgb_results = sliding_window_xgb_prediction(
    df=merged,
    date_column='date', 
    target_column='MthRet',
    train_years=10,
    test_years=1,
    tune_frequency=2
)
    
# Comparaison
comparison, window_comparison = compare_sliding_window_models(rf_results, xgb_results)

# Visualisation (optionnelle)
plot_sliding_window_results(rf_results, xgb_results)

    """

