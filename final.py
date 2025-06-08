# =============================================================================
# TRADING PARAMETERS - Configuration centralisée
# =============================================================================

TRADING_PARAM = {
    # Paramètres de données
    'date_column': 'date',
    'target_column': 'target_ret',
    
    # Paramètres de fenêtre glissante
    'train_years': 20,
    'test_years_rf_xgb': 1,
    'test_years_dl': 1,
    'tune_frequency': 8, #on rebalance les paramétres 5-6 fois sur TOUT le data 
    
    # Paramètres de stratégie
    'percentile_threshold': 0.1,  # 10% top et bottom
    'rebalance_frequency': 'yearly',
    
    # Paramètres Deep Learning + Lasso
    'dl_n_trials': 5, #on s'arrete après 5 tentatives d'optimization car trop demandant niveau calcul
    'dl_early_stopping_rounds': 6,
    'feature_selection_method': 'lasso',  # 'lasso', 'elastic_net', 'adaptive_lasso'
    'feature_selection_frequency': 1,
    'max_features': None,
    'min_features': 50,
    'lasso_alpha_range': (1e-6, 1e1),
    'stability_threshold': 0.4,
    'memory_factor': 0.3,
    
    # Fichiers et dossiers
    'data_filename': 'formatted_full_data_test_1.csv',
    'output_folder': 'results'
}

# Configuration d'exécution - Choisir quels modèles exécuter
MODELS_TO_RUN = {
    'random_forest': False,      # 1 - Random Forest
    'xgboost': False,           # 2 - XGBoost  
    'deep_learning': True      # 3 - Deep Learning + Lasso
}

# =============================================================================
# IMPORTS
# =============================================================================

# Typage
from typing import Dict, Any, List, Optional, Tuple

# Data manipulation
import pandas as pd
import numpy as np

# Machine Learning & Prétraitement
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import (
    cross_val_score, cross_validate, RandomizedSearchCV,
    KFold, TimeSeriesSplit, train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Affichage
import matplotlib.pyplot as plt
from tqdm import tqdm

# Statistiques & tests
from scipy import stats
from scipy.stats import qmc
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

# Divers
import xgboost as xgb
import wrds
import optuna
import os
import logging
from datetime import datetime, timedelta
import warnings

from functions import (
    sliding_window_dl_prediction_with_lasso, 
    analyze_feature_selection_results, 
    sliding_window_r_prediction,
    sliding_window_xgb_prediction,
    compare_sliding_window_models,
    plot_sliding_window_results,
    implement_long_short_strategy,
    analyze_strategy_performance,
    plot_strategy_results,
    compare_strategies
)

# =============================================================================
# CONFIGURATION
# =============================================================================

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device utilisé :", device)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    # Configuration des chemins
    folder = os.path.dirname(os.path.abspath(__file__))
    main_csv = os.path.join(folder, TRADING_PARAM['data_filename'])
    
    # Chargement des données
    print("Chargement des données...")
    merged = pd.read_csv(main_csv)
    
    # Dictionnaires pour stocker les résultats
    results = {}
    strategies = {}
    analyses = {}
    
    # =============================================================================
    # EXÉCUTION DES MODÈLES SÉLECTIONNÉS
    # =============================================================================
    
    # 1. Random Forest
    if MODELS_TO_RUN.get('random_forest', False):
        print("\n=== EXÉCUTION RANDOM FOREST ===")
        results['rf'] = sliding_window_r_prediction(
            df=merged,
            date_column=TRADING_PARAM['date_column'],
            target_column=TRADING_PARAM['target_column'],
            train_years=TRADING_PARAM['train_years'],
            test_years=TRADING_PARAM['test_years_rf_xgb'],
            tune_frequency=TRADING_PARAM['tune_frequency']
        )
    
    # 2. XGBoost
    if MODELS_TO_RUN.get('xgboost', False):
        print("\n=== EXÉCUTION XGBOOST ===")
        results['xgb'] = sliding_window_xgb_prediction(
            df=merged,
            date_column=TRADING_PARAM['date_column'], 
            target_column=TRADING_PARAM['target_column'],
            train_years=TRADING_PARAM['train_years'],
            test_years=TRADING_PARAM['test_years_rf_xgb'],
            tune_frequency=TRADING_PARAM['tune_frequency']
        )
    
    # 3. Deep Learning + Lasso
    if MODELS_TO_RUN.get('deep_learning', False):
        print("\n=== EXÉCUTION DEEP LEARNING + LASSO ===")
        results['dl_lasso'] = sliding_window_dl_prediction_with_lasso(
            df=merged,
            date_column=TRADING_PARAM['date_column'],
            target_column=TRADING_PARAM['target_column'],
            feature_columns=None,
            train_years=TRADING_PARAM['train_years'],
            test_years=TRADING_PARAM['test_years_dl'],
            tune_frequency=TRADING_PARAM['tune_frequency'],
            n_trials=TRADING_PARAM['dl_n_trials'],
            early_stopping_rounds=TRADING_PARAM['dl_early_stopping_rounds'],
            feature_selection_method=TRADING_PARAM['feature_selection_method'],
            feature_selection_frequency=TRADING_PARAM['feature_selection_frequency'],
            max_features=TRADING_PARAM['max_features'],
            min_features=TRADING_PARAM['min_features'],
            lasso_alpha_range=TRADING_PARAM['lasso_alpha_range'],
            stability_threshold=TRADING_PARAM['stability_threshold'],
            memory_factor=TRADING_PARAM['memory_factor']
        )
        
        print("Rapport de stabilité des features:")
        stability_report = results['dl_lasso']['final_feature_stability']
        print(stability_report)
    
    # =============================================================================
    # COMPARAISON DES MODÈLES (si RF et XGB sont exécutés)
    # =============================================================================
    
    if 'rf' in results and 'xgb' in results:
        print("\n=== COMPARAISON RF vs XGB ===")
        comparison, window_comparison = compare_sliding_window_models(results['rf'], results['xgb'])
        plot_sliding_window_results(results['rf'], results['xgb'])
    
    # =============================================================================
    # CRÉATION ET ANALYSE DES STRATÉGIES
    # =============================================================================
    
    print("\n=== CRÉATION DES STRATÉGIES ===")
    
    for model_name, model_results in results.items():
        print(f"Création stratégie {model_name}...")
        strategies[model_name] = implement_long_short_strategy(
            results=model_results,
            df=merged,
            percentile_threshold=TRADING_PARAM['percentile_threshold'],
            rebalance_frequency=TRADING_PARAM['rebalance_frequency']
        )
        
        print(f"Analyse performance {model_name}...")
        analyses[model_name] = analyze_strategy_performance(strategies[model_name])
        
        print(f"Génération graphique {model_name}...")
        plot_strategy_results(strategies[model_name])
    
    # =============================================================================
    # COMPARAISON DES STRATÉGIES
    # =============================================================================
    
    strategy_comparison = None
    if len(strategies) >= 2:
        print("\n=== COMPARAISON DES STRATÉGIES ===")
        strategy_keys = list(strategies.keys())
        strategy_comparison = compare_strategies(
            strategies[strategy_keys[0]], 
            strategies[strategy_keys[1]]
        )
    
    # =============================================================================
    # SAUVEGARDE DES RÉSULTATS
    # =============================================================================
    
    print("\n=== SAUVEGARDE DES RÉSULTATS ===")
    
    output_folder = os.path.join(folder, TRADING_PARAM['output_folder'])
    os.makedirs(output_folder, exist_ok=True)
    
    # Sauvegarde des performances
    with open(os.path.join(output_folder, 'strategy_performance.txt'), 'w') as f:
        for model_name, analysis in analyses.items():
            f.write(f"{model_name.upper()}:\n")
            f.write(str(analysis) + '\n\n')
        
        if strategy_comparison:
            f.write("Strategy Comparison:\n")
            f.write(str(strategy_comparison) + '\n')
    
    # Sauvegarde de la stabilité des features (si DL+Lasso disponible)
    if 'dl_lasso' in results:
        results['dl_lasso']['final_feature_stability'].to_csv(
            os.path.join(output_folder, 'feature_stability.csv'), index=False
        )
    
    # Sauvegarde des graphiques
    plots_folder = os.path.join(output_folder, 'plots')
    os.makedirs(plots_folder, exist_ok=True)
    
    if 'rf' in results and 'xgb' in results:
        plot_sliding_window_results(results['rf'], results['xgb'])
        plt.savefig(os.path.join(plots_folder, 'sliding_window_results.png'), bbox_inches='tight')
        plt.close()
    
    for model_name in strategies.keys():
        plot_strategy_results(strategies[model_name])
        plt.savefig(os.path.join(plots_folder, f'{model_name}_strategy.png'), bbox_inches='tight')
        plt.close()
    
    print("\n=== ANALYSE TERMINÉE ===")
    print(f"Modèles exécutés: {list(results.keys())}")
    print(f"Résultats sauvegardés dans: {output_folder}")

if __name__ == "__main__":
    main()