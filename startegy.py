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


from functions import (sliding_window_dl_prediction_with_lasso, 
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



# Configuration
warnings.filterwarnings('ignore')

# Définit le device pour tout le notebook
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device utilisé :", device)

# Configuration des dossiers
folder = os.path.expanduser('~/ML/')

# Chargement des données
main_csv = os.path.join(folder, 'formatted_full_data_test_1.csv')


merged = pd.read_csv(main_csv)


rf_results = sliding_window_r_prediction(
    df=merged,
    date_column='date',
    target_column='target_ret',
    train_years=20,
    test_years=5,
    tune_frequency=5  # Tuning tous les pas j'ai remarqué que chque fois que je tune les performances grandissent 
)


xgb_results = sliding_window_xgb_prediction(
    df=merged,
    date_column='date', 
    target_column='target_ret',
    train_years=20,
    test_years=5,
    tune_frequency=5
)

# Comparaison
comparison, window_comparison = compare_sliding_window_models(rf_results, xgb_results)

# Visualisation (optionnelle)
plot_sliding_window_results(rf_results, xgb_results)

rf_strategy = implement_long_short_strategy(
    results=rf_results,
    df=merged,  # votre DataFrame original
    percentile_threshold=0.1,  # 10% top et 10% bottom
    rebalance_frequency='yearly'
)

xgb_strategy = implement_long_short_strategy(
    results=xgb_results,
    df=merged,
    percentile_threshold=0.1,
    rebalance_frequency='yearly'
)

# Analyser les performances
rf_analysis = analyze_strategy_performance(rf_strategy)
xgb_analysis = analyze_strategy_performance(xgb_strategy)

# Visualiser les résultats
plot_strategy_results(rf_strategy)
plot_strategy_results(xgb_strategy)

# Comparer les deux stratégies
strategy_comparison = compare_strategies(rf_strategy, xgb_strategy)

dl_lasso_results = sliding_window_dl_prediction_with_lasso(
    df=merged,
    date_column='date',
    target_column='target_ret',
    feature_columns=None,  # Sera automatiquement défini
    train_years=20, #on peut essayer sur 30 40 10 donne des résultats mauvais
    test_years=1,
    tune_frequency=5,
    n_trials=3,
    early_stopping_rounds=8,
    # Paramètres spécifiques à la sélection Lasso
    feature_selection_method='adaptive_lasso',  # 'lasso', 'elastic_net', 'adaptive_lasso'
    feature_selection_frequency=1,     # Re-sélection à chaque fenêtre
    max_features= None,                   # Maximum 50 features
    min_features=10,                   # Minimum 10 features
    lasso_alpha_range=(1e-6, 1e-1),   # Plage alpha pour Lasso
    stability_threshold=0.4,           # Seuil de stabilité
    memory_factor=0.3                  # Facteur de mémoire
)

print(stability_report = dl_lasso_results['final_feature_stability'])

dl_strategy = implement_long_short_strategy(
    results= dl_lasso_results,
    df=merged,
    percentile_threshold=0.1,
    rebalance_frequency='yearly'
)

# Analyser les performances
dl_analysis = analyze_strategy_performance(dl_strategy)

# Visualiser les résultats
plot_strategy_results(dl_strategy)


