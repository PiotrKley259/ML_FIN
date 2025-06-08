from datetime import datetime, timedelta

# Typing
from typing import Any, Dict

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    
