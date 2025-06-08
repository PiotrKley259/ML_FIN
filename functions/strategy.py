import warnings
from datetime import datetime, timedelta

# Typing
from typing import Any, Dict

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Progress bar
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

def implement_long_short_strategy(
    results: Dict[str, Any],
    df: pd.DataFrame,
    date_column: str = 'date',
    target_column: str = 'target_ret',
    ticker_column: str = 'stock_idx',
    percentile_threshold: float = 0.1,
    rebalance_frequency: str = 'yearly',  # 'yearly', 'quarterly', 'monthly'
    # TRADING COSTS ===
    include_trading_costs: bool = True,
    transaction_cost_bps: float = 10.0,
    bid_ask_spread_bps: float = 5.0,
    market_impact_bps: float = 2.0,
    borrowing_cost_annual: float = 0.02,
    management_fee_annual: float = 0.015
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
    include_trading_costs : bool
        Si True, inclut les coûts de trading dans les calculs
    transaction_cost_bps : float
        Coût de transaction en basis points
    bid_ask_spread_bps : float
        Spread bid-ask en basis points
    market_impact_bps : float
        Impact de marché en basis points
    borrowing_cost_annual : float
        Coût annuel d'emprunt pour les positions short
    management_fee_annual : float
        Frais de gestion annuels
        
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
        period_duration_years = 1.0
    elif rebalance_frequency == 'quarterly':
        strategy_df['period'] = strategy_df['date'].dt.year.astype(str) + '-Q' + strategy_df['date'].dt.quarter.astype(str)
        period_duration_years = 0.25
    else:  # monthly
        strategy_df['period'] = strategy_df['date'].dt.to_period('M')
        period_duration_years = 1/12
    
    portfolio_results = []
    previous_long_tickers = set()
    previous_short_tickers = set()
    
    # === CALCUL DES COÛTS (si activés) ===
    if include_trading_costs:
        total_cost_per_transaction_bps = transaction_cost_bps + bid_ask_spread_bps + market_impact_bps
        total_cost_per_transaction = total_cost_per_transaction_bps / 10000
        print(f"=== STRATÉGIE LONG-SHORT AVEC COÛTS ({percentile_threshold*100:.0f}% - {percentile_threshold*100:.0f}%) ===")
        print(f"Coût total par transaction: {total_cost_per_transaction_bps:.1f} bps")
    else:
        print(f"=== STRATÉGIE LONG-SHORT ({percentile_threshold*100:.0f}% - {percentile_threshold*100:.0f}%) ===")
    
    print(f"Rebalancement: {rebalance_frequency}")
    print(f"Périodes analysées: {strategy_df['period'].nunique()}")
    
    for i, period in enumerate(strategy_df['period'].unique()):
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
        
        # Calculer les rendements bruts
        long_return_gross = long_positions['actual_return'].mean()
        short_return_gross = short_positions['actual_return'].mean()
        strategy_return_gross = long_return_gross - short_return_gross
        
        # Calculer les coûts si activés
        if include_trading_costs:
            current_long_tickers = set(long_positions[ticker_column])
            current_short_tickers = set(short_positions[ticker_column])
            
            # Calcul des transactions
            if i == 0:
                n_new_long = len(current_long_tickers)
                n_new_short = len(current_short_tickers)
                n_closed_long = 0
                n_closed_short = 0
            else:
                n_new_long = len(current_long_tickers - previous_long_tickers)
                n_new_short = len(current_short_tickers - previous_short_tickers)
                n_closed_long = len(previous_long_tickers - current_long_tickers)
                n_closed_short = len(previous_short_tickers - current_short_tickers)
            
            total_transactions = n_new_long + n_new_short + n_closed_long + n_closed_short
            rebalancing_cost = total_transactions * total_cost_per_transaction
            
            # Coûts périodiques
            borrowing_cost_period = borrowing_cost_annual * period_duration_years * 0.5
            management_fee_period = management_fee_annual * period_duration_years
            total_period_costs = rebalancing_cost + borrowing_cost_period + management_fee_period
            
            # Rendements nets
            strategy_return = strategy_return_gross - total_period_costs
            long_return = long_return_gross - (rebalancing_cost * 0.5 + management_fee_period * 0.5)
            short_return = short_return_gross - (rebalancing_cost * 0.5 + borrowing_cost_period + management_fee_period * 0.5)
            
            previous_long_tickers = current_long_tickers.copy()
            previous_short_tickers = current_short_tickers.copy()
        else:
            # Sans coûts - rendements bruts
            strategy_return = strategy_return_gross
            long_return = long_return_gross
            short_return = short_return_gross
            total_period_costs = 0.0
            total_transactions = 0
        
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
        
        # Ajouter les métriques de coûts si activés
        if include_trading_costs:
            period_result.update({
                'long_return_gross': long_return_gross,
                'short_return_gross': short_return_gross,
                'strategy_return_gross': strategy_return_gross,
                'total_costs': total_period_costs,
                'total_transactions': total_transactions,
                'cost_impact': strategy_return_gross - strategy_return
            })
        
        portfolio_results.append(period_result)
        
        if include_trading_costs:
            print(f"Période {period}: Net = {strategy_return:.4f}, "
                  f"Brut = {strategy_return_gross:.4f}, "
                  f"Coûts = {total_period_costs:.4f}")
        else:
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
    
    # Ajouter les métriques de coûts si activés
    if include_trading_costs:
        strategy_returns_gross = [r['strategy_return_gross'] for r in portfolio_results]
        total_costs_list = [r['total_costs'] for r in portfolio_results]
        
        overall_metrics.update({
            'avg_strategy_return_gross': np.mean(strategy_returns_gross),
            'total_cumulative_return_gross': np.prod([1 + r for r in strategy_returns_gross]) - 1,
            'avg_total_costs': np.mean(total_costs_list),
            'total_costs_cumulative': sum(total_costs_list),
            'avg_cost_impact': np.mean([r['cost_impact'] for r in portfolio_results]),
            'avg_transactions_per_period': np.mean([r['total_transactions'] for r in portfolio_results])
        })
    
    results_dict = {
        'portfolio_results': portfolio_results,
        'overall_metrics': overall_metrics,
        'strategy_config': {
            'percentile_threshold': percentile_threshold,
            'rebalance_frequency': rebalance_frequency,
            'model_type': results.get('model_type', 'Unknown'),
            'include_trading_costs': include_trading_costs
        }
    }
    
    # Ajouter la config des coûts si activés
    if include_trading_costs:
        results_dict['strategy_config'].update({
            'transaction_cost_bps': transaction_cost_bps,
            'bid_ask_spread_bps': bid_ask_spread_bps,
            'market_impact_bps': market_impact_bps,
            'borrowing_cost_annual': borrowing_cost_annual,
            'management_fee_annual': management_fee_annual
        })
    
    print(f"\n=== RÉSULTATS GLOBAUX DE LA STRATÉGIE ===")
    if include_trading_costs:
        print(f"Rendement moyen NET: {overall_metrics['avg_strategy_return']:.4f}")
        print(f"Rendement moyen BRUT: {overall_metrics['avg_strategy_return_gross']:.4f}")
        print(f"Impact moyen des coûts: {overall_metrics['avg_cost_impact']:.4f}")
        print(f"Coûts totaux cumulés: {overall_metrics['total_costs_cumulative']:.4f}")
    else:
        print(f"Rendement moyen par période: {overall_metrics['avg_strategy_return']:.4f}")
        
    print(f"Rendement cumulé total: {overall_metrics['total_cumulative_return']:.4f}")
    print(f"Volatilité: {overall_metrics['volatility']:.4f}")
    print(f"Ratio de Sharpe: {overall_metrics['sharpe_ratio']:.4f}")
    print(f"Alpha moyen: {overall_metrics['avg_alpha']:.4f}")
    print(f"Taux de réussite: {overall_metrics['win_rate']:.2%}")
    
    return results_dict

def analyze_strategy_performance(strategy_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Analyse détaillée des performances de la stratégie (avec ou sans coûts)
    """
    portfolio_results = strategy_results['portfolio_results']
    include_costs = strategy_results['strategy_config'].get('include_trading_costs', False)
    
    # Créer un DataFrame avec tous les résultats
    analysis_df = pd.DataFrame(portfolio_results)
    
    # Statistiques détaillées
    if include_costs:
        stats = {
            'Metric': [
                'Rendement Moyen NET (%)',
                'Rendement Moyen BRUT (%)',
                'Impact Moyen des Coûts (%)',
                'Rendement Cumulé NET (%)',
                'Rendement Cumulé BRUT (%)',
                'Volatilité (%)',
                'Ratio de Sharpe NET',
                'Alpha Moyen (%)',
                'Taux de Réussite (%)',
                'Coûts Totaux Cumulés (%)',
                'Transactions Moy./Période',
                'Nombre de Périodes'
            ],
            'Value': [
                f"{strategy_results['overall_metrics']['avg_strategy_return']*100:.2f}",
                f"{strategy_results['overall_metrics']['avg_strategy_return_gross']*100:.2f}",
                f"{strategy_results['overall_metrics']['avg_cost_impact']*100:.2f}",
                f"{strategy_results['overall_metrics']['total_cumulative_return']*100:.2f}",
                f"{strategy_results['overall_metrics']['total_cumulative_return_gross']*100:.2f}",
                f"{strategy_results['overall_metrics']['volatility']*100:.2f}",
                f"{strategy_results['overall_metrics']['sharpe_ratio']:.3f}",
                f"{strategy_results['overall_metrics']['avg_alpha']*100:.2f}",
                f"{strategy_results['overall_metrics']['win_rate']*100:.1f}",
                f"{strategy_results['overall_metrics']['total_costs_cumulative']*100:.2f}",
                f"{strategy_results['overall_metrics']['avg_transactions_per_period']:.1f}",
                f"{strategy_results['overall_metrics']['n_periods']}"
            ]
        }
    else:
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
    Visualise les résultats de la stratégie (avec ou sans coûts)
    """
    try:
        portfolio_results = strategy_results['portfolio_results']
        include_costs = strategy_results['strategy_config'].get('include_trading_costs', False)
        
        # Extraire les données pour les graphiques
        periods = [str(r['period']) for r in portfolio_results]
        strategy_returns = [r['strategy_return'] for r in portfolio_results]
        long_returns = [r['long_return'] for r in portfolio_results]
        short_returns = [r['short_return'] for r in portfolio_results]
        alphas = [r['alpha'] for r in portfolio_results]
        
        # Données spécifiques aux coûts si disponibles
        if include_costs:
            strategy_returns_gross = [r['strategy_return_gross'] for r in portfolio_results]
            costs = [r['total_costs'] for r in portfolio_results]
            cost_impacts = [r['cost_impact'] for r in portfolio_results]
        
        # Calculer les rendements cumulés
        cumulative_strategy = np.cumprod([1 + r for r in strategy_returns])
        
        # Adapter le layout selon les coûts
        if include_costs:
            plt.figure(figsize=(18, 15))
            subplot_layout = (3, 3)
        else:
            plt.figure(figsize=(16, 12))
            subplot_layout = (2, 3)
        
        # Graphique 1: Rendements par période
        plt.subplot(*subplot_layout, 1)
        if include_costs:
            plt.plot(periods, strategy_returns_gross, 'o-', label='Rendements Bruts', linewidth=2, markersize=6, alpha=0.7)
            plt.plot(periods, strategy_returns, 'o-', label='Rendements Nets', linewidth=2, markersize=6)
        else:
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
        plt.subplot(*subplot_layout, 2)
        plt.plot(periods, cumulative_strategy, 'o-', linewidth=2, markersize=6, color='green')
        if include_costs:
            cumulative_gross = np.cumprod([1 + r for r in strategy_returns_gross])
            plt.plot(periods, cumulative_gross, 'o--', linewidth=2, markersize=4, color='lightgreen', alpha=0.7, label='Brut')
            plt.legend()
        
        title = 'Performance Cumulée' + (' (Net vs Brut)' if include_costs else '')
        plt.title(title)
        plt.ylabel('Valeur du Portefeuille (base 1)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Graphique 3: Distribution des rendements
        plt.subplot(*subplot_layout, 3)
        plt.hist(strategy_returns, bins=min(15, len(strategy_returns)//2), alpha=0.7, edgecolor='black', label='Net' if include_costs else 'Stratégie')
        if include_costs:
            plt.hist(strategy_returns_gross, bins=min(15, len(strategy_returns_gross)//2), alpha=0.5, edgecolor='black', label='Brut')
            plt.legend()
        
        plt.axvline(x=np.mean(strategy_returns), color='red', linestyle='--', 
                   label=f'Moyenne: {np.mean(strategy_returns):.3f}')
        plt.title('Distribution des Rendements')
        plt.xlabel('Rendement de la Stratégie')
        plt.ylabel('Fréquence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graphique 4: Alpha par période
        plt.subplot(*subplot_layout, 4)
        colors = ['green' if alpha > 0 else 'red' for alpha in alphas]
        plt.bar(periods, alphas, color=colors, alpha=0.7, edgecolor='black')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.title('Alpha par Période')
        plt.ylabel('Alpha')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Graphique 5: Corrélation Long vs Short
        plt.subplot(*subplot_layout, 5)
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
        plt.subplot(*subplot_layout, 6)
        if include_costs:
            metrics = ['Rdt Net\n(%)', 'Rdt Brut\n(%)', 'Coûts\n(%)', 'Sharpe', 'Win Rate\n(%)']
            values = [
                np.mean(strategy_returns) * 100,
                np.mean(strategy_returns_gross) * 100,
                np.mean(costs) * 100,
                strategy_results['overall_metrics']['sharpe_ratio'],
                strategy_results['overall_metrics']['win_rate'] * 100
            ]
        else:
            metrics = ['Rendement\nMoyen', 'Volatilité', 'Ratio Sharpe', 'Win Rate']
            values = [
                np.mean(strategy_returns) * 100,
                np.std(strategy_returns) * 100,
                strategy_results['overall_metrics']['sharpe_ratio'],
                strategy_results['overall_metrics']['win_rate'] * 100
            ]
        
        bars = plt.bar(metrics, values, color=['blue', 'orange', 'red', 'green', 'purple'][:len(metrics)], alpha=0.7)
        plt.title('Métriques Clés')
        plt.ylabel('Valeur')
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Graphiques supplémentaires pour les coûts
        if include_costs:
            # Graphique 7: Évolution des coûts
            plt.subplot(*subplot_layout, 7)
            plt.bar(periods, costs, alpha=0.7, color='red', edgecolor='black')
            plt.title('Coûts par Période')
            plt.ylabel('Coûts')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Graphique 8: Impact des coûts vs rendement brut
            plt.subplot(*subplot_layout, 8)
            plt.scatter(strategy_returns_gross, cost_impacts, alpha=0.7, s=60)
            plt.xlabel('Rendement Brut')
            plt.ylabel('Impact des Coûts')
            plt.title('Impact des Coûts vs Rendement Brut')
            plt.grid(True, alpha=0.3)
            
            # Graphique 9: Transactions par période
            if 'total_transactions' in portfolio_results[0]:
                plt.subplot(*subplot_layout, 9)
                transactions = [r['total_transactions'] for r in portfolio_results]
                plt.bar(periods, transactions, alpha=0.7, color='orange', edgecolor='black')
                plt.title('Nombre de Transactions par Période')
                plt.ylabel('Transactions')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
        
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
