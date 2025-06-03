import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import Dict, List, Any
from scipy.stats import qmc


import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration de la stratégie CORRIGÉE pour données mensuelles
TRADING_CONFIG = {
    'correlation_window': 40,        # 2 ans en mois  correlation historique de plus de 0.7 sur la période
    'min_correlation': 0.7,          # Maintient haute corrélation
    'min_cointegration_pvalue': 0.09, # cointégration moyenne
    'zscore_entry': 2.2,
    'zscore_exit': 0.2,
    'max_position_size': 0.05,
    'stop_loss': 0.015,
    'min_observations': 100,          # 5 ans en mois 
    'test_size': 0.2,
    'lookback_window': 16,           # 1 an et demi en mois 
    'rebalance_frequency': 12        # Rebalancer annuellement 
}

class FlexibleMLP(nn.Module):
    """Modèle MLP flexible pour prédiction des rendements"""
    def __init__(self, layers: list, scale: float = 1.0, bias_scale: float = 0.0, activation=nn.GELU()):
        super(FlexibleMLP, self).__init__()
        self.layer_sizes = layers
        self.scale = scale
        self.bias_scale = bias_scale
        self.activation_fn = activation
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self._build_layers()
        self.reset_parameters()

    def _build_layers(self):
        for i in range(len(self.layer_sizes) - 1):
            layer = nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1])
            self.layers.append(layer)
            if i < len(self.layer_sizes) - 2:
                self.activations.append(self.activation_fn)
            else:
                self.activations.append(nn.Identity())

    def reset_parameters(self):
        for i, layer in enumerate(self.layers):
            nn.init.normal_(layer.weight, mean=0.0, std=self.scale / np.sqrt(self.layer_sizes[i]))
            nn.init.normal_(layer.bias, mean=0.0, std=self.bias_scale / np.sqrt(self.layer_sizes[i]))

    def forward(self, x, return_last_hidden=False):
        last_hidden = None
        for layer, activation in zip(self.layers[:-1], self.activations[:-1]):
            x = activation(layer(x))
            last_hidden = x
        x = self.layers[-1](x)
        if return_last_hidden:
            return x, last_hidden
        return x

class PairValidator:
    """Classe pour valider statistiquement les paires de trading"""
    
    @staticmethod
    def test_cointegration(series1, series2, significance_level=0.1):
        """Test de cointégration d'Engle-Granger adapté pour données mensuelles"""
        try:
            # Supprimer les NaN
            valid_data = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
            if len(valid_data) < 24:  # Au moins 2 ans de données mensuelles
                return False, 1.0, "Insufficient data"
            
            score, pvalue, _ = coint(valid_data['s1'], valid_data['s2'])
            is_cointegrated = pvalue < significance_level
            return is_cointegrated, pvalue, f"Cointegration test passed" if is_cointegrated else "Not cointegrated"
        except Exception as e:
            return False, 1.0, f"Cointegration test failed: {str(e)}"
    
    @staticmethod
    def test_spread_stationarity(series1, series2, significance_level=0.1):
        """Test de stationnarité du spread adapté pour données mensuelles"""
        try:
            valid_data = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
            if len(valid_data) < 24:  # Au moins 2 ans de données mensuelles
                return False, 1.0, "Insufficient data"
            
            spread = valid_data['s1'] - valid_data['s2']
            adf_stat, adf_pvalue, _, _, _, _ = adfuller(spread, autolag='AIC')
            is_stationary = adf_pvalue < significance_level
            return is_stationary, adf_pvalue, f"Spread is stationary" if is_stationary else "Spread is not stationary"
        except Exception as e:
            return False, 1.0, f"Stationarity test failed: {str(e)}"
    
    @staticmethod
    def validate_pair(series1, series2, min_correlation=0.7):
        """Validation complète d'une paire"""
        try:
            # Test de corrélation
            valid_data = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
            if len(valid_data) < 24:  # Au moins 2 ans de données mensuelles
                return {'valid': False, 'reason': 'Insufficient data'}
            
            correlation = valid_data['s1'].corr(valid_data['s2'])
            
            if abs(correlation) < min_correlation:
                return {'valid': False, 'reason': f'Low correlation: {correlation:.3f}'}
            
            # Test de cointégration
            cointegrated, coint_pvalue, coint_msg = PairValidator.test_cointegration(series1, series2)
            
            # Test de stationnarité
            stationary, stat_pvalue, stat_msg = PairValidator.test_spread_stationarity(series1, series2)
            
            # Version permissive: corrélation élevée ET (cointégration OU stationnarité)
            is_valid = abs(correlation) >= min_correlation and (cointegrated or stationary)
            
            return {
                'valid': is_valid,
                'correlation': correlation,
                'cointegrated': cointegrated,
                'coint_pvalue': coint_pvalue,
                'stationary': stationary,
                'stat_pvalue': stat_pvalue,
                'reason': f"{coint_msg}; {stat_msg}" if is_valid else f"Failed: {coint_msg}; {stat_msg}"
            }
        except Exception as e:
            return {'valid': False, 'reason': f'Validation error: {str(e)}'}

class PairSelector:
    """Classe pour sélectionner les paires de trading sans biais temporel"""
    
    def __init__(self, config=TRADING_CONFIG):
        self.config = config
        self.validator = PairValidator()
    
    def find_valid_pairs_at_date(self, data, current_date, lookback_months=36):
        """Trouve les paires valides à une date donnée sans look-ahead bias """
        
        logger.info(f"Selecting pairs at date: {current_date}")
        
        # Obtenir les données jusqu'à la date actuelle
        cutoff_date = pd.to_datetime(current_date)
        start_date = cutoff_date - pd.DateOffset(months=lookback_months)
        
        logger.debug(f"Date range: {start_date} to {cutoff_date}")
        
        historical_data = data[
            (pd.to_datetime(data['MthCalDt']) >= start_date) & 
            (pd.to_datetime(data['MthCalDt']) <= cutoff_date)
        ].copy()
        
        logger.debug(f"Historical data: {len(historical_data)} records")
        
        if len(historical_data) == 0:
            logger.warning("No historical data found for the date range")
            return []
        
        # Pivoter pour avoir les rendements par ticker
        pivot_data = historical_data.pivot_table(
            index='MthCalDt', 
            columns='Ticker', 
            values='MthRet'
        )
        
        logger.debug(f"Pivot data shape: {pivot_data.shape}")
        logger.debug(f"Available tickers: {len(pivot_data.columns)}")
        
        # Garder seulement les tickers avec suffisamment de données
        min_obs = max(36, lookback_months // 2)  # Au moins 2 ans ou la moitié de la fenêtre
        ticker_counts = pivot_data.count()
        valid_tickers = ticker_counts[ticker_counts >= min_obs].index.tolist()
        
        logger.info(f"Tickers with >= {min_obs} observations: {len(valid_tickers)}")
        
        if len(valid_tickers) < 2:
            logger.warning(f"Not enough valid tickers: {len(valid_tickers)}")
            return []
        
        # Limiter le nombre de tickers pour accélérer le processus si nécessaire
        if len(valid_tickers) > 100:
            # Prendre les 100 tickers avec le plus de données
            top_tickers = ticker_counts.nlargest(1000).index.tolist() #10000 marche super bien 
            valid_tickers = [t for t in valid_tickers if t in top_tickers]
            logger.info(f"Limited to top 100 tickers by data availability")
        
        pivot_data = pivot_data[valid_tickers]
        
        # Tester toutes les paires possibles
        valid_pairs = []
        total_pairs = len(valid_tickers) * (len(valid_tickers) - 1) // 2
        logger.info(f"Testing {total_pairs} possible pairs")
        
        pairs_tested = 0
        for i, ticker1 in enumerate(valid_tickers):
            for ticker2 in valid_tickers[i+1:]:
                pairs_tested += 1
                
                if pairs_tested % 500 == 0:
                    logger.info(f"Tested {pairs_tested}/{total_pairs} pairs, found {len(valid_pairs)} valid")
                
                series1 = pivot_data[ticker1].dropna()
                series2 = pivot_data[ticker2].dropna()
                
                # Aligner les séries
                aligned_data = pd.DataFrame({
                    'ticker1': series1,
                    'ticker2': series2
                }).dropna()
                
                if len(aligned_data) < min_obs:
                    continue
                
                # Valider la paire
                validation_result = self.validator.validate_pair(
                    aligned_data['ticker1'], 
                    aligned_data['ticker2'],
                    self.config['min_correlation']
                )
                
                if validation_result['valid']:
                    valid_pairs.append({
                        'ticker1': ticker1,
                        'ticker2': ticker2,
                        'selection_date': current_date,
                        'correlation': validation_result['correlation'],
                        'coint_pvalue': validation_result['coint_pvalue'],
                        'stat_pvalue': validation_result['stat_pvalue'],
                        'observations': len(aligned_data),
                        'reason': validation_result['reason']
                    })
                    logger.debug(f"Valid pair found: {ticker1}-{ticker2}, corr={validation_result['correlation']:.3f}")
        
        logger.info(f"Found {len(valid_pairs)} valid pairs out of {pairs_tested} tested at {current_date}")
        return valid_pairs

class PredictionEngine:
    """Moteur de prédiction utilisant les MLPs"""
    
    def __init__(self, config=TRADING_CONFIG):
        self.config = config
        self.models = {}
        self.scalers = {}
    
    def train_model(self, num_epochs, train_loader, criterion, optimizer, model, ridge_penalty=0.001):
        """Entraîne un modèle MLP"""
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for inputs, targets in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets) + ridge_penalty * sum(p.pow(2.0).sum() for p in model.parameters())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = epoch_loss / len(train_loader)
                logger.debug(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    def predict_returns_for_ticker(self, ticker_data, ticker, epochs=50, batch_size=32, num_seeds=5):
        """Prédit les rendements pour un ticker donné"""
        try:
            # Préparer les features et target
            feature_cols = [col for col in ticker_data.columns if col not in ['MthCalDt', 'sprtrn', 'Ticker', 'MthRet']]
            X = ticker_data[feature_cols].fillna(0)
            y = ticker_data['MthRet'].fillna(0)
            dates = ticker_data['MthCalDt']
            
            # Ajuster le seuil minimum pour données mensuelles
            if len(X) < self.config['min_observations']:
                logger.warning(f"Insufficient data for ticker {ticker}: {len(X)} records")
                return None
            
            # Split des données
            split_idx = int(len(X) * (1 - self.config['test_size']))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            dates_train, dates_test = dates.iloc[:split_idx], dates.iloc[split_idx:]
            
            # Standardisation
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Conversion en tenseurs
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            
            # DataLoader
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Entraînement avec plusieurs seeds
            test_predictions_all = []
            input_size = X_train.shape[1]
            
            for seed in range(num_seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                model = FlexibleMLP(layers=[input_size, 64, 32, 1])
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                self.train_model(epochs, train_loader, criterion, optimizer, model)
                
                # Prédictions
                model.eval()
                with torch.no_grad():
                    test_predictions = model(X_test_tensor).squeeze().numpy()
                    if len(y_test) == 1:
                        test_predictions = np.array([test_predictions])
                    test_predictions_all.append(test_predictions)
            
            # Moyenne des prédictions
            test_predictions_avg = np.mean(test_predictions_all, axis=0)
            
            # Créer le DataFrame de résultats
            predictions_df = pd.DataFrame({
                'Date': dates_test.values,
                'Actual_MthRet': y_test.values,
                'Predicted_MthRet': test_predictions_avg
            })
            
            # Sauvegarder le modèle et le scaler
            self.scalers[ticker] = scaler
            
            return {
                'ticker': ticker,
                'predictions_df': predictions_df,
                'mse': mean_squared_error(y_test.values, test_predictions_avg),
                'test_start_date': dates_test.iloc[0]
            }
            
        except Exception as e:
            logger.error(f"Error predicting returns for ticker {ticker}: {str(e)}")
            return None

class RiskManager:
    """Gestionnaire de risque pour les positions"""
    
    def __init__(self, config=TRADING_CONFIG):
        self.config = config
    
    def calculate_position_size(self, volatility, base_size=None):
        """Calcule la taille de position basée sur la volatilité"""
        if base_size is None:
            base_size = self.config['max_position_size']
        
        if volatility <= 0:
            return base_size
        
        # Position inversement proportionnelle à la volatilité
        vol_adjusted_size = base_size * (0.02 / max(volatility, 0.005))  # 2% vol de référence
        return min(vol_adjusted_size, self.config['max_position_size'])
    
    def check_stop_loss(self, cumulative_pnl, entry_pnl=0):
        """Vérifie si le stop-loss est déclenché"""
        drawdown = cumulative_pnl - entry_pnl
        return drawdown < -self.config['stop_loss']
    
    def apply_risk_limits(self, signal, volatility, cumulative_pnl, entry_pnl=0):
        """Applique les limites de risque à un signal"""
        # Vérifier le stop-loss
        if self.check_stop_loss(cumulative_pnl, entry_pnl):
            return 0  # Fermer la position
        
        # Ajuster la taille de position selon la volatilité
        if signal != 0:
            position_size = self.calculate_position_size(volatility)
            return np.sign(signal) * position_size
        
        return signal

class PairsTrader:
    """Trader de paires avec logique de mean reversion"""
    
    def __init__(self, config=TRADING_CONFIG):
        self.config = config
        self.risk_manager = RiskManager(config)
    
    def calculate_spread_zscore(self, series1, series2, window=12):  
        """Calcule le z-score du spread sans look-ahead bias """
        spread = series1 - series2
        
        # Moyennes mobiles avec décalage pour éviter le look-ahead bias
        spread_mean = spread.shift(1).rolling(window=window, min_periods=window).mean()
        spread_std = spread.shift(1).rolling(window=window, min_periods=window).std()
        
        # Z-score sans biais temporel
        z_score = (spread - spread_mean) / spread_std
        return z_score, spread, spread_mean, spread_std
    
    def generate_trading_signals(self, ticker1_data, ticker2_data, predictions1, predictions2):
        """Génère les signaux de trading pour une paire"""
        
        # Aligner toutes les données
        combined_data = pd.DataFrame({
            'date': ticker1_data['Date'],
            'ret1': ticker1_data['Actual_MthRet'],
            'ret2': ticker2_data['Actual_MthRet'],
            'pred1': predictions1,
            'pred2': predictions2
        }).dropna()
        
        if len(combined_data) < self.config['lookback_window'] + 1:
            logger.warning("Insufficient data for trading signals")
            return pd.DataFrame()
        
        # Calculer le z-score du spread
        z_score, spread, spread_mean, spread_std = self.calculate_spread_zscore(
            combined_data['ret1'], 
            combined_data['ret2'], 
            self.config['lookback_window']
        )
        
        combined_data['z_score'] = z_score
        combined_data['spread'] = spread
        
        # Générer les signaux de trading
        signals = pd.DataFrame(index=combined_data.index)
        signals['position'] = 0
        signals['entry_reason'] = ''
        
        # Logique de mean reversion corrigée
        entry_threshold = self.config['zscore_entry']
        exit_threshold = self.config['zscore_exit']
        
        # Signaux d'entrée
        # Long spread (long ticker1, short ticker2) quand z_score < -entry_threshold
        long_spread_condition = (z_score < -entry_threshold) & (z_score.notna())
        signals.loc[long_spread_condition, 'position'] = 1
        signals.loc[long_spread_condition, 'entry_reason'] = 'Long spread (mean reversion)'
        
        # Short spread (short ticker1, long ticker2) quand z_score > entry_threshold
        short_spread_condition = (z_score > entry_threshold) & (z_score.notna())
        signals.loc[short_spread_condition, 'position'] = -1
        signals.loc[short_spread_condition, 'entry_reason'] = 'Short spread (mean reversion)'
        
        # Signaux de sortie
        exit_condition = (abs(z_score) < exit_threshold) & (z_score.notna())
        signals.loc[exit_condition, 'position'] = 0
        signals.loc[exit_condition, 'entry_reason'] = 'Exit (mean reversion complete)'
        
        # Propager les signaux (maintenir la position jusqu'au signal de sortie)
        signals['position'] = signals['position'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        # Ajouter les données de marché
        signals['z_score'] = z_score
        signals['ret1'] = combined_data['ret1']
        signals['ret2'] = combined_data['ret2']
        signals['pred1'] = combined_data['pred1']
        signals['pred2'] = combined_data['pred2']
        signals['date'] = combined_data['date']
        
        return signals.dropna()
    
    def calculate_pair_returns(self, signals, ticker1, ticker2):
        """Calcule les rendements de la stratégie de paire"""
        if len(signals) == 0:
            return pd.DataFrame()
        
        # Calcul des rendements de la stratégie
        # Position = 1: Long ticker1, Short ticker2
        # Position = -1: Short ticker1, Long ticker2
        strategy_returns = signals['position'].shift(1) * (signals['ret1'] - signals['ret2'])
        
        # Calculer la volatilité pour la gestion du risque - CORRIGÉ: fenêtre adaptée aux données mensuelles
        spread_volatility = (signals['ret1'] - signals['ret2']).rolling(12).std()  # 1 an au lieu de 20 jours
        
        # Appliquer la gestion du risque
        adjusted_returns = []
        cumulative_pnl = 0
        entry_pnl = 0
        
        for i, (idx, row) in enumerate(signals.iterrows()):
            if i == 0:
                adjusted_returns.append(0)
                continue
            
            current_return = strategy_returns.iloc[i]
            current_vol = spread_volatility.iloc[i] if not pd.isna(spread_volatility.iloc[i]) else 0.02
            
            # Appliquer les limites de risque
            if row['position'] != 0:
                # Nouvelle position ou position existante
                if signals['position'].iloc[i-1] == 0:  # Nouvelle position
                    entry_pnl = cumulative_pnl
                
                risk_adjusted_return = self.risk_manager.apply_risk_limits(
                    current_return, current_vol, cumulative_pnl, entry_pnl
                )
            else:
                risk_adjusted_return = 0
            
            adjusted_returns.append(risk_adjusted_return)
            cumulative_pnl += risk_adjusted_return
        
        # Créer le DataFrame de résultats
        results_df = signals.copy()
        results_df['strategy_returns'] = adjusted_returns
        results_df['cumulative_returns'] = np.cumsum(adjusted_returns)
        results_df['ticker1'] = ticker1
        results_df['ticker2'] = ticker2
        
        return results_df

class Backtester:
    """Classe pour le backtesting complet de la stratégie"""
    
    def __init__(self, config=TRADING_CONFIG):
        self.config = config
    
    def calculate_performance_metrics(self, returns, benchmark_returns=None):
        """Calcule les métriques de performance"""
        if len(returns) == 0 or returns.std() == 0:
            return {}
        
        # Métriques de base
        total_return = returns.sum()
        annualized_return = returns.mean() * 12  # Données mensuelles
        volatility = returns.std() * np.sqrt(12)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = returns.cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Ratio profit/perte
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        # Test de significativité
        t_stat, p_value = stats.ttest_1samp(returns.dropna(), 0)
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'num_trades': len(returns[returns != 0]),
            'significant': p_value < 0.05,
            'p_value': p_value
        }
        
        # Métriques vs benchmark si fourni
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            # Aligner les séries
            aligned_data = pd.DataFrame({
                'strategy': returns,
                'benchmark': benchmark_returns
            }).dropna()
            
            if len(aligned_data) > 10:
                # Calcul alpha et beta
                covariance = np.cov(aligned_data['strategy'], aligned_data['benchmark'])[0, 1]
                benchmark_var = aligned_data['benchmark'].var()
                
                beta = covariance / benchmark_var if benchmark_var > 0 else 0
                alpha = aligned_data['strategy'].mean() - beta * aligned_data['benchmark'].mean()
                
                # Information ratio
                excess_returns = aligned_data['strategy'] - aligned_data['benchmark']
                information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
                
                metrics.update({
                    'alpha': alpha * 12,  # Annualisé
                    'beta': beta,
                    'information_ratio': information_ratio * np.sqrt(12)
                })
        
        return metrics

def main():
    """Fonction principale"""
    
    # Configuration des dossiers
    folder = os.path.expanduser('~/Desktop/Machine Learning/data_ML_Project')
    folder1 = os.path.expanduser('~/Desktop/Machine Learning/data_filtered')
    output_dir = os.path.expanduser('~/Desktop/Machine Learning/predictions_improved_2')
    os.makedirs(output_dir, exist_ok=True)
    
    # Chargement des données
    logger.info("Loading data...")
    try:
        main_csv = os.path.join(folder, 'monthly_crsp.csv')
        jkp_csv = os.path.join(folder1, 'JKP_filtered.csv')
        zimmer_csv = os.path.join(folder1, 'Zimmer_filtered.csv')
        
        data_main = pd.read_csv(main_csv)
        data_jkp = pd.read_csv(jkp_csv)
        data_zimmer = pd.read_csv(zimmer_csv)
        
    except FileNotFoundError as e:
        logger.error(f"Data files not found: {e}")
        return
    
    # Préprocessing
    logger.info("Preprocessing data...")
    data_main['date'] = pd.to_datetime(data_main['MthCalDt'])
    data_jkp['date'] = pd.to_datetime(data_jkp['date'])
    data_zimmer['date'] = pd.to_datetime(data_zimmer['date'], errors='coerce')
    
    start_date = pd.to_datetime('2010-01-01')
    data_main = data_main[data_main['date'] >= start_date]
    data_jkp = data_jkp[data_jkp['date'] >= start_date]
    data_zimmer = data_zimmer[data_zimmer['date'] >= start_date]
    
    for df in [data_main, data_jkp, data_zimmer]:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
    
    # Fusion des données
    merged = data_main.merge(
        data_jkp.drop(columns=['date']),
        on=['year', 'month'],
        how='left',
        suffixes=('', '_jkp')
    ).merge(
        data_zimmer.drop(columns=['date']),
        on=['year', 'month'],
        how='left',
        suffixes=('', '_zimmer')
    )
    
    # Filtrer les tickers avec suffisamment de données - CORRIGÉ pour données mensuelles
    merged = merged[merged.groupby('Ticker')['Ticker'].transform('count') >= TRADING_CONFIG['min_observations']]
    
    # Nettoyer les colonnes
    columns_to_drop = ['PERMNO', 'HdrCUSIP', 'CUSIP', 'TradingSymbol', 'PERMCO', 'SICCD', 'NAICS', 'date', 'year', 'month']
    merged.drop(columns=[col for col in columns_to_drop if col in merged.columns], inplace=True)
    
    # Décaler les rendements (pas de look-ahead)
    merged[['MthRet', 'sprtrn']] = merged.groupby('Ticker')[['MthRet', 'sprtrn']].shift(1).fillna(0)
    
    logger.info(f"Data loaded: {len(merged)} records, {merged['Ticker'].nunique()} unique tickers")
    
    # Initialiser les composants
    pair_selector = PairSelector(TRADING_CONFIG)
    prediction_engine = PredictionEngine(TRADING_CONFIG)
    pairs_trader = PairsTrader(TRADING_CONFIG)
    backtester = Backtester(TRADING_CONFIG)
    
    # Sélection des paires sans biais temporel 
    logger.info("Selecting trading pairs without look-ahead bias...")
    
    # Obtenir les dates uniques pour la sélection des paires
    unique_dates = sorted(merged['MthCalDt'].unique())
    
    # CORRIGÉ: Commencer après avoir assez de données et utiliser des intervalles mensuels
    start_idx = TRADING_CONFIG['correlation_window']  # 36 mois minimum
    rebalance_dates = unique_dates[start_idx::TRADING_CONFIG['rebalance_frequency']]  # Tous les 12 mois
    
    logger.info(f"Rebalance dates: {len(rebalance_dates)} dates starting from {rebalance_dates[0] if rebalance_dates else 'None'}")
    
    all_pairs = []
    # Tester plus de dates pour avoir plus de chances de trouver des paires
    for rebalance_date in rebalance_dates:  # Tester les 10 premières dates
        logger.info(f"Processing rebalance date: {rebalance_date}")
        pairs_at_date = pair_selector.find_valid_pairs_at_date(merged, rebalance_date)
        all_pairs.extend(pairs_at_date)
        
        # Si on a trouvé assez de paires, on peut s'arrêter
        if len(all_pairs) >=500:  # Au moins 500 paires pour continuer
            logger.info(f"Found sufficient pairs ({len(all_pairs)}), proceeding...")
            break
    
    if not all_pairs:
        logger.error("No valid pairs found!")
        logger.info("Try reducing min_correlation or increasing min_cointegration_pvalue in TRADING_CONFIG")
        return
    
    pairs_df = pd.DataFrame(all_pairs)
    pairs_df.to_csv(os.path.join(output_dir, 'selected_pairs.csv'), index=False)
    logger.info(f"Selected {len(pairs_df)} valid pairs")
    
    # Afficher quelques statistiques sur les paires trouvées
    logger.info("=== PAIR SELECTION STATISTICS ===")
    logger.info(f"Average correlation: {pairs_df['correlation'].mean():.3f}")
    logger.info(f"Min correlation: {pairs_df['correlation'].min():.3f}")
    logger.info(f"Max correlation: {pairs_df['correlation'].max():.3f}")
    logger.info(f"Average observations per pair: {pairs_df['observations'].mean():.1f}")
    
    # Prédictions pour les tickers des paires sélectionnées
    logger.info("Generating predictions for selected tickers...")
    unique_tickers = pd.concat([pairs_df['ticker1'], pairs_df['ticker2']]).unique()
    
    predictions_dict = {}
    prediction_results = []
    
    for ticker in tqdm(unique_tickers, desc="Generating predictions"):
        ticker_data = merged[merged['Ticker'] == ticker].copy()
        if len(ticker_data) < TRADING_CONFIG['min_observations']:
            logger.warning(f"Skipping ticker {ticker}: insufficient data")
            continue
        
        result = prediction_engine.predict_returns_for_ticker(ticker_data, ticker)
        if result is not None:
            predictions_dict[ticker] = result['predictions_df']
            prediction_results.append(result)
            
            # Sauvegarder les prédictions individuelles
            #result['predictions_df'].to_csv(
                #os.path.join(output_dir, f'{ticker}_predictions.csv'), index=False
            #)
    
    logger.info(f"Generated predictions for {len(predictions_dict)} tickers")
    
    # Résumé des performances de prédiction
    if prediction_results:
        pred_summary = pd.DataFrame([
            {'ticker': r['ticker'], 'mse': r['mse'], 'test_start': r['test_start_date']}
            for r in prediction_results
        ])
        pred_summary.to_csv(os.path.join(output_dir, 'prediction_summary.csv'), index=False)
        
        logger.info(f"Prediction MSE statistics:")
        logger.info(f"Mean: {pred_summary['mse'].mean():.6f}")
        logger.info(f"Std: {pred_summary['mse'].std():.6f}")
        logger.info(f"Min: {pred_summary['mse'].min():.6f}")
        logger.info(f"Max: {pred_summary['mse'].max():.6f}")
    
    # Trading des paires
    logger.info("Executing pairs trading strategy...")
    
    trading_results = []
    performance_summary = []
    
    for _, pair_info in pairs_df.iterrows():
        ticker1 = pair_info['ticker1']
        ticker2 = pair_info['ticker2']
        
        # Vérifier que nous avons les prédictions pour les deux tickers
        if ticker1 not in predictions_dict or ticker2 not in predictions_dict:
            logger.warning(f"Skipping pair {ticker1}-{ticker2}: missing predictions")
            continue
        
        pred_df1 = predictions_dict[ticker1]
        pred_df2 = predictions_dict[ticker2]
        
        # Aligner les données de prédiction par date
        aligned_predictions = pred_df1.merge(
            pred_df2, 
            on='Date', 
            suffixes=('_1', '_2'),
            how='inner'
        )
        
        if len(aligned_predictions) < TRADING_CONFIG['lookback_window'] + 5:
            logger.warning(f"Skipping pair {ticker1}-{ticker2}: insufficient aligned data")
            continue
        
        # Générer les signaux de trading
        signals = pairs_trader.generate_trading_signals(
            aligned_predictions[['Date', 'Actual_MthRet_1']].rename(columns={'Actual_MthRet_1': 'Actual_MthRet'}),
            aligned_predictions[['Date', 'Actual_MthRet_2']].rename(columns={'Actual_MthRet_2': 'Actual_MthRet'}),
            aligned_predictions['Predicted_MthRet_1'],
            aligned_predictions['Predicted_MthRet_2']
        )
        
        if len(signals) == 0:
            logger.warning(f"No trading signals generated for pair {ticker1}-{ticker2}")
            continue
        
        # Calculer les rendements de la stratégie
        pair_results = pairs_trader.calculate_pair_returns(signals, ticker1, ticker2)
        
        if len(pair_results) == 0:
            continue
        
        # Sauvegarder les résultats de trading pour cette paire
        #pair_results.to_csv(
            #os.path.join(output_dir, f'{ticker1}_{ticker2}_trading_results.csv'),
            #index=False
        #)
        
        trading_results.append(pair_results)
        
        # Calculer les métriques de performance pour cette paire
        strategy_returns = pair_results['strategy_returns'].dropna()
        if len(strategy_returns) > 0 and strategy_returns.std() > 0:
            
            # Obtenir les rendements de marché alignés
            market_returns = merged[merged['Ticker'] == ticker1]['sprtrn']
            market_dates = merged[merged['Ticker'] == ticker1]['MthCalDt']
            market_data = pd.DataFrame({'Date': market_dates, 'sprtrn': market_returns})
            
            # Aligner avec les dates de trading
            aligned_market = pair_results[['date']].merge(
                market_data, 
                left_on='date', 
                right_on='Date',
                how='left'
            )['sprtrn'].fillna(0)
            
            pair_metrics = backtester.calculate_performance_metrics(
                strategy_returns,
                aligned_market
            )
            
            pair_metrics.update({
                'ticker1': ticker1,
                'ticker2': ticker2,
                'correlation': pair_info['correlation'],
                'num_observations': len(strategy_returns),
                'selection_date': pair_info['selection_date']
            })
            
            performance_summary.append(pair_metrics)
            
            logger.info(f"Pair {ticker1}-{ticker2}: Sharpe={pair_metrics.get('sharpe_ratio', 0):.3f}, "
                       f"Total Return={pair_metrics.get('total_return', 0):.4f}")
    
    # Analyser les résultats globaux
    if trading_results and performance_summary:
        logger.info("Analyzing overall strategy performance...")
        
        # Sauvegarder le résumé des performances
        performance_df = pd.DataFrame(performance_summary)
        performance_df.to_csv(os.path.join(output_dir, 'pairs_performance_summary.csv'), index=False)
        
        # Consolider tous les rendements
        all_returns = []
        all_dates = []
        
        for pair_result in trading_results:
            returns_with_dates = pair_result[['date', 'strategy_returns']].copy()
            returns_with_dates = returns_with_dates[returns_with_dates['strategy_returns'] != 0]
            all_returns.extend(returns_with_dates['strategy_returns'].tolist())
            all_dates.extend(returns_with_dates['date'].tolist())
        
        if all_returns:
            # Créer une série temporelle consolidée
            consolidated_returns = pd.DataFrame({
                'date': all_dates,
                'returns': all_returns
            }).groupby('date')['returns'].mean().reset_index()  # Moyenne si plusieurs trades le même jour
            
            # Métriques globales
            overall_metrics = backtester.calculate_performance_metrics(
                pd.Series(consolidated_returns['returns'])
            )
            
            logger.info("=== OVERALL STRATEGY PERFORMANCE ===")
            logger.info(f"Total Return: {overall_metrics.get('total_return', 0):.4f}")
            logger.info(f"Annualized Return: {overall_metrics.get('annualized_return', 0):.4f}")
            logger.info(f"Sharpe Ratio: {overall_metrics.get('sharpe_ratio', 0):.3f}")
            logger.info(f"Max Drawdown: {overall_metrics.get('max_drawdown', 0):.4f}")
            logger.info(f"Win Rate: {overall_metrics.get('win_rate', 0):.3f}")
            logger.info(f"Number of Trades: {overall_metrics.get('num_trades', 0)}")
            logger.info(f"Statistically Significant: {overall_metrics.get('significant', False)}")
            
            # Sauvegarder les métriques globales
            overall_df = pd.DataFrame([overall_metrics])
            overall_df.to_csv(os.path.join(output_dir, 'overall_performance.csv'), index=False)
            
            # Graphiques de performance
            create_performance_plots(consolidated_returns, performance_df, output_dir)
            
            # Statistiques des paires
            logger.info("=== PAIRS STATISTICS ===")
            logger.info(f"Number of profitable pairs: {(performance_df['total_return'] > 0).sum()}")
            logger.info(f"Best pair return: {performance_df['total_return'].max():.4f}")
            logger.info(f"Worst pair return: {performance_df['total_return'].min():.4f}")
            logger.info(f"Average pair Sharpe: {performance_df['sharpe_ratio'].mean():.3f}")
            
        else:
            logger.warning("No trading returns generated")
    
    else:
        logger.warning("No successful pair trading results")
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")

def create_performance_plots(consolidated_returns, performance_df, output_dir):
    """Crée les graphiques de performance"""
    
    # 1. Rendements cumulés de la stratégie
    plt.figure(figsize=(12, 8))
    
    # Sous-graphique 1: Rendements cumulés
    plt.subplot(2, 2, 1)
    cumulative_returns = consolidated_returns['returns'].cumsum()
    plt.plot(pd.to_datetime(consolidated_returns['date']), cumulative_returns, 
             label='Strategy', linewidth=2)
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Sous-graphique 2: Distribution des rendements
    plt.subplot(2, 2, 2)
    plt.hist(consolidated_returns['returns'], bins=30, alpha=0.7, edgecolor='black')
    plt.title('Returns Distribution')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Sous-graphique 3: Drawdown
    plt.subplot(2, 2, 3)
    running_max = cumulative_returns.expanding().max()
    drawdown = cumulative_returns - running_max
    plt.fill_between(pd.to_datetime(consolidated_returns['date']), drawdown, 0, 
                     alpha=0.7, color='red', label='Drawdown')
    plt.title('Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Sous-graphique 4: Performance par paire
    plt.subplot(2, 2, 4)
    if len(performance_df) > 0:
        pair_labels = [f"{row['ticker1']}-{row['ticker2']}" for _, row in performance_df.iterrows()]
        plt.barh(range(len(performance_df)), performance_df['total_return'])
        plt.yticks(range(len(performance_df)), pair_labels)
        plt.title('Return by Pair')
        plt.xlabel('Total Return')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_performance_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Graphique détaillé des métriques de performance
    if len(performance_df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Sharpe ratios
        axes[0, 0].bar(range(len(performance_df)), performance_df['sharpe_ratio'])
        axes[0, 0].set_title('Sharpe Ratio by Pair')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Max drawdowns
        axes[0, 1].bar(range(len(performance_df)), performance_df['max_drawdown'])
        axes[0, 1].set_title('Max Drawdown by Pair')
        axes[0, 1].set_ylabel('Max Drawdown')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Win rates
        axes[1, 0].bar(range(len(performance_df)), performance_df['win_rate'])
        axes[1, 0].set_title('Win Rate by Pair')
        axes[1, 0].set_ylabel('Win Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Number of trades
        axes[1, 1].bar(range(len(performance_df)), performance_df['num_trades'])
        axes[1, 1].set_title('Number of Trades by Pair')
        axes[1, 1].set_ylabel('Number of Trades')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Labels pour les x-axis
        pair_labels = [f"{row['ticker1']}-{row['ticker2']}" for _, row in performance_df.iterrows()]
        for ax in axes.flat:
            ax.set_xticks(range(len(performance_df)))
            ax.set_xticklabels(pair_labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pairs_performance_details.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("Performance plots saved")


class MetaLearningKnowledgeBase:
    def __init__(self, history_file: str = "trading_optimization_history.json"):
        self.history_file = history_file
        self.optimization_history = self._load_history()
        self.param_names = []
        self.classifier = None

    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                return pd.read_json(self.history_file).to_dict(orient='records')
            except:
                return []
        return []

    def save_history(self):
        pd.DataFrame(self.optimization_history).to_json(self.history_file, orient='records', indent=2)

    def record_optimization(self, config: Dict, results: Dict):
        self.optimization_history.append({'config': config, 'results': results})
        self.save_history()

    def train_config_classifier(self, threshold: float = 1.0):
        records = [r for r in self.optimization_history if 'sharpe_ratio' in r.get('results', {})]
        if len(records) < 20:
            logging.warning("Pas assez d'optimisations pour entraîner un modèle.")
            return

        X, y = [], []
        for r in records:
            config = r['config']
            score = r['results'].get('sharpe_ratio', 0)
            X.append(list(config.values()))
            y.append(1 if score >= threshold else 0)

        self.param_names = list(records[0]['config'].keys())
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X, y)
        logging.info("Classifieur entraîné avec succès.")

    def predict_config_quality(self, config: Dict[str, Any]) -> float:
        if self.classifier is None:
            return 0.0
        vec = [config.get(k, 0) for k in self.param_names]
        return self.classifier.predict_proba([vec])[0][1]

    def suggest_high_quality_configs(self, param_space: Dict[str, Dict], n: int = 5) -> List[Dict]:
        if self.classifier is None:
            return []

        sampler = qmc.LatinHypercube(d=len(param_space), scramble=True)
        samples = sampler.random(n * 20)

        param_names = list(param_space.keys())
        configs_with_probs = []

        for s in samples:
            config = {}
            for i, param in enumerate(param_names):
                p_info = param_space[param]
                val = p_info['low'] + s[i] * (p_info['high'] - p_info['low'])
                if p_info['type'] == 'int':
                    val = int(round(val))
                elif 'step' in p_info:
                    val = round(val / p_info['step']) * p_info['step']
                config[param] = val

            prob = self.predict_config_quality(config)
            configs_with_probs.append((config, prob))

        sorted_configs = sorted(configs_with_probs, key=lambda x: x[1], reverse=True)
        return [cfg for cfg, _ in sorted_configs[:n]]

def optimize_trading_config(n_trials: int = 20):
    logger.info("=== LANCEMENT OPTIMISATION ML DES PARAMÈTRES ===")
    
    param_space = {
        'correlation_window': {'type': 'int', 'low': 12, 'high': 60, 'step': 6},
        'min_correlation': {'type': 'float', 'low': 0.6, 'high': 0.98, 'step': 0.05},
        'min_cointegration_pvalue': {'type': 'float', 'low': 0.01, 'high': 0.15, 'step': 0.01},
        'zscore_entry': {'type': 'float', 'low': 1.0, 'high': 3.0, 'step': 0.2},
        'zscore_exit': {'type': 'float', 'low': 0.1, 'high': 1.0, 'step': 0.1},
        'max_position_size': {'type': 'float', 'low': 0.01, 'high': 0.10, 'step': 0.01},
        'stop_loss': {'type': 'float', 'low': 0.005, 'high': 0.03, 'step': 0.005},
        'min_observations': {'type': 'int', 'low': 36, 'high': 120, 'step': 12},
        'lookback_window': {'type': 'int', 'low': 6, 'high': 24, 'step': 3},
        'rebalance_frequency': {'type': 'int', 'low': 3, 'high': 24, 'step': 3}
    }

    meta = MetaLearningKnowledgeBase()
    meta.train_config_classifier(threshold=1.0)

    suggested_configs = meta.suggest_high_quality_configs(param_space, n=n_trials)

    if not suggested_configs:
        logger.warning("Aucune configuration intelligente trouvée, échantillonnage aléatoire...")
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=len(param_space), scramble=True)
        samples = sampler.random(n_trials)
        param_names = list(param_space.keys())
        suggested_configs = []

        for s in samples:
            config = {}
            for i, p in enumerate(param_names):
                pinfo = param_space[p]
                val = pinfo['low'] + s[i] * (pinfo['high'] - pinfo['low'])
                if pinfo['type'] == 'int':
                    val = int(round(val))
                elif 'step' in pinfo:
                    val = round(val / pinfo['step']) * pinfo['step']
                config[p] = val
            suggested_configs.append(config)

    for idx, config in enumerate(suggested_configs):
        logger.info(f"=== ÉVALUATION CONFIGURATION {idx+1}/{len(suggested_configs)} ===")
        try:
            # Met à jour la config globale
            global TRADING_CONFIG
            original_config = TRADING_CONFIG.copy()
            TRADING_CONFIG.update(config)

            # Exécute la stratégie avec la nouvelle config
            main()

            # Lecture des résultats globaux
            result_path = os.path.expanduser('~/Desktop/Machine Learning/predictions_improved_2/overall_performance.csv')
            if os.path.exists(result_path):
                df = pd.read_csv(result_path)
                if not df.empty:
                    result = df.iloc[0].to_dict()
                    meta.record_optimization(config, result)
                    logger.info(f"Sharpe: {result.get('sharpe_ratio', 0):.3f} | Return: {result.get('total_return', 0):.4f}")
                else:
                    logger.warning("Résultat vide, ignoré.")
            else:
                logger.warning("Fichier de résultat non trouvé.")

        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation : {e}")
        finally:
            TRADING_CONFIG = original_config

    logger.info("=== OPTIMISATION TERMINÉE ===")


if __name__ == "__main__":
    main()  # on utilise pour essayé la meilleur configuration trouvée  avec optimize_trading_config
    #optimize_trading_config(n_trials=20)


#1. Génère 20 configurations (intelligentes ou aléatoires)
#2. Pour chaque config :
    ####a. Applique la config à TRADING_CONFIG
    ###b. Lance main() → exécution complète de la stratégie
    ##c. Lit les résultats du fichier de performance
    #d. Enregistre les résultats dans un historique JSON

