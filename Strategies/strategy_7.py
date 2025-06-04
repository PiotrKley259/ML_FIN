import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from itertools import combinations
import warnings
from typing import Dict, List, Tuple, Optional, Union
import pickle
from datetime import datetime, timedelta
from dataclasses import dataclass
import ta  # Technical Analysis library

warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



@dataclass
class Config:
    # Data parameters
    min_observations: int = 60
    min_pairs_history: int = 24
    max_tickers: int = 100
    lookback_window: int = 252  # 1 year for calculations
    
    # ML Models
    pair_selection_hidden: List[int] = None
    threshold_lstm_hidden: int = 128
    threshold_lstm_layers: int = 2
    sequence_length: int = 20
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    dropout_rate: float = 0.3
    
    # Trading parameters
    top_pairs_percent: float = 0.1
    confidence_threshold: float = 0.6
    default_entry_zscore: float = 2.0
    default_exit_zscore: float = 0.5
    max_holding_period: int = 20
    position_size: float = 1.0
    
    # Risk management
    max_correlation: float = 0.95
    min_half_life: int = 5
    max_half_life: int = 60
    
    def __post_init__(self):
        if self.pair_selection_hidden is None:
            self.pair_selection_hidden = [256, 128, 64, 32]

CONFIG = Config()

class FeatureEngineer:
    """Feature engineering complet avec tous les indicateurs du paper"""
    
    def __init__(self):
        self.feature_names = []
        
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des features techniques complètes sans look-ahead bias"""
        data = data.copy()
        data = data.sort_values(['Ticker', 'date']).reset_index(drop=True)
        
        features = []
        for ticker in data['Ticker'].unique():
            ticker_data = data[data['Ticker'] == ticker].copy()
            
            if len(ticker_data) < CONFIG.min_observations:
                continue
            
            # Ensure we have necessary columns
            if 'Close' not in ticker_data.columns:
                ticker_data['Close'] = ticker_data['MthRet'].cumsum() + 100  # Synthetic close price
            if 'High' not in ticker_data.columns:
                ticker_data['High'] = ticker_data['Close'] * 1.02
            if 'Low' not in ticker_data.columns:
                ticker_data['Low'] = ticker_data['Close'] * 0.98
            if 'Volume' not in ticker_data.columns:
                ticker_data['Volume'] = np.random.lognormal(10, 2, len(ticker_data))
            
            # Basic price features
            ticker_data['returns'] = ticker_data['Close'].pct_change()
            ticker_data['log_returns'] = np.log(ticker_data['Close'] / ticker_data['Close'].shift(1))
            
            # Moving averages (various periods)
            for period in [5, 10, 20, 50, 200]:
                ticker_data[f'sma_{period}'] = ticker_data['Close'].rolling(period).mean()
                ticker_data[f'ema_{period}'] = ticker_data['Close'].ewm(span=period, adjust=False).mean()
            
            # Price ratios to moving averages
            for period in [20, 50]:
                ticker_data[f'price_to_sma_{period}'] = ticker_data['Close'] / ticker_data[f'sma_{period}']
            
            # Momentum indicators
            ticker_data['rsi'] = ta.momentum.RSIIndicator(ticker_data['Close'], window=14).rsi()
            ticker_data['stoch'] = ta.momentum.StochasticOscillator(
                ticker_data['High'], ticker_data['Low'], ticker_data['Close']
            ).stoch()
            
            # MACD
            macd = ta.trend.MACD(ticker_data['Close'])
            ticker_data['macd'] = macd.macd()
            ticker_data['macd_signal'] = macd.macd_signal()
            ticker_data['macd_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(ticker_data['Close'])
            ticker_data['bb_high'] = bb.bollinger_hband()
            ticker_data['bb_low'] = bb.bollinger_lband()
            ticker_data['bb_mid'] = bb.bollinger_mavg()
            ticker_data['bb_width'] = ticker_data['bb_high'] - ticker_data['bb_low']
            ticker_data['bb_position'] = (ticker_data['Close'] - ticker_data['bb_low']) / (ticker_data['bb_width'] + 1e-8)
            
            # Volatility features
            ticker_data['volatility_10'] = ticker_data['returns'].rolling(10).std() * np.sqrt(252)
            ticker_data['volatility_20'] = ticker_data['returns'].rolling(20).std() * np.sqrt(252)
            ticker_data['volatility_60'] = ticker_data['returns'].rolling(60).std() * np.sqrt(252)
            
            # ATR (Average True Range)
            ticker_data['atr'] = ta.volatility.AverageTrueRange(
                ticker_data['High'], ticker_data['Low'], ticker_data['Close']
            ).average_true_range()
            
            # Volume features
            ticker_data['volume_sma_10'] = ticker_data['Volume'].rolling(10).mean()
            ticker_data['volume_ratio'] = ticker_data['Volume'] / ticker_data['volume_sma_10']
            ticker_data['dollar_volume'] = ticker_data['Close'] * ticker_data['Volume']
            
            # Market microstructure
            ticker_data['high_low_spread'] = (ticker_data['High'] - ticker_data['Low']) / ticker_data['Close']
            ticker_data['close_to_high'] = (ticker_data['Close'] - ticker_data['Low']) / (ticker_data['High'] - ticker_data['Low'] + 1e-8)
            
            # Statistical features
            for window in [10, 20, 60]:
                ticker_data[f'skew_{window}'] = ticker_data['returns'].rolling(window).skew()
                ticker_data[f'kurt_{window}'] = ticker_data['returns'].rolling(window).kurt()
            
            # Autocorrelation
            for lag in [1, 5, 10]:
                ticker_data[f'autocorr_{lag}'] = ticker_data['returns'].rolling(20).apply(
                    lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
                )
            
            # Hurst exponent (simplified)
            ticker_data['hurst'] = ticker_data['returns'].rolling(60).apply(
                self._calculate_hurst_exponent
            )
            
            # Market beta (requires market data)
            market_returns = data.groupby('date')['MthRet'].mean()
            ticker_data = ticker_data.merge(
                market_returns.rename('market_return'), 
                left_on='date', 
                right_index=True, 
                how='left'
            )
            
            for window in [20, 60]:
                ticker_data[f'beta_{window}'] = ticker_data['returns'].rolling(window).cov(
                    ticker_data['market_return']
                ) / ticker_data['market_return'].rolling(window).var()
            
            # Regime indicators
            ticker_data['bull_market'] = (ticker_data['sma_50'] > ticker_data['sma_200']).astype(int)
            ticker_data['high_volatility_regime'] = (
                ticker_data['volatility_20'] > ticker_data['volatility_20'].rolling(252).mean()
            ).astype(int)
            
            # Clean and fill NaN values properly (no look-ahead)
            ticker_data = ticker_data.fillna(method='ffill').fillna(0)
            
            features.append(ticker_data)
        
        if features:
            result = pd.concat(features, ignore_index=True)
            
            # Store feature names for later use
            self.feature_names = [col for col in result.columns if col not in 
                                ['Ticker', 'date', 'Close', 'High', 'Low', 'Open', 'Volume']]
            
            return result
        else:
            return pd.DataFrame()
    
    def create_pair_features(self, data1: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:
        """Crée des features pour une paire d'actifs avec attention au look-ahead bias"""
        # Merge sur les dates communes
        merged = pd.merge(data1, data2, on='date', suffixes=('_1', '_2'), how='inner')
        
        if len(merged) < CONFIG.min_pairs_history:
            return pd.DataFrame()
        
        # Sort by date to ensure temporal order
        merged = merged.sort_values('date').reset_index(drop=True)
        
        # Basic spread features
        merged['price_spread'] = merged['Close_1'] - merged['Close_2']
        merged['return_spread'] = merged['returns_1'] - merged['returns_2']
        merged['log_price_ratio'] = np.log(merged['Close_1'] / merged['Close_2'])
        
        # Normalized spread (z-score of different windows)
        for window in [10, 20, 30, 60]:
            spread_mean = merged['return_spread'].rolling(window).mean()
            spread_std = merged['return_spread'].rolling(window).std()
            merged[f'spread_zscore_{window}'] = (merged['return_spread'] - spread_mean) / (spread_std + 1e-8)
        
        # Correlation features (rolling)
        for window in [20, 60, 120]:
            merged[f'correlation_{window}'] = merged['returns_1'].rolling(window).corr(merged['returns_2'])
        
        # Cointegration test p-value (rolling window)
        merged['coint_pvalue'] = merged.apply(
            lambda x: self._rolling_cointegration_test(merged, x.name, window=60), 
            axis=1
        )
        
        # Beta between assets
        for window in [20, 60]:
            cov = merged['returns_1'].rolling(window).cov(merged['returns_2'])
            var = merged['returns_2'].rolling(window).var()
            merged[f'pair_beta_{window}'] = cov / (var + 1e-8)
        
        # Volatility ratio
        merged['volatility_ratio'] = merged['volatility_20_1'] / (merged['volatility_20_2'] + 1e-8)
        
        # Mean reversion indicators
        merged['half_life'] = merged['return_spread'].rolling(60).apply(
            self._estimate_half_life
        )
        
        # Relative performance metrics
        merged['cumret_ratio'] = (1 + merged['returns_1']).cumprod() / (1 + merged['returns_2']).cumprod()
        merged['cumret_ratio_ma'] = merged['cumret_ratio'].rolling(20).mean()
        merged['cumret_ratio_std'] = merged['cumret_ratio'].rolling(20).std()
        
        # Volume features
        merged['volume_ratio'] = merged['Volume_1'] / (merged['Volume_2'] + 1e-8)
        merged['dollar_volume_ratio'] = merged['dollar_volume_1'] / (merged['dollar_volume_2'] + 1e-8)
        
        # Regime alignment
        merged['regime_agreement'] = (merged['bull_market_1'] == merged['bull_market_2']).astype(int)
        merged['volatility_regime_diff'] = merged['high_volatility_regime_1'] - merged['high_volatility_regime_2']
        
        # Technical indicator divergence
        merged['rsi_diff'] = merged['rsi_1'] - merged['rsi_2']
        merged['macd_diff_spread'] = merged['macd_diff_1'] - merged['macd_diff_2']
        
        # Market microstructure spread
        merged['bid_ask_spread_diff'] = merged['high_low_spread_1'] - merged['high_low_spread_2']
        
        # Cross-asset momentum
        merged['momentum_1_2'] = merged['returns_1'].rolling(20).mean() - merged['returns_2'].rolling(20).mean()
        
        # Clean data
        merged = merged.replace([np.inf, -np.inf], np.nan)
        merged = merged.fillna(method='ffill').fillna(0)
        
        return merged
    
    def _calculate_hurst_exponent(self, returns):
        """Calcule l'exposant de Hurst (version simplifiée)"""
        try:
            if len(returns) < 20:
                return 0.5
            
            # Range/Standard deviation method
            n = len(returns)
            mean_returns = np.mean(returns)
            
            # Calculate cumulative deviations
            deviations = returns - mean_returns
            cumulative_deviations = np.cumsum(deviations)
            
            # Calculate range
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            
            # Calculate standard deviation
            S = np.std(returns)
            
            if S == 0:
                return 0.5
            
            # Calculate Hurst exponent
            return np.log(R/S) / np.log(n/2)
        except:
            return 0.5
    
    def _rolling_cointegration_test(self, data, idx, window=60):
        """Test de cointégration roulant"""
        try:
            if idx < window:
                return 1.0
            
            start_idx = max(0, idx - window)
            
            series1 = data.iloc[start_idx:idx]['Close_1'].values
            series2 = data.iloc[start_idx:idx]['Close_2'].values
            
            if len(series1) < 20:
                return 1.0
            
            _, pvalue, _ = coint(series1, series2)
            return pvalue
        except:
            return 1.0
    
    def _estimate_half_life(self, spread_series):
        """Estime la demi-vie du spread"""
        try:
            if len(spread_series) < 20:
                return 30
            
            y = spread_series.values
            y_lag = np.roll(y, 1)[1:]
            y = y[1:]
            
            # OLS regression: y_t = alpha + beta * y_{t-1} + epsilon
            X = np.column_stack([np.ones(len(y_lag)), y_lag])
            
            # Solve using numpy (faster than statsmodels for simple OLS)
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            beta = coeffs[1]
            
            if beta >= 1 or beta <= 0:
                return 30
            
            # Half-life = -ln(2) / ln(beta)
            half_life = -np.log(2) / np.log(beta)
            
            # Constrain to reasonable values
            return np.clip(half_life, CONFIG.min_half_life, CONFIG.max_half_life)
        except:
            return 30

class PairSelectionModel(nn.Module):
    """Deep Neural Network pour la sélection des paires"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], dropout_rate: float = 0.3):
        super(PairSelectionModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.extend([
            nn.Linear(prev_size, 2)  # 2 outputs: profitability score, quality score
        ])
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        output = self.network(x)
        
        # Apply sigmoid to get probabilities
        profitability_score = torch.sigmoid(output[:, 0])
        quality_score = torch.sigmoid(output[:, 1])
        
        return profitability_score, quality_score
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class ThresholdOptimizationModel(nn.Module):
    """LSTM + Dense pour l'optimisation dynamique des seuils"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout_rate: float = 0.3):
        super(ThresholdOptimizationModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Dense layers for threshold prediction
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # [entry_threshold, exit_threshold]
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Generate thresholds
        raw_thresholds = self.dense(context_vector)
        
        # Apply constraints to ensure valid thresholds
        entry_threshold = torch.sigmoid(raw_thresholds[:, 0]) * 3 + 0.5  # Range: [0.5, 3.5]
        exit_threshold = torch.sigmoid(raw_thresholds[:, 1]) * 0.8 + 0.1  # Range: [0.1, 0.9]
        
        # Ensure entry > exit
        exit_threshold = torch.min(exit_threshold, entry_threshold * 0.5)
        
        return torch.stack([entry_threshold, exit_threshold], dim=1)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

class MLPairsTrader:
    """Système principal de trading ML avec gestion du look-ahead bias"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.pair_selection_model = None
        self.threshold_model = None
        self.scaler_pairs = None
        self.scaler_threshold = None
        self.selected_features_pairs = None
        self.selected_features_threshold = None
        
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prépare les données avec feature engineering complet"""
        logger.info("Preparing data with comprehensive feature engineering...")
        
        # Ensure data is sorted by date
        data = data.sort_values(['Ticker', 'date']).reset_index(drop=True)
        
        # Filter tickers with sufficient data
        ticker_counts = data.groupby('Ticker')['date'].count()
        valid_tickers = ticker_counts[ticker_counts >= CONFIG.min_observations].index.tolist()
        
        # Limit number of tickers for computational efficiency
        if len(valid_tickers) > CONFIG.max_tickers:
            # Select most liquid tickers based on average volume or market cap if available
            valid_tickers = valid_tickers[:CONFIG.max_tickers]
        
        data_filtered = data[data['Ticker'].isin(valid_tickers)].copy()
        
        # Feature engineering
        enhanced_data = self.feature_engineer.create_technical_features(data_filtered)
        
        logger.info(f"Data prepared: {len(enhanced_data)} records, {len(valid_tickers)} tickers")
        logger.info(f"Features created: {len(self.feature_engineer.feature_names)} features")
        
        return enhanced_data
    
    def generate_pair_dataset(self, data: pd.DataFrame, train_end_date: Optional[str] = None) -> pd.DataFrame:
        """Génère le dataset pour toutes les paires avec respect de la temporalité"""
        logger.info("Generating pair dataset...")
        
        # If train_end_date is specified, only use data up to that date for feature calculation
        if train_end_date:
            data = data[data['date'] <= train_end_date].copy()
        
        tickers = data['Ticker'].unique()
        all_pairs_data = []
        
        # Calculate number of pairs to process
        n_pairs = len(list(combinations(tickers, 2)))
        logger.info(f"Processing {n_pairs} potential pairs...")
        
        processed = 0
        for ticker1, ticker2 in combinations(tickers, 2):
            if processed % 100 == 0 and processed > 0:
                logger.info(f"Processed {processed}/{n_pairs} pairs")
            
            try:
                data1 = data[data['Ticker'] == ticker1].copy()
                data2 = data[data['Ticker'] == ticker2].copy()
                
                # Generate pair features
                pair_features = self.feature_engineer.create_pair_features(data1, data2)
                
                if len(pair_features) >= CONFIG.min_pairs_history:
                    pair_features['ticker1'] = ticker1
                    pair_features['ticker2'] = ticker2
                    pair_features['pair_id'] = f"{ticker1}_{ticker2}"
                    all_pairs_data.append(pair_features)
                    
            except Exception as e:
                logger.warning(f"Error processing pair {ticker1}-{ticker2}: {str(e)}")
                continue
            
            processed += 1
        
        if all_pairs_data:
            result = pd.concat(all_pairs_data, ignore_index=True)
            logger.info(f"Generated {len(result)} pair observations from {len(all_pairs_data)} pairs")
            return result
        else:
            logger.warning("No valid pairs found")
            return pd.DataFrame()
    
    def create_labels(self, pair_data: pd.DataFrame, forward_window: int = 20) -> pd.DataFrame:
        """Crée les labels sans look-ahead bias en utilisant des rendements futurs"""
        logger.info("Creating training labels with forward-looking returns...")
        
        labeled_data = []
        
        for pair_id in pair_data['pair_id'].unique():
            pair_subset = pair_data[pair_data['pair_id'] == pair_id].copy()
            pair_subset = pair_subset.sort_values('date').reset_index(drop=True)
            
            if len(pair_subset) < forward_window + CONFIG.sequence_length:
                continue
            
            # Calculate forward returns for labeling
            pair_subset['forward_spread_return'] = (
                pair_subset['return_spread'].rolling(forward_window).mean().shift(-forward_window)
            )
            
            # Simulate trading to determine profitability
            pair_subset['trades'] = 0
            pair_subset['pnl'] = 0
            
            for i in range(len(pair_subset) - forward_window):
                # Look at next 'forward_window' periods
                future_spreads = pair_subset.iloc[i:i+forward_window]['return_spread'].values
                future_zscores = pair_subset.iloc[i:i+forward_window]['spread_zscore_20'].values
                
                # Simulate a simple trading strategy
                pnl = self._simulate_future_trading(future_zscores, future_spreads)
                
                pair_subset.loc[i, 'pnl'] = pnl
                pair_subset.loc[i, 'trades'] = 1 if pnl != 0 else 0
            
            # Create labels based on profitability
            pair_subset['is_profitable'] = (pair_subset['pnl'] > 0).astype(int)
            
            # Quality score based on Sharpe ratio of forward returns
            rolling_sharpe = pair_subset['pnl'].rolling(forward_window).mean() / (
                pair_subset['pnl'].rolling(forward_window).std() + 1e-8
            ) * np.sqrt(252 / forward_window)
            pair_subset['quality_score'] = rolling_sharpe.clip(-3, 3) / 6 + 0.5  # Normalize to [0, 1]
            
            # Remove rows without labels (last forward_window rows)
            pair_subset = pair_subset[:-forward_window]
            
            if len(pair_subset) > 0:
                labeled_data.append(pair_subset)
        
        if labeled_data:
            result = pd.concat(labeled_data, ignore_index=True)
            
            # Log label statistics
            profitable_ratio = result['is_profitable'].mean()
            avg_quality = result['quality_score'].mean()
            logger.info(f"Labels created: {profitable_ratio:.2%} profitable pairs, avg quality: {avg_quality:.3f}")
            
            return result
        else:
            return pd.DataFrame()
    
    def _simulate_future_trading(self, zscores: np.ndarray, spreads: np.ndarray, 
                               entry_z: float = 2.0, exit_z: float = 0.5) -> float:
        """Simule le trading sur une période future pour créer les labels"""
        position = 0
        entry_spread = 0
        total_pnl = 0
        
        for i, (zscore, spread) in enumerate(zip(zscores, spreads)):
            if position == 0:  # No position
                if abs(zscore) > entry_z:
                    position = -np.sign(zscore)  # Go short if zscore > 0, long if zscore < 0
                    entry_spread = spread
            else:  # Have position
                if abs(zscore) < exit_z or i == len(zscores) - 1:  # Exit or end of period
                    pnl = position * (spread - entry_spread)
                    total_pnl += pnl
                    position = 0
        
        return total_pnl
    
    def train_pair_selection_model(self, labeled_data: pd.DataFrame) -> bool:
        """Entraîne le modèle de sélection des paires avec validation temporelle"""
        logger.info("Training pair selection model...")
        # NETTOYAGE : On garde que les lignes où le label est 0 ou 1
        labeled_data = labeled_data[(labeled_data['is_profitable'] == 0) | (labeled_data['is_profitable'] == 1)]

        # Ensuite, on enlève les lignes où il manque la cible ou le score de qualité
        labeled_data = labeled_data.dropna(subset=['is_profitable', 'quality_score'])



        
        # Select features for the model (exclude target and metadata columns)
        exclude_cols = ['date', 'pair_id', 'ticker1', 'ticker2', 'is_profitable', 
                       'quality_score', 'pnl', 'trades', 'forward_spread_return']
        feature_cols = [col for col in labeled_data.columns if col not in exclude_cols 
                       and not col.endswith(('_1', '_2'))]  # Avoid raw ticker features
        
        # Focus on pair-specific features
        pair_feature_keywords = ['spread', 'correlation', 'coint', 'pair_beta', 'volatility_ratio',
                               'half_life', 'cumret_ratio', 'volume_ratio', 'regime', 'momentum_1_2']
        
        selected_features = [col for col in feature_cols 
                           if any(keyword in col for keyword in pair_feature_keywords)]
        
        if len(selected_features) < 10:
            logger.error(f"Insufficient features for pair selection: {len(selected_features)}")
            return False
        
        self.selected_features_pairs = selected_features
        logger.info(f"Selected {len(selected_features)} features for pair selection")
        
        # Prepare data
        # Group by pair and date to get one row per pair per time period
        pair_data = labeled_data.groupby(['pair_id', 'date']).agg({
            **{col: 'mean' for col in selected_features},
            'is_profitable': 'first',
            'quality_score': 'first'
        }).reset_index()
        
        # Sort by date for time series split
        pair_data = pair_data.sort_values('date')
        
        X = pair_data[selected_features].values
        y_profit = pair_data['is_profitable'].values
        y_quality = pair_data['quality_score'].values
        
        # Handle NaN and inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Normalize features
        self.scaler_pairs = RobustScaler()
        X_scaled = self.scaler_pairs.fit_transform(X)
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pair_selection_model = PairSelectionModel(
            input_size=X_scaled.shape[1],
            hidden_sizes=CONFIG.pair_selection_hidden,
            dropout_rate=CONFIG.dropout_rate
        ).to(device)
        
        # Training setup
        criterion_profit = nn.BCELoss()
        criterion_quality = nn.MSELoss()
        optimizer = optim.Adam(self.pair_selection_model.parameters(), lr=CONFIG.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_score = -np.inf
        patience_counter = 0
        
        # Training loop
        for epoch in range(CONFIG.epochs):
            train_losses = []
            val_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                # Prepare data
                X_train = torch.FloatTensor(X_scaled[train_idx]).to(device)
                y_profit_train = torch.FloatTensor(y_profit[train_idx]).to(device)
                y_quality_train = torch.FloatTensor(y_quality[train_idx]).to(device)
                
                X_val = torch.FloatTensor(X_scaled[val_idx]).to(device)
                y_profit_val = torch.FloatTensor(y_profit[val_idx]).to(device)
                y_quality_val = torch.FloatTensor(y_quality[val_idx]).to(device)
                
                # Training
                self.pair_selection_model.train()
                optimizer.zero_grad()
                
                profit_pred, quality_pred = self.pair_selection_model(X_train)
                
                loss_profit = criterion_profit(profit_pred, y_profit_train)
                loss_quality = criterion_quality(quality_pred, y_quality_train)
                total_loss = loss_profit + 0.5 * loss_quality  # Weight quality less
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.pair_selection_model.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(total_loss.item())
                
                # Validation
                self.pair_selection_model.eval()
                with torch.no_grad():
                    profit_pred_val, quality_pred_val = self.pair_selection_model(X_val)
                    
                    # Calculate validation metrics
                    profit_acc = ((profit_pred_val > 0.5) == y_profit_val).float().mean().item()
                    quality_mae = torch.abs(quality_pred_val - y_quality_val).mean().item()
                    
                    # Combined score
                    val_score = profit_acc - quality_mae
                    val_scores.append(val_score)
            
            # Average metrics across folds
            avg_train_loss = np.mean(train_losses)
            avg_val_score = np.mean(val_scores)
            
            # Learning rate scheduling
            scheduler.step(-avg_val_score)
            
            # Early stopping
            if avg_val_score > best_val_score:
                best_val_score = avg_val_score
                patience_counter = 0
                # Save best model
                self.best_pair_model_state = self.pair_selection_model.state_dict()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Score: {avg_val_score:.4f}")
            
            if patience_counter >= 10:
                logger.info("Early stopping triggered")
                break
        
        # Load best model
        self.pair_selection_model.load_state_dict(self.best_pair_model_state)
        
        logger.info("Pair selection model training completed")
        return True
    
    def train_threshold_model(self, labeled_data: pd.DataFrame) -> bool:
        """Entraîne le modèle LSTM pour l'optimisation des seuils"""
        logger.info("Training threshold optimization model...")
        
        # Select features for sequences
        exclude_cols = ['date', 'pair_id', 'ticker1', 'ticker2', 'is_profitable', 
                       'quality_score', 'pnl', 'trades', 'forward_spread_return']
        
        # Focus on time-varying features
        time_feature_keywords = ['zscore', 'spread', 'correlation', 'volatility', 'volume', 
                               'rsi', 'macd', 'momentum', 'beta', 'regime']
        
        feature_cols = [col for col in labeled_data.columns if col not in exclude_cols 
                       and any(keyword in col for keyword in time_feature_keywords)]
        
        if len(feature_cols) < 10:
            logger.error(f"Insufficient features for threshold model: {len(feature_cols)}")
            return False
        
        self.selected_features_threshold = feature_cols
        logger.info(f"Selected {len(feature_cols)} features for threshold optimization")
        
        # Prepare sequences
        sequences = []
        targets = []
        
        for pair_id in labeled_data['pair_id'].unique():
            pair_subset = labeled_data[labeled_data['pair_id'] == pair_id].sort_values('date')
            
            if len(pair_subset) < CONFIG.sequence_length + 20:
                continue
            
            features = pair_subset[feature_cols].values
            
            for i in range(CONFIG.sequence_length, len(pair_subset) - 20):
                # Input sequence
                seq = features[i-CONFIG.sequence_length:i]
                
                # Calculate optimal thresholds for next 20 periods
                future_data = pair_subset.iloc[i:i+20]
                optimal_thresholds = self._calculate_optimal_thresholds(future_data)
                
                if optimal_thresholds is not None:
                    sequences.append(seq)
                    targets.append(optimal_thresholds)
        
        if len(sequences) < 100:
            logger.error(f"Insufficient sequences for training: {len(sequences)}")
            return False
        
        X = np.array(sequences)
        y = np.array(targets)
        
        # Handle NaN and inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize
        self.scaler_threshold = RobustScaler()
        X_flat = X.reshape(-1, X.shape[-1])
        X_flat_scaled = self.scaler_threshold.fit_transform(X_flat)
        X_scaled = X_flat_scaled.reshape(X.shape)
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=3)
        indices = np.arange(len(X_scaled))
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold_model = ThresholdOptimizationModel(
            input_size=X_scaled.shape[2],
            hidden_size=CONFIG.threshold_lstm_hidden,
            num_layers=CONFIG.threshold_lstm_layers,
            dropout_rate=CONFIG.dropout_rate
        ).to(device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.threshold_model.parameters(), lr=CONFIG.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = np.inf
        patience_counter = 0
        
        # Training loop
        for epoch in range(CONFIG.epochs):
            train_losses = []
            val_losses = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(indices)):
                # Prepare batches
                train_dataset = torch.utils.data.TensorDataset(
                    torch.FloatTensor(X_scaled[train_idx]),
                    torch.FloatTensor(y[train_idx])
                )
                val_dataset = torch.utils.data.TensorDataset(
                    torch.FloatTensor(X_scaled[val_idx]),
                    torch.FloatTensor(y[val_idx])
                )
                
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=CONFIG.batch_size, shuffle=True
                )
                val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=CONFIG.batch_size, shuffle=False
                )
                
                # Training
                self.threshold_model.train()
                fold_train_losses = []
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    optimizer.zero_grad()
                    predictions = self.threshold_model(batch_X)
                    loss = criterion(predictions, batch_y)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.threshold_model.parameters(), 1.0)
                    optimizer.step()
                    
                    fold_train_losses.append(loss.item())
                
                train_losses.extend(fold_train_losses)
                
                # Validation
                self.threshold_model.eval()
                fold_val_losses = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        predictions = self.threshold_model(batch_X)
                        loss = criterion(predictions, batch_y)
                        fold_val_losses.append(loss.item())
                
                val_losses.extend(fold_val_losses)
            
            # Average metrics
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                self.best_threshold_model_state = self.threshold_model.state_dict()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            if patience_counter >= 10:
                logger.info("Early stopping triggered")
                break
        
        # Load best model
        self.threshold_model.load_state_dict(self.best_threshold_model_state)
        
        logger.info("Threshold model training completed")
        return True
    
    def _calculate_optimal_thresholds(self, future_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Calcule les seuils optimaux par grid search sur données futures"""
        try:
            zscores = future_data['spread_zscore_20'].values
            spreads = future_data['return_spread'].values
            
            if len(spreads) < 5:
                return None
            
            best_sharpe = -np.inf
            best_thresholds = None
            
            # Grid search
            entry_range = np.arange(1.0, 3.5, 0.25)
            exit_range = np.arange(0.1, 1.0, 0.1)
            
            for entry_thresh in entry_range:
                for exit_thresh in exit_range:
                    if exit_thresh >= entry_thresh * 0.5:
                        continue
                    
                    # Simulate trading
                    returns = self._simulate_trading_returns(zscores, spreads, entry_thresh, exit_thresh)
                    
                    if len(returns) > 0:
                        # Calculate Sharpe ratio
                        avg_return = np.mean(returns)
                        std_return = np.std(returns)
                        
                        if std_return > 0:
                            sharpe = avg_return / std_return * np.sqrt(252)
                            
                            if sharpe > best_sharpe:
                                best_sharpe = sharpe
                                best_thresholds = [entry_thresh, exit_thresh]
            
            return np.array(best_thresholds) if best_thresholds else np.array([2.0, 0.5])
        except Exception as e:
            logger.warning(f"Error calculating optimal thresholds: {str(e)}")
            return np.array([2.0, 0.5])
    
    def _simulate_trading_returns(self, zscores: np.ndarray, spreads: np.ndarray, 
                                 entry_thresh: float, exit_thresh: float) -> List[float]:
        """Simule le trading et retourne la série des rendements"""
        position = 0
        entry_spread = 0
        returns = []
        holding_period = 0
        
        for zscore, spread in zip(zscores, spreads):
            if position == 0:  # No position
                if zscore > entry_thresh:
                    position = -1
                    entry_spread = spread
                    holding_period = 0
                elif zscore < -entry_thresh:
                    position = 1
                    entry_spread = spread
                    holding_period = 0
            else:  # Have position
                holding_period += 1
                
                # Exit conditions
                exit_signal = (abs(zscore) < exit_thresh or 
                             holding_period >= CONFIG.max_holding_period)
                
                if exit_signal:
                    trade_return = position * (spread - entry_spread)
                    returns.append(trade_return)
                    position = 0
                    holding_period = 0
        
        return returns
    
    def select_pairs_ml(self, pair_data: pd.DataFrame, top_n: Optional[int] = None) -> List[Tuple[str, str, float, float]]:
        """Sélectionne les meilleures paires avec le modèle ML"""
        logger.info("Selecting pairs with ML model...")
        
        if self.pair_selection_model is None:
            logger.error("Pair selection model not trained")
            return []
        
        # Use saved feature list
        if self.selected_features_pairs is None:
            logger.error("No features selected for pair selection")
            return []
        
        # Aggregate features by pair
        pair_features = pair_data.groupby('pair_id')[self.selected_features_pairs].mean().reset_index()
        
        X = pair_features[self.selected_features_pairs].values
        pair_ids = pair_features['pair_id'].values
        
        # Handle NaN and inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if len(X) == 0:
            return []
        
        # Normalize and predict
        X_scaled = self.scaler_pairs.transform(X)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.pair_selection_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            profit_scores, quality_scores = self.pair_selection_model(X_tensor)
            
            profit_scores = profit_scores.cpu().numpy()
            quality_scores = quality_scores.cpu().numpy()
        
        # Combined score (weighted average)
        combined_scores = 0.7 * profit_scores + 0.3 * quality_scores
        
        # Filter by confidence threshold
        confident_indices = np.where(
            (profit_scores > CONFIG.confidence_threshold) & 
            (quality_scores > 0.3)
        )[0]
        
        if len(confident_indices) == 0:
            # If no pairs meet threshold, take top percentage
            n_pairs = max(1, int(len(combined_scores) * CONFIG.top_pairs_percent))
            confident_indices = np.argsort(combined_scores)[-n_pairs:]
        
        # Determine number of pairs to select
        if top_n is None:
            top_n = min(len(confident_indices), 
                       max(5, int(len(pair_ids) * CONFIG.top_pairs_percent)))
        
        # Sort by combined score and select top pairs
        sorted_indices = confident_indices[np.argsort(combined_scores[confident_indices])[::-1]]
        selected_indices = sorted_indices[:top_n]
        
        # Extract selected pairs with scores
        selected_pairs = []
        for idx in selected_indices:
            pair_id = pair_ids[idx]
            ticker1, ticker2 = pair_id.split('_')
            selected_pairs.append((
                ticker1, 
                ticker2, 
                profit_scores[idx], 
                quality_scores[idx]
            ))
        
        logger.info(f"Selected {len(selected_pairs)} pairs with ML model")
        for i, (t1, t2, p_score, q_score) in enumerate(selected_pairs[:5]):
            logger.info(f"  Top {i+1}: {t1}-{t2}, Profit: {p_score:.3f}, Quality: {q_score:.3f}")
        
        return selected_pairs
    
    def get_dynamic_thresholds(self, pair_data: pd.DataFrame, pair_id: str) -> np.ndarray:
        """Obtient les seuils dynamiques pour une paire spécifique"""
        if self.threshold_model is None or self.selected_features_threshold is None:
            # Return default thresholds
            return np.array([[CONFIG.default_entry_zscore, CONFIG.default_exit_zscore]] * len(pair_data))
        
        try:
            # Filter data for specific pair
            pair_subset = pair_data[pair_data['pair_id'] == pair_id].sort_values('date')
            
            if len(pair_subset) < CONFIG.sequence_length:
                return np.array([[CONFIG.default_entry_zscore, CONFIG.default_exit_zscore]] * len(pair_subset))
            
            # Prepare sequences
            features = pair_subset[self.selected_features_threshold].values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            thresholds = []
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.threshold_model.eval()
            with torch.no_grad():
                for i in range(len(features)):
                    if i < CONFIG.sequence_length:
                        # Not enough history, use defaults
                        thresholds.append([CONFIG.default_entry_zscore, CONFIG.default_exit_zscore])
                    else:
                        # Get sequence
                        seq = features[i-CONFIG.sequence_length:i]
                        seq_scaled = self.scaler_threshold.transform(seq)
                        
                        # Predict
                        seq_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)
                        pred_thresholds = self.threshold_model(seq_tensor).cpu().numpy()[0]
                        
                        thresholds.append(pred_thresholds)
            
            return np.array(thresholds)
            
        except Exception as e:
            logger.warning(f"Error getting dynamic thresholds: {str(e)}")
            return np.array([[CONFIG.default_entry_zscore, CONFIG.default_exit_zscore]] * len(pair_data))
    
    def execute_ml_strategy(self, data: pd.DataFrame, pair_data: pd.DataFrame,
                           selected_pairs: List[Tuple[str, str, float, float]], 
                           start_date: Optional[str] = None) -> Dict:
        """Exécute la stratégie ML avec seuils dynamiques"""
        logger.info("Executing ML strategy with dynamic thresholds...")
        
        all_trades = []
        
        for ticker1, ticker2, profit_score, quality_score in selected_pairs:
            try:
                pair_id = f"{ticker1}_{ticker2}"
                
                # Get pair data
                pair_subset = pair_data[pair_data['pair_id'] == pair_id].sort_values('date')
                
                if start_date:
                    pair_subset = pair_subset[pair_subset['date'] >= start_date]
                
                if len(pair_subset) < 20:
                    continue
                
                # Get dynamic thresholds
                thresholds = self.get_dynamic_thresholds(pair_subset, pair_id)
                
                # Trading simulation
                position = 0
                entry_price = None
                entry_date = None
                entry_idx = None
                holding_period = 0
                
                for i, row in pair_subset.iterrows():
                    idx = pair_subset.index.get_loc(i)
                    
                    zscore = row['spread_zscore_20']
                    spread = row['return_spread']
                    date = row['date']
                    entry_thresh, exit_thresh = thresholds[idx]
                    
                    if position == 0:  # No position
                        if zscore > entry_thresh:
                            position = -1
                            entry_price = spread
                            entry_date = date
                            entry_idx = idx
                            holding_period = 0
                        elif zscore < -entry_thresh:
                            position = 1
                            entry_price = spread
                            entry_date = date
                            entry_idx = idx
                            holding_period = 0
                    
                    else:  # Have position
                        holding_period += 1
                        
                        # Exit conditions
                        exit_signal = (
                            (abs(zscore) < exit_thresh) or
                            (holding_period >= CONFIG.max_holding_period) or
                            (position * zscore < -entry_thresh * 1.5)  # Stop loss
                        )
                        
                        if exit_signal:
                            pnl = position * (spread - entry_price) * CONFIG.position_size
                            
                            all_trades.append({
                                'pair': pair_id,
                                'ticker1': ticker1,
                                'ticker2': ticker2,
                                'entry_date': entry_date,
                                'exit_date': date,
                                'holding_period': holding_period,
                                'position': position,
                                'entry_spread': entry_price,
                                'exit_spread': spread,
                                'pnl': pnl,
                                'return': pnl / CONFIG.position_size,
                                'entry_threshold': entry_thresh,
                                'exit_threshold': exit_thresh,
                                'entry_zscore': pair_subset.iloc[entry_idx]['spread_zscore_20'],
                                'exit_zscore': zscore,
                                'profit_score': profit_score,
                                'quality_score': quality_score
                            })
                            
                            position = 0
                            entry_price = None
                            entry_date = None
                            holding_period = 0
                            
            except Exception as e:
                logger.warning(f"Error executing trades for pair {ticker1}-{ticker2}: {str(e)}")
                continue
        
        return self._calculate_performance_metrics(all_trades)
    
    def _calculate_performance_metrics(self, trades: List[Dict]) -> Dict:
        """Calcule les métriques de performance détaillées"""
        if not trades:
            return {
                'total_trades': 0,
                'total_return': 0,
                'avg_return': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'max_drawdown': 0,
                'max_drawdown_duration': 0,
                'avg_holding_period': 0,
                'trades_df': pd.DataFrame()
            }
        
        trades_df = pd.DataFrame(trades)
        trades_df = trades_df.sort_values('exit_date').reset_index(drop=True)
        
        # Basic metrics
        total_trades = len(trades_df)
        pnl_series = trades_df['pnl']
        returns_series = trades_df['return']
        
        total_return = pnl_series.sum()
        avg_return = returns_series.mean()
        
        # Win rate and profit factor
        winning_trades = pnl_series[pnl_series > 0]
        losing_trades = pnl_series[pnl_series < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        profit_factor = (winning_trades.sum() / abs(losing_trades.sum()) 
                        if len(losing_trades) > 0 and losing_trades.sum() != 0 else np.inf)
        
        # Risk-adjusted returns
        returns_std = returns_series.std()
        sharpe_ratio = avg_return / returns_std * np.sqrt(252) if returns_std > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns_series[returns_series < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = avg_return / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + returns_series).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        max_drawdown = drawdown.min()
        
        # Maximum drawdown duration
        drawdown_start = None
        max_dd_duration = 0
        current_dd_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and drawdown_start is None:
                drawdown_start = i
            elif dd >= 0 and drawdown_start is not None:
                current_dd_duration = i - drawdown_start
                max_dd_duration = max(max_dd_duration, current_dd_duration)
                drawdown_start = None
        
        # Calmar ratio
        calmar_ratio = avg_return * 252 / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Average holding period
        avg_holding = trades_df['holding_period'].mean()
        
        # Add cumulative returns to trades_df
        trades_df['cumulative_return'] = (1 + trades_df['return']).cumprod() - 1
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        
        return {
            'total_trades': total_trades,
            'total_return': total_return,
            'avg_return': avg_return,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_dd_duration,
            'avg_holding_period': avg_holding,
            'avg_win': winning_trades.mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades.mean() if len(losing_trades) > 0 else 0,
            'best_trade': pnl_series.max(),
            'worst_trade': pnl_series.min(),
            'trades_df': trades_df
        }


class BenchmarkStrategy:
    """Stratégie benchmark traditionnelle pour comparaison"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
    
    def select_pairs_traditional(self, data: pd.DataFrame, pair_data: pd.DataFrame) -> List[Tuple[str, str]]:
        """Sélection traditionnelle basée sur corrélation et cointégration"""
        logger.info("Selecting pairs using traditional correlation/cointegration method...")
        
        # Use the most recent data for pair selection
        recent_date = pair_data['date'].max()
        lookback_date = recent_date - pd.Timedelta(days=252)  # 1 year lookback
        
        recent_data = pair_data[pair_data['date'] > lookback_date]
        
        pair_metrics = []
        
        for pair_id in recent_data['pair_id'].unique():
            try:
                pair_subset = recent_data[recent_data['pair_id'] == pair_id]
                
                if len(pair_subset) < 60:  # Need at least 60 observations
                    continue
                
                # Calculate metrics
                avg_correlation = pair_subset['correlation_60'].mean()
                min_correlation = pair_subset['correlation_60'].min()
                avg_coint_pvalue = pair_subset['coint_pvalue'].mean()
                half_life = pair_subset['half_life'].mean()
                
                # Filter criteria
                if (avg_correlation > 0.6 and 
                    min_correlation > 0.4 and
                    avg_coint_pvalue < 0.05 and
                    CONFIG.min_half_life < half_life < CONFIG.max_half_life):
                    
                    ticker1, ticker2 = pair_id.split('_')
                    
                    pair_metrics.append({
                        'ticker1': ticker1,
                        'ticker2': ticker2,
                        'correlation': avg_correlation,
                        'coint_pvalue': avg_coint_pvalue,
                        'half_life': half_life,
                        'score': avg_correlation * (1 - avg_coint_pvalue)
                    })
                    
            except Exception as e:
                continue
        
        # Sort by score and select top pairs
        pair_metrics.sort(key=lambda x: x['score'], reverse=True)
        n_pairs = max(5, int(len(pair_metrics) * CONFIG.top_pairs_percent))
        
        selected_pairs = [(p['ticker1'], p['ticker2']) for p in pair_metrics[:n_pairs]]
        
        logger.info(f"Traditional method selected {len(selected_pairs)} pairs")
        return selected_pairs
    
    def execute_benchmark_strategy(self, data: pd.DataFrame, pair_data: pd.DataFrame,
                                 selected_pairs: List[Tuple[str, str]], 
                                 start_date: Optional[str] = None) -> Dict:
        """Exécute la stratégie benchmark avec seuils fixes"""
        logger.info("Executing benchmark strategy with fixed thresholds...")
        
        all_trades = []
        
        for ticker1, ticker2 in selected_pairs:
            try:
                pair_id = f"{ticker1}_{ticker2}"
                
                # Get pair data
                pair_subset = pair_data[pair_data['pair_id'] == pair_id].sort_values('date')
                
                if start_date:
                    pair_subset = pair_subset[pair_subset['date'] >= start_date]
                
                if len(pair_subset) < 20:
                    continue
                
                # Trading simulation with fixed thresholds
                position = 0
                entry_price = None
                entry_date = None
                holding_period = 0
                
                for _, row in pair_subset.iterrows():
                    zscore = row['spread_zscore_20']
                    spread = row['return_spread']
                    date = row['date']
                    
                    if position == 0:  # No position
                        if zscore > CONFIG.default_entry_zscore:
                            position = -1
                            entry_price = spread
                            entry_date = date
                            holding_period = 0
                        elif zscore < -CONFIG.default_entry_zscore:
                            position = 1
                            entry_price = spread
                            entry_date = date
                            holding_period = 0
                    
                    else:  # Have position
                        holding_period += 1
                        
                        # Exit conditions
                        exit_signal = (
                            (abs(zscore) < CONFIG.default_exit_zscore) or
                            (holding_period >= CONFIG.max_holding_period)
                        )
                        
                        if exit_signal:
                            pnl = position * (spread - entry_price) * CONFIG.position_size
                            
                            all_trades.append({
                                'pair': pair_id,
                                'ticker1': ticker1,
                                'ticker2': ticker2,
                                'entry_date': entry_date,
                                'exit_date': date,
                                'holding_period': holding_period,
                                'position': position,
                                'entry_spread': entry_price,
                                'exit_spread': spread,
                                'pnl': pnl,
                                'return': pnl / CONFIG.position_size,
                                'entry_threshold': CONFIG.default_entry_zscore,
                                'exit_threshold': CONFIG.default_exit_zscore,
                                'entry_zscore': zscore if position == -1 else -zscore,
                                'exit_zscore': zscore
                            })
                            
                            position = 0
                            entry_price = None
                            entry_date = None
                            holding_period = 0
                            
            except Exception as e:
                logger.warning(f"Error executing benchmark trades for pair {ticker1}-{ticker2}: {str(e)}")
                continue
        
        # Use the same performance calculation method
        ml_trader = MLPairsTrader()
        return ml_trader._calculate_performance_metrics(all_trades)


class Visualizer:
    """Classe pour les visualisations avancées"""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_comprehensive_comparison(self, ml_results: Dict, benchmark_results: Dict, 
                                    save_prefix: str = 'comparison'):
        """Crée une comparaison complète entre ML et Benchmark"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Performance metrics comparison
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_performance_bars(ax1, ml_results, benchmark_results)
        
        # 2. Cumulative returns
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_cumulative_returns(ax2, ml_results, benchmark_results)
        
        # 3. Return distribution
        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_return_distribution(ax3, ml_results, benchmark_results)
        
        # 4. Risk-return scatter
        ax4 = fig.add_subplot(gs[2, 1])
        self._plot_risk_return(ax4, ml_results, benchmark_results)
        
        # 5. Trade statistics
        ax5 = fig.add_subplot(gs[2, 2])
        self._plot_trade_statistics(ax5, ml_results, benchmark_results)
        
        # 6. Drawdown comparison
        ax6 = fig.add_subplot(gs[3, :])
        self._plot_drawdown(ax6, ml_results, benchmark_results)
        
        plt.suptitle('ML vs Benchmark Strategy Comparison', fontsize=16, y=0.995)
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, f'{save_prefix}_comprehensive.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Additional detailed plots
        self.plot_detailed_analysis(ml_results, benchmark_results, save_prefix)
    
    def _plot_performance_bars(self, ax, ml_results, benchmark_results):
        """Plot performance metrics comparison"""
        metrics = {
            'Total Return': ('total_return', '{:.2%}'),
            'Sharpe Ratio': ('sharpe_ratio', '{:.2f}'),
            'Sortino Ratio': ('sortino_ratio', '{:.2f}'),
            'Win Rate': ('win_rate', '{:.1%}'),
            'Profit Factor': ('profit_factor', '{:.2f}'),
            'Calmar Ratio': ('calmar_ratio', '{:.2f}')
        }
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ml_values = []
        bench_values = []
        labels = []
        
        for label, (key, fmt) in metrics.items():
            ml_val = ml_results.get(key, 0)
            bench_val = benchmark_results.get(key, 0)
            
            ml_values.append(ml_val)
            bench_values.append(bench_val)
            labels.append(label)
        
        bars1 = ax.bar(x - width/2, ml_values, width, label='ML Strategy', alpha=0.8)
        bars2 = ax.bar(x + width/2, bench_values, width, label='Benchmark', alpha=0.8)
        
        # Add value labels on bars
        for i, (ml_val, bench_val, (label, (key, fmt))) in enumerate(zip(ml_values, bench_values, metrics.items())):
            ax.text(i - width/2, ml_val + 0.01, fmt.format(ml_val), 
                   ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, bench_val + 0.01, fmt.format(bench_val), 
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_cumulative_returns(self, ax, ml_results, benchmark_results):
        """Plot cumulative returns over time"""
        if not ml_results['trades_df'].empty:
            ml_trades = ml_results['trades_df'].sort_values('exit_date')
            ml_cumret = (1 + ml_trades['return']).cumprod() - 1
            ax.plot(ml_cumret.values, label='ML Strategy', linewidth=2)
        
        if not benchmark_results['trades_df'].empty:
            bench_trades = benchmark_results['trades_df'].sort_values('exit_date')
            bench_cumret = (1 + bench_trades['return']).cumprod() - 1
            ax.plot(bench_cumret.values, label='Benchmark', linewidth=2)
        
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative Return')
        ax.set_title('Cumulative Returns Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    def _plot_return_distribution(self, ax, ml_results, benchmark_results):
        """Plot return distribution"""
        if not ml_results['trades_df'].empty:
            ax.hist(ml_results['trades_df']['return'], bins=30, alpha=0.6, 
                   label='ML Strategy', density=True, edgecolor='black')
        
        if not benchmark_results['trades_df'].empty:
            ax.hist(benchmark_results['trades_df']['return'], bins=30, alpha=0.6, 
                   label='Benchmark', density=True, edgecolor='black')
        
        ax.set_xlabel('Trade Return')
        ax.set_ylabel('Density')
        ax.set_title('Return Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    def _plot_risk_return(self, ax, ml_results, benchmark_results):
        """Plot risk-return scatter"""
        strategies = []
        returns = []
        risks = []
        sharpes = []
        
        if ml_results['total_trades'] > 0:
            strategies.append('ML Strategy')
            returns.append(ml_results['avg_return'] * 252)  # Annualized
            risks.append(abs(ml_results['max_drawdown']))
            sharpes.append(ml_results['sharpe_ratio'])
        
        if benchmark_results['total_trades'] > 0:
            strategies.append('Benchmark')
            returns.append(benchmark_results['avg_return'] * 252)  # Annualized
            risks.append(abs(benchmark_results['max_drawdown']))
            sharpes.append(benchmark_results['sharpe_ratio'])
        
        scatter = ax.scatter(risks, returns, s=200, alpha=0.7, c=sharpes, 
                           cmap='viridis', edgecolors='black', linewidth=2)
        
        for i, strategy in enumerate(strategies):
            ax.annotate(strategy, (risks[i], returns[i]), 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Maximum Drawdown (Risk)')
        ax.set_ylabel('Annualized Return')
        ax.set_title('Risk-Return Profile')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for Sharpe ratio
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio')
    
    def _plot_trade_statistics(self, ax, ml_results, benchmark_results):
        """Plot trade statistics"""
        categories = ['Total\nTrades', 'Avg Holding\nPeriod', 'Win\nRate']
        
        ml_stats = [
            ml_results.get('total_trades', 0),
            ml_results.get('avg_holding_period', 0),
            ml_results.get('win_rate', 0) * 100
        ]
        
        bench_stats = [
            benchmark_results.get('total_trades', 0),
            benchmark_results.get('avg_holding_period', 0),
            benchmark_results.get('win_rate', 0) * 100
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, ml_stats, width, label='ML Strategy', alpha=0.8)
        ax.bar(x + width/2, bench_stats, width, label='Benchmark', alpha=0.8)
        
        ax.set_ylabel('Values')
        ax.set_title('Trade Statistics')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_drawdown(self, ax, ml_results, benchmark_results):
        """Plot drawdown over time"""
        if not ml_results['trades_df'].empty:
            ml_trades = ml_results['trades_df'].sort_values('exit_date')
            ml_cumret = (1 + ml_trades['return']).cumprod()
            ml_running_max = ml_cumret.expanding().max()
            ml_drawdown = (ml_cumret - ml_running_max) / ml_running_max
            ax.fill_between(range(len(ml_drawdown)), 0, ml_drawdown.values, 
                          alpha=0.3, label='ML Strategy')
        
        if not benchmark_results['trades_df'].empty:
            bench_trades = benchmark_results['trades_df'].sort_values('exit_date')
            bench_cumret = (1 + bench_trades['return']).cumprod()
            bench_running_max = bench_cumret.expanding().max()
            bench_drawdown = (bench_cumret - bench_running_max) / bench_running_max
            ax.fill_between(range(len(bench_drawdown)), 0, bench_drawdown.values, 
                          alpha=0.3, label='Benchmark')
        
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Drawdown')
        ax.set_title('Drawdown Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    def plot_detailed_analysis(self, ml_results: Dict, benchmark_results: Dict, 
                             save_prefix: str = 'detailed'):
        """Create additional detailed analysis plots"""
        
        # 1. Monthly returns heatmap
        if not ml_results['trades_df'].empty:
            self._plot_monthly_returns(ml_results['trades_df'], 
                                     f'{save_prefix}_ml_monthly_returns.png')
        
        if not benchmark_results['trades_df'].empty:
            self._plot_monthly_returns(benchmark_results['trades_df'], 
                                     f'{save_prefix}_benchmark_monthly_returns.png')
        
        # 2. Trade analysis
        self._plot_trade_analysis(ml_results, benchmark_results, save_prefix)
        
        # 3. Pair performance
        if not ml_results['trades_df'].empty:
            self._plot_pair_performance(ml_results['trades_df'], 
                                      f'{save_prefix}_ml_pair_performance.png')
    
    def _plot_monthly_returns(self, trades_df: pd.DataFrame, filename: str):
        """Plot monthly returns heatmap"""
        if trades_df.empty:
            return
        
        # Convert to monthly returns
        trades_df['exit_month'] = pd.to_datetime(trades_df['exit_date']).dt.to_period('M')
        monthly_returns = trades_df.groupby('exit_month')['return'].sum()
        
        # Create matrix for heatmap
        years = sorted(monthly_returns.index.year.unique())
        months = range(1, 13)
        
        returns_matrix = np.zeros((len(years), 12))
        
        for i, year in enumerate(years):
            for j, month in enumerate(months):
                try:
                    period = pd.Period(year=year, month=month, freq='M')
                    if period in monthly_returns.index:
                        returns_matrix[i, j] = monthly_returns[period]
                except:
                    returns_matrix[i, j] = 0
        
        # Plot heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(returns_matrix, 
                   xticklabels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                   yticklabels=years,
                   cmap='RdYlGn',
                   center=0,
                   annot=True,
                   fmt='.1%',
                   cbar_kws={'label': 'Monthly Return'})
        
        plt.title('Monthly Returns Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
    
    def _plot_trade_analysis(self, ml_results: Dict, benchmark_results: Dict, 
                           save_prefix: str):
        """Detailed trade analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Trade PnL distribution
        ax = axes[0, 0]
        if not ml_results['trades_df'].empty:
            ml_pnls = ml_results['trades_df']['pnl']
            ax.hist(ml_pnls, bins=30, alpha=0.6, label=f'ML (μ={ml_pnls.mean():.3f})', 
                   density=True)
        
        if not benchmark_results['trades_df'].empty:
            bench_pnls = benchmark_results['trades_df']['pnl']
            ax.hist(bench_pnls, bins=30, alpha=0.6, 
                   label=f'Benchmark (μ={bench_pnls.mean():.3f})', density=True)
        
        ax.set_xlabel('Trade PnL')
        ax.set_ylabel('Density')
        ax.set_title('Trade PnL Distribution')
        ax.legend()
        ax.axvline(x=0, color='black', linestyle='--')
        
        # 2. Holding period distribution
        ax = axes[0, 1]
        if not ml_results['trades_df'].empty:
            ax.hist(ml_results['trades_df']['holding_period'], bins=20, alpha=0.6, 
                   label='ML', density=True)
        
        if not benchmark_results['trades_df'].empty:
            ax.hist(benchmark_results['trades_df']['holding_period'], bins=20, 
                   alpha=0.6, label='Benchmark', density=True)
        
        ax.set_xlabel('Holding Period (days)')
        ax.set_ylabel('Density')
        ax.set_title('Holding Period Distribution')
        ax.legend()
        
        # 3. Entry/Exit Z-score analysis (ML only)
        ax = axes[1, 0]
        if not ml_results['trades_df'].empty and 'entry_zscore' in ml_results['trades_df'].columns:
            trades = ml_results['trades_df']
            winning_trades = trades[trades['pnl'] > 0]
            losing_trades = trades[trades['pnl'] <= 0]
            
            ax.scatter(winning_trades['entry_zscore'], winning_trades['exit_zscore'], 
                      alpha=0.5, c='green', label='Winning', s=30)
            ax.scatter(losing_trades['entry_zscore'], losing_trades['exit_zscore'], 
                      alpha=0.5, c='red', label='Losing', s=30)
            
            ax.set_xlabel('Entry Z-score')
            ax.set_ylabel('Exit Z-score')
            ax.set_title('Entry vs Exit Z-scores (ML Strategy)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Cumulative PnL by pair
        ax = axes[1, 1]
        if not ml_results['trades_df'].empty:
            top_pairs = ml_results['trades_df'].groupby('pair')['pnl'].sum().nlargest(5)
            
            for pair in top_pairs.index:
                pair_trades = ml_results['trades_df'][ml_results['trades_df']['pair'] == pair]
                pair_trades = pair_trades.sort_values('exit_date')
                cum_pnl = pair_trades['pnl'].cumsum()
                ax.plot(cum_pnl.values, label=pair[:10], linewidth=2)
            
            ax.set_xlabel('Trade Number')
            ax.set_ylabel('Cumulative PnL')
            ax.set_title('Top 5 Pairs Cumulative PnL (ML)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{save_prefix}_trade_analysis.png'), 
                   dpi=300)
        plt.close()
    
    def _plot_pair_performance(self, trades_df: pd.DataFrame, filename: str):
        """Plot performance by pair"""
        if trades_df.empty:
            return
        
        # Calculate metrics by pair
        pair_metrics = trades_df.groupby('pair').agg({
            'pnl': ['sum', 'mean', 'count'],
            'return': lambda x: (1 + x).prod() - 1  # Compound return
        })
        
        pair_metrics.columns = ['total_pnl', 'avg_pnl', 'n_trades', 'compound_return']
        pair_metrics = pair_metrics.sort_values('total_pnl', ascending=False).head(20)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Total PnL by pair
        ax = axes[0, 0]
        pair_metrics['total_pnl'].plot(kind='barh', ax=ax)
        ax.set_xlabel('Total PnL')
        ax.set_title('Total PnL by Pair')
        ax.grid(True, alpha=0.3)
        
        # 2. Average PnL by pair
        ax = axes[0, 1]
        pair_metrics['avg_pnl'].plot(kind='barh', ax=ax)
        ax.set_xlabel('Average PnL per Trade')
        ax.set_title('Average PnL by Pair')
        ax.grid(True, alpha=0.3)
        
        # 3. Number of trades by pair
        ax = axes[1, 0]
        pair_metrics['n_trades'].plot(kind='barh', ax=ax)
        ax.set_xlabel('Number of Trades')
        ax.set_title('Trade Count by Pair')
        ax.grid(True, alpha=0.3)
        
        # 4. Compound return by pair
        ax = axes[1, 1]
        pair_metrics['compound_return'].plot(kind='barh', ax=ax)
        ax.set_xlabel('Compound Return')
        ax.set_title('Compound Return by Pair')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
    
    def save_results_summary(self, ml_results: Dict, benchmark_results: Dict, 
                           filename: str = 'results_summary.txt'):
        """Save detailed results summary to text file"""
        with open(os.path.join(self.output_dir, filename), 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PAIRS TRADING STRATEGY COMPARISON SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # ML Strategy Results
            f.write("ML ENHANCED STRATEGY RESULTS:\n")
            f.write("-" * 40 + "\n")
            self._write_strategy_summary(f, ml_results)
            
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Benchmark Results
            f.write("BENCHMARK STRATEGY RESULTS:\n")
            f.write("-" * 40 + "\n")
            self._write_strategy_summary(f, benchmark_results)
            
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Comparison
            f.write("PERFORMANCE COMPARISON:\n")
            f.write("-" * 40 + "\n")
            self._write_comparison(f, ml_results, benchmark_results)
    
    def _write_strategy_summary(self, f, results: Dict):
        """Write strategy summary to file"""
        f.write(f"Total Trades: {results.get('total_trades', 0)}\n")
        f.write(f"Total Return: {results.get('total_return', 0):.2%}\n")
        f.write(f"Average Return per Trade: {results.get('avg_return', 0):.3%}\n")
        f.write(f"Win Rate: {results.get('win_rate', 0):.1%}\n")
        f.write(f"Profit Factor: {results.get('profit_factor', 0):.2f}\n")
        f.write(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}\n")
        f.write(f"Sortino Ratio: {results.get('sortino_ratio', 0):.2f}\n")
        f.write(f"Calmar Ratio: {results.get('calmar_ratio', 0):.2f}\n")
        f.write(f"Maximum Drawdown: {results.get('max_drawdown', 0):.2%}\n")
        f.write(f"Max Drawdown Duration: {results.get('max_drawdown_duration', 0)} periods\n")
        f.write(f"Average Holding Period: {results.get('avg_holding_period', 0):.1f} days\n")
        f.write(f"Best Trade: {results.get('best_trade', 0):.3f}\n")
        f.write(f"Worst Trade: {results.get('worst_trade', 0):.3f}\n")
        f.write(f"Average Win: {results.get('avg_win', 0):.3f}\n")
        f.write(f"Average Loss: {results.get('avg_loss', 0):.3f}\n")
    
    def _write_comparison(self, f, ml_results: Dict, benchmark_results: Dict):
        """Write comparison metrics"""
        metrics = [
            ('Total Return', 'total_return', '{:.2%}'),
            ('Sharpe Ratio', 'sharpe_ratio', '{:.2f}'),
            ('Win Rate', 'win_rate', '{:.1%}'),
            ('Max Drawdown', 'max_drawdown', '{:.2%}'),
            ('Profit Factor', 'profit_factor', '{:.2f}')
        ]
        
        for name, key, fmt in metrics:
            ml_val = ml_results.get(key, 0)
            bench_val = benchmark_results.get(key, 0)
            
            if bench_val != 0:
                improvement = ((ml_val - bench_val) / abs(bench_val)) * 100
            else:
                improvement = 0
            
            f.write(f"\n{name}:\n")
            f.write(f"  ML Strategy: {fmt.format(ml_val)}\n")
            f.write(f"  Benchmark: {fmt.format(bench_val)}\n")
            f.write(f"  Improvement: {improvement:+.1f}%\n")


def run_with_real_data():
    """
    Main routine: runs pairs trading with real data, full pipeline.
    Pas d'arguments à passer, juste lance main() !
    """
    # 1. CONFIG : chemins d'accès
    folder = os.path.expanduser('~/Desktop/Machine Learning/data_ML_Project')
    folder1 = os.path.expanduser('~/Desktop/Machine Learning/data_filtered')
    output_dir = os.path.expanduser('~/Desktop/Machine Learning/deep_learning_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. LOAD DATA
    try:
        logger.info("Loading data files...")
        main_csv = os.path.join(folder, 'monthly_crsp.csv')
        jkp_csv = os.path.join(folder1, 'JKP_filtered.csv')
        zimmer_csv = os.path.join(folder1, 'Zimmer_filtered.csv')
        norm_csv = os.path.join(folder1, 'CompFirmCharac_filtred.csv')
        data_main = pd.read_csv(main_csv)
        data_jkp = pd.read_csv(jkp_csv)
        data_zimmer = pd.read_csv(zimmer_csv)
        data_comp = pd.read_csv(norm_csv)
    except FileNotFoundError as e:
        logger.error(f"Missing file: {e}")
        return

    # 3. PREPROCESSING
    try:
        # Dates
        data_main['date'] = pd.to_datetime(data_main['MthCalDt'], errors='coerce')
        data_jkp['date'] = pd.to_datetime(data_jkp['date'], errors='coerce')
        data_zimmer['date'] = pd.to_datetime(data_zimmer['date'], errors='coerce')
        data_comp['datadate'] = pd.to_datetime(data_comp['datadate'], errors='coerce')
        # Virer les dates nulles
        data_main.dropna(subset=['date'], inplace=True)
        data_jkp.dropna(subset=['date'], inplace=True)
        data_zimmer.dropna(subset=['date'], inplace=True)
        data_comp.dropna(subset=['datadate'], inplace=True)
        # Uniformiser les noms de colonnes
        data_comp['tic'] = data_comp['tic'].astype(str)
        data_comp.rename(columns={'tic': 'Ticker'}, inplace=True)
        # Filtrer les dates récentes
        start_date = pd.to_datetime('2010-01-01')
        data_main = data_main[data_main['date'] >= start_date]
        data_jkp = data_jkp[data_jkp['date'] >= start_date]
        data_zimmer = data_zimmer[data_zimmer['date'] >= start_date]
        # Année/mois pour merge
        for df in [data_main, data_jkp, data_zimmer]:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
        # Merge sur year/mois
        merged = data_main.merge(
            data_jkp.drop(columns=['date'], errors='ignore'),
            on=['year', 'month'],
            how='left', suffixes=('', '_jkp')
        ).merge(
            data_zimmer.drop(columns=['date'], errors='ignore'),
            on=['year', 'month'],
            how='left', suffixes=('', '_zimmer')
        )
        # Tickeurs avec assez d'observations
        ticker_counts = merged.groupby('Ticker').size()
        valid_tickers = ticker_counts[ticker_counts >= 60].index
        merged = merged[merged['Ticker'].isin(valid_tickers)]
        # Virer les colonnes inutiles
        columns_to_drop = [
            'PERMNO', 'HdrCUSIP', 'CUSIP', 'TradingSymbol',
            'PERMCO', 'SICCD', 'NAICS', 'year', 'month', 'MthCalDt',
        ]
        merged.drop(columns=[col for col in columns_to_drop if col in merged.columns], inplace=True)
        # Join avec firm chars
        if not data_comp.empty:
            try:
                merged = merged.merge(
                    data_comp.drop(columns=['gvkey', 'cusip'], errors='ignore'),
                    left_on=['Ticker', 'date'],
                    right_on=['Ticker', 'datadate'],
                    how='left'
                )
                merged.drop(columns=['datadate'], inplace=True, errors='ignore')
            except Exception as e:
                logger.warning(f"Could not merge company characteristics: {e}")
        # Nettoyage final
        merged.dropna(subset=['Ticker', 'date', 'MthRet'], inplace=True)

    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return

    # 4. STRATEGY PIPELINE
    # Initialiser composants
    ml_trader = MLPairsTrader()
    benchmark = BenchmarkStrategy()
    visualizer = Visualizer(output_dir)

    # Split train/test (par date)
    merged = merged.sort_values('date')
    unique_dates = merged['date'].unique()
    split_idx = int(len(unique_dates) * 0.7)
    train_end_date = unique_dates[split_idx]
    test_start_date = unique_dates[split_idx + 1]
    logger.info(f"Train: jusqu'à {train_end_date.date()}, test: dès {test_start_date.date()}")

    # Préparer et générer les datasets
    enhanced_data = ml_trader.prepare_data(merged)
    pair_data_train = ml_trader.generate_pair_dataset(enhanced_data, train_end_date)
    if pair_data_train.empty:
        logger.error("No valid pair data generated")
        return
    labeled_data = ml_trader.create_labels(pair_data_train)
    if labeled_data.empty:
        logger.error("No labeled data created")
        return
    if not ml_trader.train_pair_selection_model(labeled_data):
        logger.error("Failed to train pair selection model")
        return
    if not ml_trader.train_threshold_model(labeled_data):
        logger.error("Failed to train threshold model")
        return
    # Pairs pour tout le dataset
    pair_data_all = ml_trader.generate_pair_dataset(enhanced_data)
    ml_selected_pairs = ml_trader.select_pairs_ml(pair_data_all)
    benchmark_selected_pairs = benchmark.select_pairs_traditional(enhanced_data, pair_data_all)

    # Run les stratégies
    ml_results = ml_trader.execute_ml_strategy(
        enhanced_data, 
        pair_data_all,
        ml_selected_pairs, 
        start_date=test_start_date
    )
    benchmark_results = benchmark.execute_benchmark_strategy(
        enhanced_data,
        pair_data_all,
        benchmark_selected_pairs, 
        start_date=test_start_date
    )

    # Visualisation
    visualizer.plot_comprehensive_comparison(ml_results, benchmark_results)
    visualizer.save_results_summary(ml_results, benchmark_results)

    # Sauvegarde modèles
    save_models(ml_trader, output_dir)
    logger.info("Process finished!")

    return ml_results, benchmark_results


def save_models(ml_trader: MLPairsTrader, output_dir: str):
    """Save trained models and scalers"""
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save PyTorch models
    if ml_trader.pair_selection_model is not None:
        torch.save({
            'model_state_dict': ml_trader.pair_selection_model.state_dict(),
            'selected_features': ml_trader.selected_features_pairs,
            'scaler': ml_trader.scaler_pairs
        }, os.path.join(models_dir, 'pair_selection_model.pt'))
    
    if ml_trader.threshold_model is not None:
        torch.save({
            'model_state_dict': ml_trader.threshold_model.state_dict(),
            'selected_features': ml_trader.selected_features_threshold,
            'scaler': ml_trader.scaler_threshold
        }, os.path.join(models_dir, 'threshold_model.pt'))
    
    logger.info(f"Models saved to {models_dir}")


def load_models(ml_trader: MLPairsTrader, models_dir: str):
    """Load saved models and scalers"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pair selection model
    pair_model_path = os.path.join(models_dir, 'pair_selection_model.pt')
    if os.path.exists(pair_model_path):
        checkpoint = torch.load(pair_model_path, map_location=device)
        
        ml_trader.selected_features_pairs = checkpoint['selected_features']
        ml_trader.scaler_pairs = checkpoint['scaler']
        
        # Initialize model with correct input size
        input_size = len(ml_trader.selected_features_pairs)
        ml_trader.pair_selection_model = PairSelectionModel(
            input_size=input_size,
            hidden_sizes=CONFIG.pair_selection_hidden,
            dropout_rate=CONFIG.dropout_rate
        ).to(device)
        
        ml_trader.pair_selection_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Pair selection model loaded")
    
    # Load threshold model
    threshold_model_path = os.path.join(models_dir, 'threshold_model.pt')
    if os.path.exists(threshold_model_path):
        checkpoint = torch.load(threshold_model_path, map_location=device)
        
        ml_trader.selected_features_threshold = checkpoint['selected_features']
        ml_trader.scaler_threshold = checkpoint['scaler']
        
        # Initialize model with correct input size
        input_size = len(ml_trader.selected_features_threshold)
        ml_trader.threshold_model = ThresholdOptimizationModel(
            input_size=input_size,
            hidden_size=CONFIG.threshold_lstm_hidden,
            num_layers=CONFIG.threshold_lstm_layers,
            dropout_rate=CONFIG.dropout_rate
        ).to(device)
        
        ml_trader.threshold_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Threshold model loaded")


if __name__ == "__main__":

    run_with_real_data()