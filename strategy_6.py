import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
import warnings
from typing import Dict, List, Tuple, Optional
import pickle
from datetime import datetime, timedelta
import seaborn as sns
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration avancée pour Deep Learning
DEEP_LEARNING_CONFIG = {
    # Sélection de paires
    'correlation_window': 24,
    'min_correlation': 0.4, 
    'min_cointegration_pvalue': 0.15,
    'min_observations': 100,  # 9 ans de données
    'lookback_window': 12,
    
    # Deep Learning
    'sequence_length': 6,  # Utilise 6 mois de séquence
    'hidden_dim': 32,
    'num_layers': 3,
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'early_stopping_patience': 15,
    
    # Trading
    'dynamic_threshold': True,  # Seuils adaptatifs
    'base_zscore_entry': 1.8,
    'base_zscore_exit': 0.3,
    'volatility_adjustment': True,
    'regime_detection': True,
    
    # Risk Management
    'max_position_size': 0.08,
    'var_confidence': 0.05,  # Value at Risk
    'kelly_fraction': 0.25,  # Kelly criterion
    'dynamic_sizing': True,
    
    # Portfolio
    'max_pairs': 15,
    'rebalance_frequency': 6,
    'ensemble_models': 3
}

class AdvancedFeatureEngineer:
    """
    Classe d'ingénierie de features avancée pour le trading quantitatif.
    
    Cette classe fournit un ensemble complet d'outils pour transformer des données
    financières brutes en features sophistiquées utilisables pour l'apprentissage automatique :
    - Features techniques individuelles (momentum, volatilité, distribution)
    - Détection de régimes de marché (bull/bear, haute/basse volatilité)
    - Features de paires d'actifs (spread, corrélation, bêta)
    - Nettoyage robuste des outliers et gestion des valeurs aberrantes
    """
    
    def __init__(self):
        # Dictionnaire pour stocker les objets de normalisation (scalers)
        # Peut être utilisé pour standardiser les features avant modélisation
        self.scalers = {}
        
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Génère un ensemble complet de features techniques à partir des rendements mensuels.
        - Protection contre les valeurs extrêmes et les erreurs numériques
        - Calculs de momentum sur multiple horizons temporels
        - Mesures de volatilité et de forme de distribution
        - Indicateurs de tendance et de retournement
        - Features relatives au marché (si données disponibles)
        
        Args:
            data: DataFrame avec colonne 'MthRet' (rendements mensuels)
                  et optionnellement 'sprtrn' (rendements du marché)
        
        Returns:
            DataFrame enrichi avec toutes les features techniques
        """
        # Copie pour préserver les données originales
        data = data.copy()
        
        # === VALIDATION ET PRÉPARATION DES DONNÉES ===
        # Vérification de la présence de la colonne principale 'MthRet'
        if 'MthRet' not in data.columns:
            logger.warning("MthRet column not found, using first numeric column")
            # Fallback : utilise la première colonne numérique disponible
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data['MthRet'] = data[numeric_cols[0]]
            else:
                # Cas d'urgence : initialise avec des zéros
                data['MthRet'] = 0
        
        # Premier nettoyage des rendements - crucial pour la stabilité des calculs suivants
        data['MthRet'] = self._clean_extreme_values(data['MthRet'])
        
        # === TRANSFORMATION LOGARITHMIQUE PROTÉGÉE ===
        # Les log-rendements sont plus appropriés pour l'analyse statistique
        # Clipping pour éviter log(0) ou log(négatif) qui causeraient des erreurs
        returns_safe = np.clip(data['MthRet'].fillna(0), -0.5, 2.0)  # Limite : -50% à +200%
        data['log_return'] = np.log(1 + returns_safe)
        # Nettoyage supplémentaire des log-rendements
        data['log_return'] = self._clean_extreme_values(data['log_return'])
        
        # === INDICATEURS DE MOMENTUM (PERSISTANCE DES TENDANCES) ===
        # Ces moyennes mobiles capturent la persistance des mouvements de prix
        # sur différents horizons temporels - fondamental pour détecter les tendances
        data['price_momentum_3m'] = data['MthRet'].rolling(3, min_periods=1).mean()    # Court terme
        data['price_momentum_6m'] = data['MthRet'].rolling(6, min_periods=1).mean()    # Moyen terme  
        data['price_momentum_12m'] = data['MthRet'].rolling(12, min_periods=1).mean()  # Long terme
        
        # === MESURES DE VOLATILITÉ RÉALISÉE (MESURE DU RISQUE) ===
        # La volatilité quantifie l'instabilité des prix = proxy du risque
        # fillna(0.01) assure une volatilité minimale pour éviter les divisions par zéro
        data['volatility_3m'] = data['MthRet'].rolling(3, min_periods=1).std().fillna(0.01)   # Vol court terme
        data['volatility_6m'] = data['MthRet'].rolling(6, min_periods=1).std().fillna(0.01)   # Vol moyen terme
        data['volatility_12m'] = data['MthRet'].rolling(12, min_periods=1).std().fillna(0.01) # Vol long terme
        
        # Nettoyage spécifique des mesures de volatilité (sensibles aux outliers)
        for col in ['volatility_3m', 'volatility_6m', 'volatility_12m']:
            data[col] = self._clean_extreme_values(data[col])
        
        # === MESURES DE FORME DE DISTRIBUTION ===
        # Ces statistiques caractérisent la forme de la distribution des rendements
        # Critiques pour comprendre les risques de queue (tail risks)
        
        # Skewness (asymétrie) : distribution penchée vers gains (+) ou pertes (-)
        data['skewness_6m'] = data['MthRet'].rolling(6, min_periods=3).skew().fillna(0)
        # Kurtosis (aplatissement) : mesure des "queues grasses" = risques extrêmes
        data['kurtosis_6m'] = data['MthRet'].rolling(6, min_periods=3).kurt().fillna(0)
        
        # Limitation stricte pour éviter les valeurs aberrantes qui déstabilisent les modèles
        data['skewness_6m'] = np.clip(data['skewness_6m'], -10, 10)
        data['kurtosis_6m'] = np.clip(data['kurtosis_6m'], -10, 50)
        
        # === DÉTECTION DE TENDANCE LOCALE ===
        def calc_trend_safe(x):
            """
            Calcule la pente de tendance sur une fenêtre glissante avec protection robuste.
            
            Cette fonction applique une régression linéaire locale pour identifier
            la direction et l'intensité de la tendance, tout en étant résistante
            aux outliers et aux données manquantes.
            
            Args:
                x: Série de rendements sur la fenêtre
                
            Returns:
                float: Pente de la tendance (limitée entre -0.1 et 0.1)
            """
            # Vérification de suffisamment de données pour une régression fiable
            if len(x.dropna()) < 3:
                return 0
            try:
                # Forward-fill puis remplissage par zéro des valeurs manquantes
                x_clean = x.fillna(method='ffill').fillna(0)
                # Clipping pour réduire l'impact des outliers sur la régression
                x_clean = np.clip(x_clean, -1, 1)
                # Régression linéaire : y = ax + b, extraction de la pente 'a'
                trend = np.polyfit(range(len(x_clean)), x_clean, 1)[0]
                # Limitation de l'amplitude pour éviter les tendances aberrantes
                return np.clip(trend, -0.1, 0.1)
            except:
                # Gestion d'erreur (ex: matrice singulière) - retourne tendance neutre
                return 0
        
        # Application sur fenêtre glissante de 6 mois
        data['price_trend'] = data['MthRet'].rolling(6, min_periods=3).apply(calc_trend_safe, raw=False)
        
        # === SIGNAL DE RETOURNEMENT (MEAN REVERSION) ===
        # Ce signal capture la tendance naturelle des prix à revenir vers leur moyenne
        # Combine le rendement précédent (inversé) avec la volatilité actuelle
        # Logique : après un mouvement extrême en période volatile, retournement probable
        data['reversal_signal'] = -data['MthRet'].shift(1) * data['volatility_3m']
        data['reversal_signal'] = self._clean_extreme_values(data['reversal_signal'])
        
        # === FEATURES RELATIVES AU MARCHÉ DE RÉFÉRENCE ===
        # Si données du marché disponibles (ex: rendements S&P 500)
        if 'sprtrn' in data.columns:
            # Nettoyage préalable des données de marché
            data['sprtrn'] = self._clean_extreme_values(data['sprtrn'])
            
            # BÊTA DE MARCHÉ : sensibilité aux mouvements du marché
            # Calculé via corrélation roulante sur 12 mois
            # Bêta > 1 = plus volatil que le marché, < 1 = moins volatil, < 0 = anti-corrélé
            data['market_beta'] = data['MthRet'].rolling(12, min_periods=6).corr(data['sprtrn']).fillna(0)
            
            # EXCÈS DE RENDEMENT : performance relative au marché
            # Positif = surperformance, Négatif = sous-performance
            data['excess_return'] = data['MthRet'] - data['sprtrn'].fillna(0)
            
            # VOLATILITÉ DE L'EXCÈS : consistance de la performance relative
            # Faible = surperformance stable, Élevée = performance erratique
            data['excess_volatility'] = data['excess_return'].rolling(6, min_periods=3).std().fillna(0.01)
            
            # Nettoyage et contraintes sur les métriques de marché
            data['market_beta'] = np.clip(data['market_beta'], -5, 5)  # Bêta raisonnable
            data['excess_return'] = self._clean_extreme_values(data['excess_return'])
            data['excess_volatility'] = self._clean_extreme_values(data['excess_volatility'])
        
        # === NETTOYAGE GLOBAL FINAL ===
        # Application systématique du nettoyage à toutes les colonnes numériques
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'date':  # Préservation des colonnes temporelles
                data[col] = self._clean_extreme_values(data[col])
        
        # === GESTION FINALE DES VALEURS MANQUANTES ===
        # Forward-fill : propagation de la dernière valeur valide (cohérent en finance)
        # Puis remplissage par zéro pour les valeurs encore manquantes
        data = data.fillna(method='ffill').fillna(0)
        
        return data
    
    def _clean_extreme_values(self, series: pd.Series, clip_percentile: float = 99.5) -> pd.Series:
        """
        Méthode robuste de nettoyage des valeurs extrêmes.
        Elle utilise une approche basée sur les percentiles pour identifier et
        limiter les outliers tout en préservant la distribution naturelle des données.
        
        Args:
            series: Série à nettoyer
            clip_percentile: Percentile de clipping (99.5% par défaut)
            
        Returns:
            Série nettoyée avec valeurs extrêmes limitées
        """
        # Gestion des séries vides
        if len(series) == 0:
            return series
            
        # Étape 1 : Remplacement des valeurs infinies par NaN
        # Les inf/-inf peuvent survenir lors de divisions par zéro
        series_clean = series.replace([np.inf, -np.inf], np.nan)
        
        # Étape 2 : Calcul des bornes de clipping basées sur les percentiles
        try:
            # Bornes symétriques autour du percentile spécifié
            lower_bound = series_clean.quantile((100 - clip_percentile) / 100)
            upper_bound = series_clean.quantile(clip_percentile / 100)
            
            # Validation des bornes calculées
            if pd.isna(lower_bound) or pd.isna(upper_bound):
                # Fallback : clipping conservateur si percentiles invalides
                series_clean = np.clip(series_clean.fillna(0), -10, 10)
            else:
                # Application du clipping basé sur les percentiles
                series_clean = np.clip(series_clean, lower_bound, upper_bound)
        except:
            # Gestion d'erreur : clipping par défaut en cas de problème
            series_clean = np.clip(series_clean.fillna(0), -10, 10)
        
        return series_clean
    
    def create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Détecte et encode les régimes de marché dominants.
        
        Cette méthode identifie des phases distinctes du marché basées sur
        la volatilité et la tendance.     
        Régimes détectés :
        - Volatilité : Haute vs Basse (relative aux 24 derniers mois)
        - Tendance : Bull (hausse) vs Bear (baisse) vs Neutre
        
        Args:
            data: DataFrame avec rendements mensuels
            
        Returns:
            DataFrame enrichi avec indicateurs binaires de régimes
        """
        data = data.copy()
        
        # === DÉTECTION DE RÉGIME DE VOLATILITÉ ===
        # Calcul de la volatilité sur 6 mois
        vol_6m = data['MthRet'].rolling(6, min_periods=3).std().fillna(0.01)
        
        # Rang percentile de la volatilité actuelle vs historique (24 mois)
        # Permet de définir "haute" et "basse" volatilité de manière relative
        vol_percentile = vol_6m.rolling(24, min_periods=12).rank(pct=True).fillna(0.5)
        
        # Création d'indicateurs binaires pour les régimes de volatilité
        data['high_vol_regime'] = (vol_percentile > 0.7).astype(int)  # Top 30% = haute volatilité
        data['low_vol_regime'] = (vol_percentile < 0.3).astype(int)   # Bottom 30% = basse volatilité
        
        # === DÉTECTION DE RÉGIME DE TENDANCE ===
        # Moyenne mobile 6 mois comme proxy de la tendance dominante
        trend_6m = data['MthRet'].rolling(6, min_periods=3).mean().fillna(0)
        
        # Calcul des seuils dynamiques basés sur l'historique récent (12 mois)
        # Quantiles 70% et 30% définissent les seuils bull/bear
        trend_quantile_high = trend_6m.rolling(12, min_periods=6).quantile(0.7).fillna(trend_6m.quantile(0.7))
        trend_quantile_low = trend_6m.rolling(12, min_periods=6).quantile(0.3).fillna(trend_6m.quantile(0.3))
        
        # Création d'indicateurs binaires pour les régimes de tendance
        data['bull_regime'] = (trend_6m > trend_quantile_high).astype(int)  # Marché haussier
        data['bear_regime'] = (trend_6m < trend_quantile_low).astype(int)    # Marché baissier
        # Note : Les périodes entre ces seuils correspondent à un régime neutre/sideways
        
        return data
    
    def engineer_pair_features(self, data1: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:
        """
        Génère des features sophistiquées pour l'analyse de paires d'actifs.
        Elle calcule des métriques relationnelles entre deux actifs qui capturent :
        - Les écarts de performance (spreads)
        - Les relations de dépendance (corrélations, bêta)
        - Les opportunités d'arbitrage (z-scores, ratios)
        
        Args:
            data1: DataFrame du premier actif
            data2: DataFrame du second actif
            
        Returns:
            DataFrame avec features de paires alignées temporellement
        """
        # === PRÉPARATION ET VALIDATION DES DONNÉES ===
        # Vérification et création de la colonne MthRet pour chaque actif
        for df in [data1, data2]:
            if 'MthRet' not in df.columns:
                # Fallback vers la première colonne numérique
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df['MthRet'] = df[numeric_cols[0]]
                else:
                    df['MthRet'] = 0
            # Nettoyage précoce des rendements pour éviter la propagation d'outliers
            df['MthRet'] = self._clean_extreme_values(df['MthRet'])
        
        # === ALIGNEMENT TEMPOREL DES DONNÉES ===
        # Fusion sur les dates communes - crucial pour l'analyse de paires
        aligned = pd.merge(data1[['date', 'MthRet']], data2[['date', 'MthRet']], 
                          on='date', suffixes=('_1', '_2'), how='inner')
        
        # Validation de l'alignement
        if len(aligned) == 0:
            logger.warning("No overlapping dates found between assets")
            return pd.DataFrame()
        
        # Nettoyage des rendements alignés
        aligned['MthRet_1'] = self._clean_extreme_values(aligned['MthRet_1'])
        aligned['MthRet_2'] = self._clean_extreme_values(aligned['MthRet_2'])
        
        # === GÉNÉRATION DES FEATURES TECHNIQUES INDIVIDUELLES ===
        # Application des features techniques à chaque actif séparément
        temp1 = data1.copy()
        temp2 = data2.copy()
        temp1 = self.create_technical_features(temp1)
        temp2 = self.create_technical_features(temp2)
        
        # === FUSION DES FEATURES TECHNIQUES ===
        # Récupération des colonnes de features (excluant date et MthRet)
        feature_cols_1 = [col for col in temp1.columns if col not in ['date', 'MthRet']]
        feature_cols_2 = [col for col in temp2.columns if col not in ['date', 'MthRet']]
        
        # Fusion des features avec suffixes pour différencier les actifs
        if feature_cols_1:
            aligned = pd.merge(aligned, temp1[['date'] + feature_cols_1], 
                              on='date', how='left')
        if feature_cols_2:
            aligned = pd.merge(aligned, temp2[['date'] + feature_cols_2], 
                              on='date', suffixes=('_1', '_2'), how='left')
        
        # === CALCUL DES SPREADS (ÉCARTS) ===
        # Spread des rendements : différence de performance brute
        aligned['return_spread'] = aligned['MthRet_1'] - aligned['MthRet_2']
        aligned['return_spread'] = self._clean_extreme_values(aligned['return_spread'])
        
        # Spread logarithmique : plus approprié statistiquement
        log_ret_1 = aligned.get('log_return_1', np.log(1 + aligned['MthRet_1']))
        log_ret_2 = aligned.get('log_return_2', np.log(1 + aligned['MthRet_2']))
        aligned['log_spread'] = log_ret_1 - log_ret_2
        aligned['log_spread'] = self._clean_extreme_values(aligned['log_spread'])
        
        # === STATISTIQUES DU SPREAD POUR MEAN REVERSION ===
        # Moyenne mobile du spread (niveau "normal")
        aligned['spread_ma'] = aligned['return_spread'].rolling(12, min_periods=6).mean().fillna(0)
        # Volatilité du spread (mesure de stabilité de la relation)
        aligned['spread_std'] = aligned['return_spread'].rolling(12, min_periods=6).std().fillna(0.01)
        
        # Z-score du spread : signal d'arbitrage (mean reversion)
        # Protection contre division par zéro
        std_safe = np.maximum(aligned['spread_std'], 0.001)
        aligned['spread_zscore'] = (aligned['return_spread'] - aligned['spread_ma']) / std_safe
        aligned['spread_zscore'] = self._clean_extreme_values(aligned['spread_zscore'])
        
        # === CORRÉLATION DYNAMIQUE ===
        # Corrélation glissante sur différents horizons
        # Mesure la force et la direction de la relation entre les actifs
        corr_6m = aligned['MthRet_1'].rolling(6, min_periods=3).corr(aligned['MthRet_2'])
        corr_12m = aligned['MthRet_1'].rolling(12, min_periods=6).corr(aligned['MthRet_2'])
        
        # Limitation des corrélations dans l'intervalle [-1, 1]
        aligned['correlation_6m'] = np.clip(corr_6m.fillna(0), -1, 1)
        aligned['correlation_12m'] = np.clip(corr_12m.fillna(0), -1, 1)
        
        # === RATIO DE VOLATILITÉ ===
        # Compare la volatilité relative des deux actifs
        # Ratio > 1 : actif 1 plus volatil, < 1 : actif 2 plus volatil
        vol1 = aligned.get('volatility_6m_1', aligned['MthRet_1'].rolling(6).std()).fillna(0.01)
        vol2 = aligned.get('volatility_6m_2', aligned['MthRet_2'].rolling(6).std()).fillna(0.01)
        
        # Nettoyage des volatilités individuelles
        vol1 = self._clean_extreme_values(vol1)
        vol2 = self._clean_extreme_values(vol2)
        
        # Calcul du ratio avec protection contre division par zéro
        vol_ratio = vol1 / np.maximum(vol2, 0.001)
        aligned['vol_ratio'] = np.clip(vol_ratio, 0.1, 10)  # Limitation du ratio
        
        # === RELATION BÊTA ENTRE LES ACTIFS ===
        def rolling_beta_safe(window):
            """
            Calcule le bêta glissant entre les deux actifs avec protection robuste.
            
            Le bêta mesure la sensibilité de l'actif 1 aux mouvements de l'actif 2.
            Formule : Beta = Covariance(Asset1, Asset2) / Variance(Asset2)
            
            Args:
                window: Taille de la fenêtre glissante
                
            Returns:
                Série de bêtas avec outliers limités
            """
            # Covariance glissante entre les deux actifs
            cov = aligned['MthRet_1'].rolling(window, min_periods=window//2).cov(aligned['MthRet_2'])
            # Variance glissante de l'actif de référence (actif 2)
            var = aligned['MthRet_2'].rolling(window, min_periods=window//2).var()
            
            # Nettoyage des composants
            cov = self._clean_extreme_values(cov)
            var = self._clean_extreme_values(var)
            
            # Calcul du bêta avec protection contre division par zéro
            beta = cov / np.maximum(var, 1e-6)
            # Limitation du bêta pour éviter les valeurs aberrantes
            return np.clip(beta.fillna(1), -10, 10)
        
        # Application sur différents horizons temporels
        aligned['beta_6m'] = rolling_beta_safe(6)   # Bêta court terme
        aligned['beta_12m'] = rolling_beta_safe(12) # Bêta long terme
        
        # === NETTOYAGE GLOBAL FINAL ===
        # Application systématique du nettoyage à toutes les colonnes numériques
        numeric_cols = aligned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['date']:  # Préservation des colonnes temporelles
                aligned[col] = self._clean_extreme_values(aligned[col])
        
        # === GESTION COMPLÈTE DES VALEURS MANQUANTES ===
        # Forward-fill suivi d'un remplissage par zéro
        aligned = aligned.fillna(method='ffill').fillna(0)
        
        # === VALIDATION FINALE DE LA PROPRETÉ DES DONNÉES ===
        # Vérification et nettoyage final des valeurs infinies ou NaN résiduelles
        for col in aligned.select_dtypes(include=[np.number]).columns:
            if np.any(np.isinf(aligned[col])) or np.any(np.isnan(aligned[col])):
                # Remplacement des inf/-inf par 0 et des NaN par 0
                aligned[col] = aligned[col].replace([np.inf, -np.inf], 0).fillna(0)
        
        return aligned

class LSTMPairsPredictor(nn.Module):
    """
    Modèle LSTM avancé pour la prédiction de paires d'actifs financiers.
    
    Cette classe implémente un réseau de neurones sophistiqué spécialement conçu
    pour prédire simultanément plusieurs métriques d'une paire d'actifs :
    - Rendements futurs des deux actifs
    - Volatilités futures des deux actifs  
    - Corrélation future entre les actifs
    
    Architecture :
    - Couches LSTM bidirectionnelles avec dropout pour capturer les dépendances temporelles
    - Mécanisme d'attention simplifié pour identifier les périodes les plus importantes
    - Têtes de prédiction séparées pour chaque type de métrique
    - Contraintes de sortie appropriées (volatilités positives, corrélation [-1,1])
    """
    
    def __init__(self,
                 input_size: int,           # Nombre de features d'entrée
                 hidden_dim: int = 128,     # Dimension des états cachés LSTM
                 num_layers: int = 4,       # Nombre de couches LSTM empilées
                 dropout: float = 0.2,      # Taux de dropout pour régularisation
                 output_size: int = 2):     # Dimension de la couche de sortie de base
        """
        Initialise l'architecture du modèle LSTM pour pairs trading.
        
        Args:
            input_size: Nombre de features en entrée (ex: 50 indicators techniques)
            hidden_dim: Taille des vecteurs d'état cachés du LSTM (256 par défaut)
            num_layers: Profondeur du réseau LSTM (8 couches par défaut)
            dropout: Probabilité de dropout entre les couches (0.2 = 20%)
            output_size: Dimension de la représentation finale avant les têtes spécialisées
        """
        super(LSTMPairsPredictor, self).__init__()
        
        # Stockage des hyperparamètres pour référence
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # === COUCHES LSTM PRINCIPALES ===
        self.lstm = nn.LSTM(
            input_size,     # Nombre de features d'entrée à chaque timestep
            hidden_dim,     # Dimension de l'état caché
            num_layers,     # Nombre de couches LSTM empilées
            batch_first=True,   # Format (batch, sequence, features) plus intuitif
            # Dropout uniquement si plusieurs couches (évite l'overfitting)
            dropout=dropout if num_layers > 1 else 0
        )
        
        # === MÉCANISME D'ATTENTION SIMPLIFIÉ ===
        # L'attention permet au modèle de se concentrer sur les timesteps les plus pertinents
        # plutôt que de traiter uniformément toute la séquence temporelle
        self.attention_dim = min(hidden_dim, 32)  # Dimension réduite pour efficacité
        
        # Transformation linéaire pour projeter les états cachés vers l'espace d'attention
        self.attention_w = nn.Linear(hidden_dim, self.attention_dim)
        # Couche finale pour calculer les scores d'attention (sans biais)
        self.attention_v = nn.Linear(self.attention_dim, 1, bias=False)
        
        # === COUCHES D'EXTRACTION DE FEATURES ===
        # Réseau feedforward pour transformer la représentation LSTM+attention
        # en features de haut niveau adaptées aux prédictions financières
        self.feature_layers = nn.Sequential(
            # Première réduction dimensionnelle avec non-linéarité
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),  # Activation ReLU pour introduire la non-linéarité
            nn.Dropout(dropout),  # Régularisation contre l'overfitting
            
            # Seconde réduction pour concentrer l'information
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # === COUCHE DE SORTIE DE BASE ===
        # Transformation finale vers une représentation compacte
        # qui sera utilisée par toutes les têtes de prédiction
        self.output_layers = nn.Linear(hidden_dim // 4, output_size)
        
        # === TÊTES DE PRÉDICTION SPÉCIALISÉES ===
        # Chaque tête est optimisée pour un type spécifique de prédiction
        
        # TÊTE RENDEMENTS : Prédit les rendements futurs des deux actifs
        # Sortie : [rendement_actif1, rendement_actif2]
        self.return_head = nn.Linear(output_size, 2)
        
        # TÊTE VOLATILITÉS : Prédit les volatilités futures des deux actifs  
        # Sortie : [volatilité_actif1, volatilité_actif2]
        self.volatility_head = nn.Linear(output_size, 2)
        
        # TÊTE CORRÉLATION : Prédit la corrélation future entre les actifs
        # Sortie : corrélation_future (scalaire)
        self.correlation_head = nn.Linear(output_size, 1)
    
    def forward(self, x):
        """
        Passe avant du modèle : transforme une séquence de features en prédictions.
        
        Ce processus se déroule en plusieurs étapes :
        1. Traitement LSTM pour capturer les dépendances temporelles
        2. Application du mécanisme d'attention pour identifier les périodes clés
        3. Extraction de features de haut niveau
        4. Génération de prédictions spécialisées avec contraintes appropriées
        
        Args:
            x: Tensor de forme (batch_size, sequence_length, input_size)
               Contient les séquences de features techniques pour chaque échantillon
        
        Returns:
            Dictionnaire contenant :
            - 'returns': Prédictions de rendements [batch_size, 2]
            - 'volatilities': Prédictions de volatilités [batch_size, 2] (garanties positives)
            - 'correlation': Prédiction de corrélation [batch_size, 1] (contrainte [-1,1])
            - 'attention_weights': Poids d'attention [batch_size, sequence_length, 1]
        """
        # Récupération de la taille du batch pour les initialisations
        batch_size = x.size(0)
        
        # === ÉTAPE 1 : TRAITEMENT LSTM ===
        # Le LSTM traite la séquence temporelle et produit :
        # - lstm_out : états cachés à chaque timestep [batch, seq_len, hidden_dim]
        # - (hidden, cell) : états finaux (non utilisés ici car on utilise l'attention)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # === ÉTAPE 2 : MÉCANISME D'ATTENTION SIMPLIFIÉ ===
        # L'attention calcule l'importance relative de chaque timestep
        
        # Transformation des états cachés vers l'espace d'attention
        # Forme : [batch, seq_len, attention_dim]
        attention_transformed = self.attention_w(lstm_out)
        
        # Application de tanh pour normalisation, puis projection vers scores scalaires
        # Forme : [batch, seq_len, 1]
        attention_scores = self.attention_v(torch.tanh(attention_transformed))
        
        # Normalisation softmax pour obtenir des poids qui somment à 1
        # Les poids élevés indiquent les timesteps les plus pertinents
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Agrégation pondérée des états cachés selon les poids d'attention
        # Forme finale : [batch, hidden_dim]
        attended_out = torch.sum(lstm_out * attention_weights, dim=1)
        
        # === ÉTAPE 3 : EXTRACTION DE FEATURES DE HAUT NIVEAU ===
        # Transformation de la représentation LSTM+attention en features adaptées
        # aux prédictions financières spécifiques
        features = self.feature_layers(attended_out)
        
        # Couche de base commune à toutes les têtes de prédiction
        base_output = self.output_layers(features)
        
        # === ÉTAPE 4 : PRÉDICTIONS SPÉCIALISÉES AVEC CONTRAINTES ===
        
        # PRÉDICTION DES RENDEMENTS
        # Pas de contrainte particulière - les rendements peuvent être positifs ou négatifs
        # Sortie : [rendement_futur_actif1, rendement_futur_actif2]
        returns = self.return_head(base_output)
        
        # PRÉDICTION DES VOLATILITÉS  
        # Contrainte cruciale : les volatilités doivent être positives
        # Utilisation d'exp() pour garantir des valeurs positives
        # Note : exp() peut causer des valeurs très grandes, surveiller en pratique
        volatilities = torch.exp(self.volatility_head(base_output))
        
        # PRÉDICTION DE LA CORRÉLATION
        # Contrainte : la corrélation doit être dans l'intervalle [-1, 1]
        # tanh() mappe naturellement vers [-1, 1]
        correlation = torch.tanh(self.correlation_head(base_output))
        
        # === RETOUR DES RÉSULTATS ===
        # Dictionnaire structuré pour un accès facile aux différentes prédictions
        return {
            'returns': returns,                    # [batch_size, 2] - Rendements futurs
            'volatilities': volatilities,          # [batch_size, 2] - Volatilités futures (>0)
            'correlation': correlation,            # [batch_size, 1] - Corrélation future [-1,1]
            'attention_weights': attention_weights # [batch_size, seq_len, 1] - Pour analyse/debug
        }

class TransformerPairsModel(nn.Module):
    """
    Modèle Transformer spécialisé pour la capture des dépendances long-terme en pairs trading.
    
    Cette architecture utilise le mécanisme d'attention des Transformers pour capturer
    efficacement les relations complexes et les dépendances à long terme dans les séries
    temporelles financières. Contrairement aux LSTM qui traitent séquentiellement,
    les Transformers peuvent examiner simultanément tous les timesteps.
    
    Avantages par rapport aux LSTM :
    - Parallélisation complète (pas de dépendance séquentielle)
    - Capture directe des dépendances long-terme sans dégradation du gradient
    - Mécanisme d'attention explicite pour identifier les patterns temporels importants
    - Meilleure performance sur les longues séquences
    
    Prédictions :
    - Rendements futurs des deux actifs de la paire
    - Direction du spread (convergence/divergence prédite)
    """
    
    def __init__(self, 
                 input_size: int,           # Nombre de features d'entrée par timestep
                 d_model: int = 128,        # Dimension du modèle Transformer
                 nhead: int = 8,            # Nombre de têtes d'attention
                 num_layers: int = 8,       # Nombre de couches d'encodeur
                 dropout: float = 0.2):     # Taux de dropout pour régularisation
        """
        Initialise l'architecture Transformer pour pairs trading.
        
        Args:
            input_size: Nombre de features techniques d'entrée (ex: 50 indicateurs)
            d_model: Dimension interne du modèle (doit être divisible par nhead)
            nhead: Nombre de têtes d'attention parallèles (8 est un standard)
            num_layers: Profondeur du modèle (4-6 couches typiques pour la finance)
            dropout: Probabilité de dropout (0.1-0.3 recommandé)
        """
        super(TransformerPairsModel, self).__init__()
        
        # Stockage de la dimension du modèle pour référence
        self.d_model = d_model
        
        # === PROJECTION D'ENTRÉE ===
        # Transformation linéaire des features d'entrée vers la dimension du modèle
        # Crucial car les Transformers opèrent dans un espace de dimension fixe
        self.input_projection = nn.Linear(input_size, d_model)
        
        # === ENCODAGE POSITIONNEL ===
        # Les Transformers n'ont pas de notion inhérente de position temporelle
        # L'encodage positionnel injecte cette information cruciale pour les séries temporelles
        # register_buffer : paramètre non-entraînable persisté avec le modèle
        self.register_buffer('pos_encoding', self._generate_positional_encoding(1000, d_model))
        
        # === ARCHITECTURE TRANSFORMER ===
        # Création d'une couche d'encodeur Transformer standard
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,        # Dimension des embeddings
            nhead=nhead,            # Nombre de têtes d'attention parallèles
            dropout=dropout,        # Régularisation
            batch_first=True        # Format (batch, sequence, features) plus intuitif
        )
        
        # Empilement de plusieurs couches d'encodeur pour créer la profondeur
        # Plus de couches = capture de patterns plus complexes mais risque d'overfitting
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # === COUCHES DE SORTIE ===
        # Réseau feedforward pour transformer la représentation Transformer
        # en prédictions financières spécifiques
        self.output_layers = nn.Sequential(
            # Réduction dimensionnelle avec non-linéarité
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),  # Activation pour capturer les non-linéarités
            nn.Dropout(dropout),  # Régularisation contre l'overfitting
            
            # Couche finale vers 3 prédictions spécialisées
            nn.Linear(d_model // 2, 3)  # [rendement_actif1, rendement_actif2, direction_spread]
        )
    
    def _generate_positional_encoding(self, max_len: int, d_model: int):
        """
        Génère l'encodage positionnel sinusoïdal pour injecter l'information temporelle.
        
        Cette méthode implémente l'encodage positionnel classique des Transformers
        utilisant des fonctions sinusoïdales et cosinusoïdales de différentes fréquences.
        Cette approche permet au modèle de distinguer les positions temporelles
        et de capturer des patterns périodiques à différentes échelles.
        
        Formule mathématique :
        - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        Args:
            max_len: Longueur maximale de séquence supportée (1000 par sécurité)
            d_model: Dimension du modèle
            
        Returns:
            Tensor d'encodage positionnel de forme [1, max_len, d_model]
        """
        # Initialisation de la matrice d'encodage positionnel
        pe = torch.zeros(max_len, d_model)
        
        # Création du vecteur des positions [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Calcul du terme de division pour les fréquences sinusoïdales
        # Crée des fréquences exponentiellement décroissantes
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(np.log(10000.0) / d_model))
        
        # Application des fonctions sinusoïdales aux dimensions paires
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Application des fonctions cosinusoïdales aux dimensions impaires
        # Gestion du cas où d_model est impair
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Ajout d'une dimension batch pour compatibilité [1, max_len, d_model]
        return pe.unsqueeze(0)
    
    def forward(self, x):
        """
        Passe avant du modèle Transformer pour prédiction de paires.
        
        Le processus de traitement suit l'architecture Transformer standard
        adaptée aux séries temporelles financières :
        1. Projection des features vers l'espace du modèle
        2. Ajout de l'encodage positionnel pour l'information temporelle
        3. Traitement par les couches Transformer (attention + feedforward)
        4. Extraction des prédictions finales du dernier timestep
        
        Args:
            x: Tensor d'entrée de forme [batch_size, sequence_length, input_size]
               Contient les séquences de features techniques pour chaque échantillon
        
        Returns:
            Dictionnaire contenant :
            - 'asset1_return': Prédiction de rendement pour le premier actif
            - 'asset2_return': Prédiction de rendement pour le second actif  
            - 'spread_direction': Direction prédite du spread [-1,1]
        """
        # Récupération de la longueur de séquence pour l'encodage positionnel
        seq_len = x.size(1)
        
        # === ÉTAPE 1 : PROJECTION VERS L'ESPACE DU MODÈLE ===
        # Transformation des features d'entrée vers la dimension d_model
        # Forme : [batch_size, seq_len, input_size] -> [batch_size, seq_len, d_model]
        x = self.input_projection(x)
        
        # === ÉTAPE 2 : INJECTION DE L'INFORMATION TEMPORELLE ===
        # Ajout de l'encodage positionnel pour que le modèle comprenne l'ordre temporel
        # Extraction de la portion correspondante à la longueur de séquence actuelle
        # .to(x.device) assure que l'encodage est sur le même device (CPU/GPU) que les données
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        
        # === ÉTAPE 3 : TRAITEMENT TRANSFORMER ===
        # Passage à travers les couches d'encodeur Transformer
        # Chaque couche applique :
        # 1. Multi-head attention (capture des relations entre timesteps)
        # 2. Feedforward network (transformation non-linéaire)
        # 3. Connexions résiduelles et normalisation de couche
        output = self.transformer(x)
        
        # === ÉTAPE 4 : EXTRACTION DES PRÉDICTIONS ===
        # Utilisation du dernier timestep comme représentation finale
        # Justification : contient l'information agrégée de toute la séquence
        # via le mécanisme d'attention
        final_output = self.output_layers(output[:, -1, :])
        
        # === STRUCTURATION DES SORTIES AVEC CONTRAINTES APPROPRIÉES ===
        return {
            # Prédictions de rendements pour les deux actifs (sans contrainte)
            'asset1_return': final_output[:, 0],
            'asset2_return': final_output[:, 1],
            
            # Direction du spread avec contrainte [-1, 1]
            # -1 : spread devrait diminuer (convergence)
            # +1 : spread devrait augmenter (divergence)
            # tanh() assure la contrainte mathématique
            'spread_direction': torch.tanh(final_output[:, 2])
        }

class EnsemblePairsModel:
    """
    Système d'ensemble de modèles pour maximiser la robustesse des prédictions en pairs trading.
    
    Cette classe implémente une approche d'ensemble learning qui combine plusieurs modèles
    de deep learning pour réduire la variance des prédictions et améliorer la généralisation.
    L'ensemble learning est particulièrement crucial en finance où les données sont bruitées
    et les patterns peuvent changer rapidement.
    
    Fonctionnalités principales :
    - Entraînement de multiples modèles avec des architectures/hyperparamètres variés
    - Gestion robuste des données avec nettoyage extensif des outliers
    - Prédictions d'ensemble avec pondération par la confiance
    - Gestion complète du cycle de vie : feature engineering → normalisation → entraînement → prédiction
    
    Avantages de l'ensemble :
    - Réduction de l'overfitting (diversité des modèles)
    - Robustesse aux données aberrantes
    - Meilleure généralisation sur nouveaux données
    - Estimation de confiance des prédictions
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le système d'ensemble avec la configuration spécifiée.
        
        Args:
            config: Dictionnaire contenant les hyperparamètres :
                   - min_observations: Nombre minimum d'observations requis
                   - sequence_length: Longueur des séquences temporelles
                   - ensemble_models: Nombre de modèles dans l'ensemble
                   - hidden_dim, num_layers, dropout_rate: Paramètres des réseaux
                   - learning_rate, epochs, early_stopping_patience: Paramètres d'entraînement
        """
        self.config = config
        self.models = []  # Stockage des modèles entraînés et de leurs métadonnées
        self.scalers = []  # Normalisateurs pour chaque paire d'actifs
        # Initialisation de l'ingénieur de features pour le preprocessing
        self.feature_engineer = AdvancedFeatureEngineer()
        
    def create_sequences(self, data: pd.DataFrame, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforme les données tabulaires en séquences temporelles pour l'entraînement.
        
        Cette méthode est cruciale car elle convertit les données financières en format
        approprié pour les réseaux de neurones récurrents (LSTM/Transformer).
        Chaque séquence contient 'sequence_length' timesteps consécutifs comme features
        et le timestep suivant comme target.
        
        Args:
            data: DataFrame avec les features et les colonnes target ('MthRet_1', 'MthRet_2')
            sequence_length: Nombre de timesteps dans chaque séquence d'entrée
            
        Returns:
            Tuple (X, y) où :
            - X: Array de forme [n_sequences, sequence_length, n_features]
            - y: Array de forme [n_sequences, 2] (rendements des deux actifs)
        """
        sequences = []
        targets = []
        
        # === VALIDATION DES COLONNES REQUISES ===
        # Vérification que les colonnes target sont présentes
        required_cols = ['MthRet_1', 'MthRet_2']
        for col in required_cols:
            if col not in data.columns:
                logger.error(f"Missing required column: {col}")
                return np.array([]), np.array([])
        
        # Nettoyage préalable des données manquantes
        data_clean = data.dropna()
        
        # === CRÉATION DES SÉQUENCES GLISSANTES ===
        # Parcours des données avec une fenêtre glissante
        for i in range(sequence_length, len(data_clean)- 1):
            try:
                # Extraction de la séquence d'entrée (features sur 'sequence_length' timesteps)
                # iloc[i-sequence_length:i] = fenêtre de 'sequence_length' timesteps précédents
                seq = data_clean.iloc[i-sequence_length:i].select_dtypes(include=[np.number]).values
                
                # Extraction du target (rendements au timestep i)
                target = data_clean.iloc[i+1][required_cols].values
                
                # === VALIDATION DE QUALITÉ DES DONNÉES ===
                # Vérification que la séquence a des features valides et pas de NaN
                if seq.shape[1] > 0 and not np.isnan(seq).any() and not np.isnan(target).any():
                    sequences.append(seq)
                    targets.append(target)
                    
            except Exception as e:
                logger.warning(f"Error creating sequence at index {i}: {e}")
                continue
        
        # Validation finale - s'assurer qu'on a des séquences valides
        if len(sequences) == 0:
            logger.warning("No valid sequences created")
            return np.array([]), np.array([])
        
        return np.array(sequences), np.array(targets)
    
    def train_ensemble(self, pair_data: pd.DataFrame, ticker1: str, ticker2: str):
        """
        Entraîne un ensemble de modèles pour une paire d'actifs spécifique.
        Split temporel AVANT feature engineering pour éviter le look-ahead bias.
        """
        logger.info(f"Training ensemble for pair {ticker1}-{ticker2}")
    
        # === VALIDATION INITIALE DES DONNÉES ===
        if len(pair_data) < self.config['min_observations']:
            logger.warning(f"Insufficient data for {ticker1}-{ticker2}: {len(pair_data)} < {self.config['min_observations']}")
            return False
    
        # === ÉTAPE 1 : SPLIT TEMPOREL AVANT TOUT PREPROCESSING ===
        # Trier par date pour respecter l'ordre chronologique
        if 'date' in pair_data.columns:
            pair_data = pair_data.sort_values('date').reset_index(drop=True)
    
        # Split temporel sur les données brutes (80%/20%)
        split_idx = max(self.config['min_observations'] // 2, int(len(pair_data) * 0.8))
        train_data = pair_data.iloc[:split_idx].copy()
        test_data = pair_data.iloc[split_idx:].copy()
    
        logger.info(f"Split: {len(train_data)} train, {len(test_data)} test observations")
    
        # === ÉTAPE 2 : FEATURE ENGINEERING SÉPARÉ POUR TRAIN ET TEST ===
    
        # Identification des colonnes numériques
        numeric_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
        if 'date' in numeric_columns:
            numeric_columns.remove('date')
    
        if len(numeric_columns) == 0:
            logger.warning(f"No numeric features found for {ticker1}-{ticker2}")
            return False
    
        try:
            # === PREPROCESSING DU TRAINING SET ===
            # 1. Nettoyage des valeurs extrêmes sur train seulement
            train_to_scale = train_data[numeric_columns].copy()
            for col in numeric_columns:
                train_to_scale[col] = self.feature_engineer._clean_extreme_values(train_to_scale[col])
        
            # 2. Vérification des valeurs problématiques
            if np.any(np.isinf(train_to_scale.values)) or np.any(np.isnan(train_to_scale.values)):
                logger.warning(f"Found inf/nan in train data for {ticker1}-{ticker2}, cleaning...")
                train_to_scale = train_to_scale.replace([np.inf, -np.inf], np.nan).fillna(0)
                for col in train_to_scale.columns:
                    train_to_scale[col] = np.clip(train_to_scale[col], -100, 100)
        
            # 3. Fit du scaler UNIQUEMENT sur les données d'entraînement
            scaler = RobustScaler()
            train_scaled = scaler.fit_transform(train_to_scale)
            train_df = pd.DataFrame(train_scaled, columns=numeric_columns, index=train_data.index)
        
            # 4. Ajout des colonnes target si nécessaires
            if 'MthRet_1' not in train_df.columns:
                train_df['MthRet_1'] = train_df.iloc[:, 0] if len(train_df.columns) > 0 else 0
            if 'MthRet_2' not in train_df.columns:
                train_df['MthRet_2'] = train_df.iloc[:, 1] if len(train_df.columns) > 1 else 0
        
            # === PREPROCESSING DU TEST SET ===
            # 1. Nettoyage des valeurs extrêmes sur test
            test_to_scale = test_data[numeric_columns].copy()
            for col in numeric_columns:
                test_to_scale[col] = self.feature_engineer._clean_extreme_values(test_to_scale[col])
        
            # 2. Vérification des valeurs problématiques
            if np.any(np.isinf(test_to_scale.values)) or np.any(np.isnan(test_to_scale.values)):
                test_to_scale = test_to_scale.replace([np.inf, -np.inf], np.nan).fillna(0)
                for col in test_to_scale.columns:
                    test_to_scale[col] = np.clip(test_to_scale[col], -100, 100)
        
            # 3. Transform du test avec le scaler ajusté sur train UNIQUEMENT
            test_scaled = scaler.transform(test_to_scale)
            test_df = pd.DataFrame(test_scaled, columns=numeric_columns, index=test_data.index)
        
            # 4. Ajout des colonnes target si nécessaires
            if 'MthRet_1' not in test_df.columns:
                test_df['MthRet_1'] = test_df.iloc[:, 0] if len(test_df.columns) > 0 else 0
            if 'MthRet_2' not in test_df.columns:
                test_df['MthRet_2'] = test_df.iloc[:, 1] if len(test_df.columns) > 1 else 0
        
        except Exception as e:
            logger.error(f"Error in preprocessing for {ticker1}-{ticker2}: {e}")
            return False
    
        # === ÉTAPE 3 : CRÉATION DES SÉQUENCES SÉPARÉMENT ===
        X_train, y_train = self.create_sequences(train_df, self.config['sequence_length'])
        X_test, y_test = self.create_sequences(test_df, self.config['sequence_length'])
    
        # Validation qu'on a suffisamment de séquences
        if len(X_train) < 10 or len(X_test) < 5:
            logger.warning(f"Insufficient sequences after split for {ticker1}-{ticker2}: train={len(X_train)}, test={len(X_test)}")
            return False
    
        logger.info(f"Created sequences: train={len(X_train)}, test={len(X_test)}")
    
        # === ÉTAPE 4 : ENTRAÎNEMENT DE L'ENSEMBLE DE MODÈLES ===
        ensemble_models = []
        r2_scores = []
    
        for i in range(min(self.config['ensemble_models'], 2)):
            try:
                if i == 0:
                    model = LSTMPairsPredictor(
                        input_size=X_train.shape[2],
                        hidden_dim=min(self.config['hidden_dim'], X_train.shape[2] * 2),
                        num_layers=min(self.config['num_layers'], 2),
                        dropout=self.config['dropout_rate']
                    )
                else:
                    model = LSTMPairsPredictor(
                        input_size=X_train.shape[2],
                        hidden_dim=min(self.config['hidden_dim'] // 2, X_train.shape[2]),
                        num_layers=2,
                        dropout=self.config['dropout_rate'] * 1.5
                    )
            
                # Entraînement avec les données correctement splittées
                trained_model, r2_score = self._train_single_model(model, X_train, y_train, X_test, y_test)
                if trained_model is not None:
                    ensemble_models.append(trained_model)
                    r2_scores.append(r2_score)
            except Exception as e:
                logger.warning(f"Error training model {i} for {ticker1}-{ticker2}: {e}")
                continue
    
        # === ÉTAPE 5 : STOCKAGE DE L'ENSEMBLE ENTRAÎNÉ ===
        if ensemble_models:
            avg_r2 = np.mean(r2_scores) if r2_scores else 0
            self.models.append({
                'models': ensemble_models,
                'scaler': scaler,  # Scaler ajusté UNIQUEMENT sur train
                'ticker1': ticker1,
                'ticker2': ticker2,
                'feature_columns': numeric_columns,
                'r2_score': avg_r2
            })
            logger.info(f"Successfully trained {len(ensemble_models)} models for {ticker1}-{ticker2}")
            return True
    
        return False
    
    def _train_single_model(self, model, X_train, y_train, X_test, y_test):
        """
        Entraîne un modèle individuel avec early stopping et optimisations avancées.
        
        Cette méthode implémente un pipeline d'entraînement robuste avec :
        - Gestion automatique des devices (CPU/GPU)
        - Optimisateur AdamW avec weight decay pour régularisation
        - Learning rate scheduling adaptatif
        - Early stopping pour éviter l'overfitting
        - Gradient clipping pour stabilité d'entraînement
        - Gestion d'erreurs exhaustive
        
        Args:
            model: Modèle PyTorch à entraîner
            X_train, y_train: Données d'entraînement
            X_test, y_test: Données de validation
            
        Returns:
            Modèle entraîné ou None en cas d'échec
        """
        try:
            # === CONFIGURATION DEVICE ET MODÈLE ===
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # === CONVERSION EN TENSEURS PYTORCH ===
            X_train_tensor = torch.FloatTensor(X_train).to(device)
            y_train_tensor = torch.FloatTensor(y_train).to(device)
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            y_test_tensor = torch.FloatTensor(y_test).to(device)
            
            # === CONFIGURATION OPTIMISATION ===
            # AdamW : version améliorée d'Adam avec weight decay découplé
            optimizer = optim.AdamW(model.parameters(), 
                                  lr=self.config['learning_rate'], 
                                  weight_decay=0.01)  # Régularisation L2
            
            # Scheduler adaptatif : réduit le learning rate si pas d'amélioration
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           patience=5, 
                                                           factor=0.5)
            
            # === CONFIGURATION EARLY STOPPING ===
            best_loss = float('inf')
            patience_counter = 0
            best_model_state = model.state_dict().copy()  # Sauvegarde du meilleur état
            
            # === BOUCLE D'ENTRAÎNEMENT ===
            for epoch in range(min(self.config['epochs'], 50)):  # Limitation pour éviter l'overfitting
                model.train()  # Mode entraînement (active dropout, batch norm, etc.)
                
                # === ÉTAPE D'ENTRAÎNEMENT ===
                optimizer.zero_grad()  # Reset des gradients
                try:
                    # Passe avant
                    outputs = model(X_train_tensor)
                    
                    # === EXTRACTION DES PRÉDICTIONS SELON LE TYPE DE MODÈLE ===
                    if isinstance(outputs, dict):
                        # Modèles avec sorties structurées (LSTMPairsPredictor)
                        if 'returns' in outputs:
                            preds = outputs['returns']
                        elif 'asset1_return' in outputs and 'asset2_return' in outputs:
                            # Modèles Transformer avec sorties séparées
                            preds = torch.stack([outputs['asset1_return'], outputs['asset2_return']], dim=1)
                        else:
                            # Fallback en cas de structure inattendue
                            preds = torch.zeros_like(y_train_tensor)
                    else:
                        # Modèles avec sortie directe
                        preds = outputs[:, :2] if outputs.shape[1] >= 2 else outputs
                    
                    # Calcul de la loss (Mean Squared Error pour régression)
                    loss = nn.MSELoss()(preds, y_train_tensor)
                    
                    # === VALIDATION DE LA LOSS ===
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning("NaN or Inf loss detected, stopping training")
                        break
                        
                    # Rétropropagation et optimisation
                    loss.backward()
                    # Gradient clipping pour éviter les explosions de gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                except Exception as e:
                    logger.warning(f"Error in training step {epoch}: {e}")
                    break
                
                # === VALIDATION PÉRIODIQUE ===
                if epoch % 5 == 0:  # Validation tous les 5 epochs pour efficacité
                    model.eval()  # Mode évaluation (désactive dropout, etc.)
                    with torch.no_grad():  # Pas de calcul de gradients pour l'évaluation
                        try:
                            # Passe avant sur les données de validation
                            val_outputs = model(X_test_tensor)
                            
                            # Extraction des prédictions (même logique que l'entraînement)
                            if isinstance(val_outputs, dict):
                                if 'returns' in val_outputs:
                                    val_preds = val_outputs['returns']
                                elif 'asset1_return' in val_outputs and 'asset2_return' in val_outputs:
                                    val_preds = torch.stack([val_outputs['asset1_return'], val_outputs['asset2_return']], dim=1)
                                else:
                                    val_preds = torch.zeros_like(y_test_tensor)
                            else:
                                val_preds = val_outputs[:, :2] if val_outputs.shape[1] >= 2 else val_outputs
                            
                            # Calcul de la loss de validation
                            val_loss = nn.MSELoss()(val_preds, y_test_tensor)
                            
                            if torch.isnan(val_loss) or torch.isinf(val_loss):
                                break
                            
                            # Mise à jour du scheduler basée sur la performance
                            scheduler.step(val_loss)
                            
                            # === LOGIQUE EARLY STOPPING ===
                            if val_loss < best_loss:
                                # Amélioration détectée
                                best_loss = val_loss
                                patience_counter = 0
                                best_model_state = model.state_dict().copy()  # Sauvegarde
                            else:
                                # Pas d'amélioration
                                patience_counter += 1
                            
                            # Arrêt si pas d'amélioration pendant trop longtemps
                            if patience_counter >= self.config['early_stopping_patience']:
                                break
                                
                        except Exception as e:
                            logger.warning(f"Error in validation step {epoch}: {e}")
                            break
            
            # === RESTAURATION DU MEILLEUR MODÈLE ===
            model.load_state_dict(best_model_state)
            
            model.eval()
            with torch.no_grad():
                try:
                    final_outputs = model(X_test_tensor)
                
                    if isinstance(final_outputs, dict):
                        if 'returns' in final_outputs:
                            final_preds = final_outputs['returns']
                        elif 'asset1_return' in final_outputs and 'asset2_return' in final_outputs:
                            final_preds = torch.stack([final_outputs['asset1_return'], final_outputs['asset2_return']], dim=1)
                        else:
                            final_preds = torch.zeros_like(y_test_tensor)
                    else:
                        final_preds = final_outputs[:, :2] if final_outputs.shape[1] >= 2 else final_outputs
                
                    # Conversion en numpy pour calcul R²
                    y_true_np = y_test_tensor.cpu().numpy()
                    y_pred_np = final_preds.cpu().numpy()
                
                    # Calcul R² pour chaque actif et moyenne
                    r2_asset1 = r2_score(y_true_np[:, 0], y_pred_np[:, 0])
                    r2_asset2 = r2_score(y_true_np[:, 1], y_pred_np[:, 1]) if y_true_np.shape[1] > 1 else 0
                    avg_r2 = (r2_asset1 + r2_asset2) / 2
                
                    return model, avg_r2
                
                except Exception as e:
                    logger.warning(f"Error calculating R²: {e}")
                    return model, 0
        
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return None, 0
    
    def predict(self, data: pd.DataFrame) -> Dict:
        """
        Génère des prédictions d'ensemble robustes pour les nouvelles données.
        
        Cette méthode combine les prédictions de tous les modèles de l'ensemble
        pour produire une prédiction finale plus robuste et une estimation de confiance.
        Le processus suit le même preprocessing que l'entraînement pour garantir la cohérence.
        
        Args:
            data: DataFrame avec les mêmes features que l'entraînement
            
        Returns:
            Dictionnaire contenant :
            - 'asset1_return': Prédiction de rendement pour le premier actif
            - 'asset2_return': Prédiction de rendement pour le second actif
            - 'confidence': Score de confiance basé sur la variance des prédictions
        """
        # Validation qu'on a des modèles entraînés
        if not self.models:
            return None
        
        all_predictions = []  # Stockage de toutes les prédictions pour moyennage
        
        # === BOUCLE SUR TOUS LES ENSEMBLES DE MODÈLES ===
        for model_info in self.models:
            try:
                # === PRÉPARATION DES DONNÉES (MÊME PIPELINE QUE L'ENTRAÎNEMENT) ===
                # Extraction des features avec les mêmes colonnes que l'entraînement
                feature_data = data[model_info['feature_columns']].fillna(0)
                
                # Application du même normalisateur que l'entraînement
                scaled_features = model_info['scaler'].transform(feature_data)
                scaled_df = pd.DataFrame(scaled_features, columns=model_info['feature_columns'])
                
                # Ajout des colonnes target si nécessaires (cohérence avec entraînement)
                if 'MthRet_1' not in scaled_df.columns:
                    scaled_df['MthRet_1'] = scaled_df.iloc[:, 0] if len(scaled_df.columns) > 0 else 0
                if 'MthRet_2' not in scaled_df.columns:
                    scaled_df['MthRet_2'] = scaled_df.iloc[:, 1] if len(scaled_df.columns) > 1 else 0
                
                # === CRÉATION DES SÉQUENCES POUR PRÉDICTION ===
                X, _ = self.create_sequences(scaled_df, self.config['sequence_length'])
                
                if len(X) == 0:
                    continue  # Pas assez de données pour créer une séquence
                
                # === PRÉDICTIONS AVEC TOUS LES MODÈLES DE CET ENSEMBLE ===
                model_predictions = []
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                for model in model_info['models']:
                    try:
                        model.eval()  # Mode évaluation
                        with torch.no_grad():  # Pas de gradients pour l'inférence
                            # Utilisation de la dernière séquence disponible pour prédiction
                            X_tensor = torch.FloatTensor(X[-1:]).to(device)  # Shape: [1, seq_len, features]
                            outputs = model(X_tensor)
                            
                            # === EXTRACTION DES PRÉDICTIONS ===
                            if isinstance(outputs, dict):
                                if 'returns' in outputs:
                                    pred = outputs['returns'].cpu().numpy()
                                else:
                                    # Fallback si structure inattendue
                                    pred = np.array([[0.0, 0.0]])
                            else:
                                pred = outputs[:, :2].cpu().numpy() if outputs.shape[1] >= 2 else np.array([[0.0, 0.0]])
                            
                            model_predictions.append(pred[0])  # Extraction du premier (et unique) échantillon
                            
                    except Exception as e:
                        logger.warning(f"Error in model prediction: {e}")
                        continue
                
                # === AGRÉGATION DES PRÉDICTIONS DE CET ENSEMBLE ===
                if model_predictions:
                    # Moyenne des prédictions des modèles de cet ensemble
                    ensemble_pred = np.mean(model_predictions, axis=0)
                    all_predictions.append(ensemble_pred)
                    
            except Exception as e:
                logger.warning(f"Error in ensemble prediction: {e}")
                continue
        
        # === AGRÉGATION FINALE ET CALCUL DE CONFIANCE ===
        if all_predictions:
            # Prédiction finale : moyenne de tous les ensembles
            final_prediction = np.mean(all_predictions, axis=0)
            
            # Calcul de confiance basé sur la variance des prédictions
            # Faible variance = haute confiance, forte variance = faible confiance
            variance = np.std(all_predictions, axis=0).mean()
            confidence = max(0.1, 1.0 / (1.0 + variance))  # Score entre 0.1 et 1.0
            
            return {
                'asset1_return': final_prediction[0] if len(final_prediction) > 0 else 0.0,
                'asset2_return': final_prediction[1] if len(final_prediction) > 1 else 0.0,
                'confidence': confidence
            }
        
        return None  # Aucune prédiction possible

class DynamicRiskManager:
    """
    Gestionnaire de risque adaptatif pour le trading quantitatif de paires.
    
    Cette classe implémente un système sophistiqué de gestion des risques qui s'adapte
    dynamiquement aux conditions de marché changeantes. Elle combine plusieurs techniques
    avancées de la finance quantitative :
    
    - Value at Risk (VaR) pour quantifier les pertes potentielles
    - Critère de Kelly pour optimiser la taille des positions
    - Ajustement dynamique des seuils selon la volatilité et la corrélation
    - Historique des performances pour l'apprentissage adaptatif
    
    L'objectif est de maximiser les rendements tout en contrôlant rigoureusement
    l'exposition au risque, particulièrement important en pairs trading où les
    corrélations peuvent se rompre brutalement.
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le gestionnaire de risque avec la configuration spécifiée.
        
        Args:
            config: Dictionnaire contenant les paramètres de risque :
                   - kelly_fraction: Fraction Kelly maximale autorisée (ex: 0.25)
                   - max_position_size: Taille maximale de position (ex: 0.1 = 10% du capital)
                   - base_zscore_entry: Seuil Z-score de base pour l'entrée (ex: 2.0)
                   - base_zscore_exit: Seuil Z-score de base pour la sortie (ex: 0.5)
        """
        self.config = config
        # Historique des positions pour analyse des patterns et performances
        self.position_history = []
        # Historique des PnL pour calcul des métriques de risque
        self.pnl_history = []
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.05) -> float:
        """
        Calcule la Value at Risk (VaR) - estimation de la perte maximale probable.
        
        La VaR est une mesure de risque fondamentale qui quantifie la perte maximale
        attendue sur un horizon temporel donné avec un niveau de confiance spécifié.
        Par exemple, une VaR de 2% à 95% signifie qu'il y a 5% de probabilité
        de perdre plus de 2% sur la période considérée.
        
        Cette implémentation utilise la méthode historique (percentile empirique)
        qui est robuste et ne fait pas d'hypothèses sur la distribution des rendements.
        
        Args:
            returns: Série des rendements historiques de la stratégie
            confidence: Niveau de confiance (0.05 = VaR à 95%, 0.01 = VaR à 99%)
            
        Returns:
            float: VaR exprimée comme fraction positive (ex: 0.02 = 2% de perte max)
        """
        # === VALIDATION DE LA QUALITÉ DES DONNÉES ===
        # Minimum 30 observations pour une estimation VaR fiable
        if len(returns) < 30:
            return 0.02  # VaR conservative par défaut (2%)
        
        # Nettoyage des valeurs manquantes
        returns_clean = returns.dropna()
        
        # Validation post-nettoyage
        if len(returns_clean) < 10:
            return 0.02  # Fallback si trop de données manquantes
        
        # === CALCUL DE LA VAR PAR MÉTHODE HISTORIQUE ===
        # Le percentile donne la valeur en dessous de laquelle se trouvent
        # "confidence * 100" % des observations
        # Comme on veut la perte (valeur négative), on prend la valeur absolue
        var_value = np.percentile(returns_clean, confidence * 100)
        
        # Retourne la VaR comme valeur positive (convention standard)
        return abs(var_value)
    
    def kelly_position_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calcule la taille optimale de position selon le critère de Kelly.
        
        Le critère de Kelly détermine la fraction optimale du capital à risquer
        pour maximiser la croissance logarithmique du portefeuille à long terme.
        C'est une approche mathématiquement rigoureuse qui équilibre l'espérance
        de gain avec la probabilité de ruine.
        
        Formule de Kelly : f* = p - q*(b/a)
        où :
        - f* = fraction Kelly optimale
        - p = probabilité de gain (win_rate)
        - q = probabilité de perte (1 - win_rate)
        - b = gain moyen des trades gagnants
        - a = perte moyenne des trades perdants (valeur absolue)
        
        Args:
            win_rate: Taux de trades gagnants (ex: 0.6 = 60% de réussite)
            avg_win: Gain moyen des trades gagnants (valeur positive)
            avg_loss: Perte moyenne des trades perdants (valeur négative)
            
        Returns:
            float: Taille de position recommandée comme fraction du capital
        """
        # === VALIDATION DES PARAMÈTRES D'ENTRÉE ===
        # Conditions nécessaires pour un calcul Kelly valide
        if avg_loss >= 0 or win_rate <= 0 or avg_win <= 0:
            return 0.01  # Position minimale conservatrice en cas de paramètres invalides
        
        # === CALCUL DE LA FRACTION KELLY ===
        # Application directe de la formule de Kelly
        # win_rate : probabilité de succès (p)
        # (1 - win_rate) : probabilité d'échec (q)
        # avg_win : gain moyen (b)
        # abs(avg_loss) : perte moyenne absolue (a)
        kelly_fraction = win_rate - (1 - win_rate) * (avg_win / abs(avg_loss))
        
        # === CONTRAINTES DE SÉCURITÉ ===
        # Limitation de la fraction Kelly pour éviter les positions excessives
        # La théorie pure de Kelly peut recommander des positions très importantes
        # qui ne sont pas pratiques en réalité (coûts de transaction, slippage, etc.)
        kelly_fraction = max(0, min(kelly_fraction, self.config['kelly_fraction']))
        
        # === APPLICATION DE LA TAILLE MAXIMALE ===
        # Multiplication par la taille maximale autorisée par la politique de risque
        return kelly_fraction * self.config['max_position_size']
    
    def dynamic_threshold_adjustment(self, volatility: float, correlation: float) -> Tuple[float, float]:
        """
        Ajuste dynamiquement les seuils de trading selon les conditions de marché.
        
        Cette méthode est cruciale pour l'adaptabilité de la stratégie aux régimes
        de marché changeants. Elle modifie les seuils d'entrée et de sortie en fonction
        de la volatilité et de la corrélation actuelles, permettant à la stratégie
        de s'adapter automatiquement aux conditions de marché.
        
        Logique d'ajustement :
        - Volatilité élevée → seuils plus élevés (éviter les faux signaux)
        - Corrélation faible → seuils plus élevés (relation moins fiable)
        - Volatilité faible → seuils plus bas (capture des opportunités subtiles)
        - Corrélation forte → seuils standards (relation robuste)
        
        Args:
            volatility: Volatilité actuelle du spread (ex: 0.02 = 2%)
            correlation: Corrélation actuelle entre les actifs (ex: 0.8)
            
        Returns:
            Tuple[float, float]: (seuil_entrée_ajusté, seuil_sortie_ajusté)
        """
        # === RÉCUPÉRATION DES SEUILS DE BASE ===
        # Seuils de référence définis dans la configuration
        base_entry = self.config['base_zscore_entry']  # Ex: 2.0 (2 écarts-types)
        base_exit = self.config['base_zscore_exit']    # Ex: 0.5 (0.5 écart-type)
        
        # === AJUSTEMENT BASÉ SUR LA VOLATILITÉ ===
        # Calcul d'un facteur d'ajustement basé sur la volatilité relative
        # Hypothèse : volatilité "normale" = 2% (0.02)
        vol_adjustment = min(1.5, max(0.5, volatility / 0.02))
        
        # Interprétation :
        # - volatilité = 0.01 (1%) → vol_adjustment = 0.5 → seuils réduits de 50%
        # - volatilité = 0.02 (2%) → vol_adjustment = 1.0 → seuils inchangés  
        # - volatilité = 0.04 (4%) → vol_adjustment = 1.5 → seuils augmentés de 50% (max)
        
        # === AJUSTEMENT BASÉ SUR LA CORRÉLATION ===
        # Plus la corrélation est forte, plus la relation est fiable
        corr_adjustment = min(1.2, max(0.8, abs(correlation)))
        
        # Interprétation :
        # - |correlation| = 0.5 → corr_adjustment = 0.8 → seuils réduits (relation faible)
        # - |correlation| = 0.8 → corr_adjustment = 0.8 → seuils réduits légèrement
        # - |correlation| = 0.9 → corr_adjustment = 0.9 → seuils quasi-normaux
        # - |correlation| = 1.0 → corr_adjustment = 1.0 → seuils inchangés
        
        # === APPLICATION DES AJUSTEMENTS ===
        # Seuil d'entrée : sensible à la fois à la volatilité ET à la corrélation
        # Plus de volatilité OU moins de corrélation → seuil plus élevé (plus conservateur)
        entry_threshold = base_entry * vol_adjustment * corr_adjustment
        
        # Seuil de sortie : sensible principalement à la volatilité
        # En période de haute volatilité, on sort plus tôt pour sécuriser les gains
        exit_threshold = base_exit * vol_adjustment
        
        return entry_threshold, exit_threshold

class DeepLearningPairsTrader:
    """
    Système de trading de paires utilisant le Deep Learning et l'intelligence artificielle.
    
    Cette classe représente l'orchestrateur principal d'une stratégie de pairs trading
    moderne combinant plusieurs technologies avancées :
    
    - Machine Learning pour la sélection automatique des paires
    - Deep Learning (LSTM/Transformer) pour les prédictions
    - Gestion dynamique des risques avec ajustements adaptatifs
    - Feature engineering sophistiqué avec indicateurs techniques avancés
    - Ensemble learning pour la robustesse des prédictions
    
    Le système automatise entièrement le processus de trading :
    1. Sélection intelligente des paires les plus prometteuses
    2. Entraînement d'ensembles de modèles pour chaque paire
    3. Génération de signaux de trading basés sur les prédictions ML
    4. Calcul optimal des tailles de position avec gestion des risques
    
    Cette approche représente l'état de l'art en trading quantitatif moderne.
    """
    
    def __init__(self, config: Dict = DEEP_LEARNING_CONFIG):
        """
        Initialise le système de trading avec tous ses composants.
        
        Args:
            config: Configuration contenant tous les hyperparamètres :
                   - Paramètres de sélection des paires (corrélation, cointégration)
                   - Hyperparamètres des modèles ML (architecture, entraînement)
                   - Paramètres de gestion des risques (VaR, Kelly, seuils)
                   - Contraintes opérationnelles (tailles max, nombre de paires)
        """
        self.config = config
        # Ingénieur de features pour transformation des données brutes
        self.feature_engineer = AdvancedFeatureEngineer()
        # Gestionnaire de risque adaptatif
        self.risk_manager = DynamicRiskManager(config)
        # Stockage des modèles entraînés pour chaque paire
        self.ensemble_models = {}
        
    def select_pairs_ml(self, data: pd.DataFrame) -> List[Dict]:
        """
        Sélection intelligente de paires d'actifs optimales pour le trading.
        
        Cette méthode implémente un processus de sélection sophistiqué qui combine
        des critères statistiques traditionnels (corrélation, cointégration) avec
        des métriques ML avancées pour identifier les paires les plus prometteuses.
        
        Le processus de sélection évalue :
        1. Corrélation historique entre les actifs
        2. Cointégration (relation d'équilibre long terme)
        3. Stationnarité du spread pour la prévisibilité
        4. Stabilité de la relation dans le temps
        5. Qualité des features ML
        
        Args:
            data: DataFrame contenant les données de tous les actifs candidats
                  Colonnes attendues : ['Ticker', 'date', 'MthRet', ...]
        
        Returns:
            Liste de dictionnaires contenant les meilleures paires avec métadonnées :
            - ticker1, ticker2 : identifiants des actifs
            - correlation : coefficient de corrélation
            - coint_pvalue : p-value du test de cointégration
            - ml_score : score de qualité ML composite
            - data : données enrichies de la paire
        """
        logger.info("Selecting pairs using ML approach...")
        
        # === PRÉPARATION ET VALIDATION DES DONNÉES ===
        # Extraction de tous les tickers uniques disponibles
        tickers = data['Ticker'].unique()
        
        if len(tickers) < 2:
            logger.warning("Less than 2 tickers found")
            return []
        
        # === OPTIMISATION PERFORMANCE : LIMITATION DU NOMBRE DE TICKERS ===
        # Pour éviter les problèmes de mémoire et de temps de calcul
        if len(tickers) > 100:
            # Sélection des tickers avec le plus de données historiques
            # (plus de données = analyses plus fiables)
            ticker_counts = data['Ticker'].value_counts()
            tickers = ticker_counts.head(100).index.tolist()
        
        valid_pairs = []
        
        # === LIMITATION DU NOMBRE DE COMBINAISONS TESTÉES ===
        # Protection contre l'explosion combinatoire (n*(n-1)/2 paires possibles)
        max_combinations = min(100, len(tickers) * (len(tickers) - 1) // 2)
        tested_combinations = 0
        
        # === BOUCLE PRINCIPALE DE SÉLECTION DES PAIRES ===
        for i, ticker1 in enumerate(tickers):
            for ticker2 in tickers[i+1:]:  # Évite les doublons (A,B) vs (B,A)
                if tested_combinations >= max_combinations:
                    break
                
                tested_combinations += 1
                
                try:
                    # === EXTRACTION ET VALIDATION DES DONNÉES INDIVIDUELLES ===
                    data1 = data[data['Ticker'] == ticker1].copy()
                    data2 = data[data['Ticker'] == ticker2].copy()
                    
                    # Vérification de suffisamment d'observations pour analyse fiable
                    if len(data1) < self.config['min_observations'] or len(data2) < self.config['min_observations']:
                        continue
                    
                    # === SPLIT TEMPOREL AVANT TOUT PREPROCESSING ===
                    # Tri chronologique
                    data1 = data1.sort_values('date').reset_index(drop=True)
                    data2 = data2.sort_values('date').reset_index(drop=True)
                
                    # Split 80/20 sur données brutes
                    split_idx_1 = int(len(data1) * 0.8)
                    split_idx_2 = int(len(data2) * 0.8)
                
                    train_data1 = data1.iloc[:split_idx_1].copy()
                    train_data2 = data2.iloc[:split_idx_2].copy()
                
                    
                    # === FEATURE ENGINEERING POUR CHAQUE ACTIF ===
                    # Application des transformations techniques avancées
                    train_data1 = self.feature_engineer.create_technical_features(train_data1)
                    train_data2 = self.feature_engineer.create_technical_features(train_data2)
                    
                    data1 = self.feature_engineer.create_technical_features(data1)
                    data2 = self.feature_engineer.create_technical_features(data2)
                    
                    
                    # === VALIDATION DE QUALITÉ DES DONNÉES ===
                    # Vérification absence de valeurs infinies (problématiques pour ML)
                    if (np.any(np.isinf(train_data1.select_dtypes(include=[np.number]).values)) or
                        np.any(np.isinf(train_data2.select_dtypes(include=[np.number]).values))):
                        logger.warning(f"Infinite values found in {ticker1} or {ticker2}, skipping")
                        continue
                    
                    # === GÉNÉRATION DES FEATURES DE PAIRE ===
                    # Création des métriques relationnelles entre les deux actifs
                    pair_features = self.feature_engineer.engineer_pair_features(train_data1, train_data2)
                    real_features = self.feature_engineer.engineer_pair_features(data1, data2)
                    
                    
                    if len(pair_features) < self.config['min_observations']:
                        continue
                    
                    # === VALIDATION FINALE DES FEATURES DE PAIRE ===
                    numeric_cols = pair_features.select_dtypes(include=[np.number]).columns
                    has_extreme_values = False
                    
                    for col in numeric_cols:
                        # Détection de valeurs aberrantes qui peuvent déstabiliser les modèles
                        if (np.any(np.isinf(pair_features[col])) or 
                            np.any(np.abs(pair_features[col]) > 1e6)):
                            has_extreme_values = True
                            break
                    
                    if has_extreme_values:
                        logger.warning(f"Extreme values detected in pair features for {ticker1}-{ticker2}, skipping")
                        continue
                    
                    # === CRITÈRE 1 : VALIDATION STATISTIQUE - CORRÉLATION ===
                    # Calcul de la corrélation entre les actifs 6m et non 12 car trop restrictif
                    if 'correlation_6m' in pair_features.columns:
                        correlation = pair_features['correlation_6m'].mean()
                    else:
                        # Fallback : corrélation directe des rendements
                        correlation = pair_features['MthRet_1'].corr(pair_features['MthRet_2'])
                        
                    # Filtre : corrélation minimale requise pour une relation significative
                    if abs(correlation) < self.config['min_correlation']:
                        continue
                    
                    # === CRITÈRE 2 : TEST DE COINTÉGRATION ===
                    # Vérification de l'existence d'une relation d'équilibre long terme
                    try:
                        # Test d'Engle-Granger pour cointégration
                        _, coint_pvalue, _ = coint(pair_features['MthRet_1'].dropna(), 
                                                 pair_features['MthRet_2'].dropna())
                        # p-value faible = forte évidence de cointégration
                        if coint_pvalue > self.config['min_cointegration_pvalue']:
                            continue
                            
                    except Exception as e:
                        logger.warning(f"Cointegration test failed for {ticker1}-{ticker2}: {e}")
                        continue
                    
                    # === CRITÈRE 3 : SCORE ML AVANCÉ ===
                    # Évaluation de la qualité de la paire pour les algorithmes ML
                    spread_stationarity = self._test_spread_ml_features(pair_features)
                    
                    # Seuil réduit pour être moins restrictif et explorer plus de paires
                    if spread_stationarity > 0.5:
                        valid_pairs.append({
                            'ticker1': ticker1,
                            'ticker2': ticker2,
                            'correlation': correlation,
                            'coint_pvalue': coint_pvalue,
                            'ml_score': spread_stationarity,
                            'data': real_features
                        })
                    
                    # === LIMITATION DU NOMBRE DE PAIRES RETENUES ===
                    if len(valid_pairs) >= self.config['max_pairs']:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing pair {ticker1}-{ticker2}: {e}")
                    continue
            
            if len(valid_pairs) >= self.config['max_pairs']:
                break
        
        # === CLASSIFICATION ET SÉLECTION FINALE ===
        # Tri par score ML décroissant (meilleures paires en premier)
        valid_pairs.sort(key=lambda x: x['ml_score'], reverse=True)
        selected_pairs = valid_pairs[:self.config['max_pairs']]
        
        logger.info(f"Selected {len(selected_pairs)} pairs from {tested_combinations} tested combinations")
        return selected_pairs
    
    def _test_spread_ml_features(self, pair_data: pd.DataFrame) -> float:
        """
        Calcule un score de qualité ML composite pour une paire d'actifs.
        
        Cette méthode évalue la "qualité ML" d'une paire en analysant plusieurs
        dimensions critiques pour le succès des algorithmes de machine learning :
        
        1. Stationnarité : Le spread doit revenir vers sa moyenne (mean reversion)
        2. Prévisibilité : Le spread doit avoir une autocorrélation modérée
        3. Stabilité : La relation entre actifs doit être consistante dans le temps
        
        Ces critères sont essentiels car ils déterminent si les patterns historiques
        peuvent être appris efficacement par les modèles ML et extrapolés au futur.
        
        Args:
            pair_data: DataFrame contenant les features de la paire
        
        Returns:
            float: Score composite entre 0 et 1 (1 = qualité ML maximale)
        """
        try:
            # === VALIDATION DES DONNÉES REQUISES ===
            if 'return_spread' not in pair_data.columns:
                return 0.0
                
            spread = pair_data['return_spread'].dropna()
            
            if len(spread) < 12:  # Minimum pour analyses statistiques fiables
                return 0.0
            
            # === MÉTRIQUE 1 : STATIONNARITÉ (40% du score) ===
            # Test d'Augmented Dickey-Fuller pour stationnarité
            try:
                adf_stat, adf_pvalue, *_ = adfuller(spread, maxlag=min(12, len(spread)//4))
                # p-value faible = forte stationnarité = bon pour mean reversion
                stationarity_score = max(0, 1.0 - adf_pvalue)
            except:
                stationarity_score = 0.0
            
            # === MÉTRIQUE 2 : PRÉVISIBILITÉ (30% du score) ===
            # Autocorrélation lag-1 : mesure la persistance des mouvements
            try:
                autocorr = abs(spread.autocorr(lag=1))
                if np.isnan(autocorr):
                    autocorr = 0
                # Normalisation : autocorrélation modérée (0.25) = score max
                predictability_score = min(autocorr, 0.5) * 2  # Mapping vers [0,1]
            except:
                predictability_score = 0.0
            
            # === MÉTRIQUE 3 : STABILITÉ DE LA RELATION (30% du score) ===
            # Variance de la corrélation glissante : faible variance = relation stable
            try:
                if 'correlation_6m' in pair_data.columns:
                    rolling_corr = pair_data['correlation_6m'].dropna()
                    if len(rolling_corr) > 0 and rolling_corr.std() > 0:
                        # Faible std de corrélation = relation stable = score élevé
                        correlation_stability = max(0, 1.0 - rolling_corr.std())
                    else:
                        correlation_stability = 0.5  # Score neutre
                else:
                    correlation_stability = 0.5
            except:
                correlation_stability = 0.5
            
            # === CALCUL DU SCORE COMPOSITE ===
            # Pondération réfléchie selon l'importance pour le ML
            composite_score = (stationarity_score * 0.4 +      # Crucial pour mean reversion
                             predictability_score * 0.3 +      # Important pour patterns
                             correlation_stability * 0.3)      # Important pour robustesse
            
            # Assurance que le score reste dans [0,1]
            return min(1.0, max(0.0, composite_score))
            
        except Exception as e:
            logger.warning(f"Error in ML features test: {e}")
            return 0.0
    
    def train_models(self, pairs_data: List[Dict]) -> bool:
        """
        Entraîne les ensembles de modèles ML pour toutes les paires sélectionnées.
        
        Cette méthode orchestre l'entraînement complet des modèles de deep learning
        pour chaque paire. Elle utilise l'ensemble learning pour maximiser la robustesse
        et la performance des prédictions.
        
        Pour chaque paire :
        1. Création d'un ensemble de modèles avec architectures variées
        2. Entraînement avec validation croisée temporelle
        3. Sélection des meilleurs modèles par early stopping
        4. Stockage pour utilisation en production
        
        Args:
            pairs_data: Liste des paires sélectionnées avec leurs données
        
        Returns:
            bool: True si au moins un modèle a été entraîné avec succès
        """
        logger.info("Training ensemble models for all pairs...")
        
        success_count = 0
        
        # === BOUCLE D'ENTRAÎNEMENT POUR CHAQUE PAIRE ===
        for pair_info in pairs_data:
            try:
                # Extraction des métadonnées de la paire
                ticker1 = pair_info['ticker1']
                ticker2 = pair_info['ticker2']
                pair_data = pair_info['data']
                
                # === CRÉATION ET ENTRAÎNEMENT DE L'ENSEMBLE ===
                ensemble = EnsemblePairsModel(self.config)
                
                # Tentative d'entraînement avec gestion d'erreur robuste
                if ensemble.train_ensemble(pair_data, ticker1, ticker2):
                    # Stockage du modèle entraîné avec clé unique
                    self.ensemble_models[f"{ticker1}_{ticker2}"] = ensemble
                    success_count += 1
                    logger.info(f"Successfully trained ensemble for {ticker1}-{ticker2}")
                else:
                    logger.warning(f"Failed to train ensemble for {ticker1}-{ticker2}")
                    
            except Exception as e:
                # Gestion robuste des erreurs pour éviter l'arrêt complet
                logger.error(f"Error training models for {pair_info.get('ticker1', 'unknown')}-{pair_info.get('ticker2', 'unknown')}: {e}")
                continue
        
        logger.info(f"Successfully trained {success_count}/{len(pairs_data)} pair models")
        return success_count > 0
    
    def generate_trading_signals(self, pair_key: str, current_data: pd.DataFrame) -> Dict:
        """
        Génère des signaux de trading sophistiqués basés sur les prédictions ML.
        
        Cette méthode est le cœur du système de trading. Elle combine :
        1. Prédictions des modèles de deep learning
        2. Calcul dynamique des seuils d'entrée/sortie
        3. Analyse du z-score du spread prédit
        4. Évaluation de la confiance des prédictions
        
        Le processus de génération de signal :
        - Utilise l'ensemble de modèles pour prédire les rendements futurs
        - Calcule le spread prédit et son z-score
        - Ajuste les seuils selon la volatilité et corrélation actuelles
        - Génère le signal final avec niveau de confiance
        
        Args:
            pair_key: Identifiant de la paire (format "TICKER1_TICKER2")
            current_data: Données actuelles de la paire avec features
        
        Returns:
            Dictionnaire contenant :
            - position : Signal (-1=short, 0=neutre, 1=long)
            - z_score : Z-score du spread prédit
            - predicted_spread : Valeur du spread prédit
            - confidence : Niveau de confiance [0,1]
            - entry_threshold, exit_threshold : Seuils adaptatifs
            - reason : Explication textuelle du signal
        """
        # === VALIDATION DE LA DISPONIBILITÉ DU MODÈLE ===
        if pair_key not in self.ensemble_models:
            return {'position': 0, 'confidence': 0, 'reason': 'No model available'}
        
        try:
            ensemble = self.ensemble_models[pair_key]
            
            # === ÉTAPE 1 : GÉNÉRATION DES PRÉDICTIONS ML ===
            predictions = ensemble.predict(current_data)
            
            if predictions is None:
                return {'position': 0, 'confidence': 0, 'reason': 'Prediction failed'}
            
            # === ÉTAPE 2 : CALCUL DU SPREAD PRÉDIT ===
            # Différence entre les rendements prédits des deux actifs
            predicted_spread = predictions['asset1_return'] - predictions['asset2_return']
            
            # === ÉTAPE 3 : CALCUL DU Z-SCORE DYNAMIQUE ===
            # Normalisation du spread prédit par rapport à l'historique récent
            if len(current_data) >= self.config['lookback_window'] and 'return_spread' in current_data.columns:
                # Utilisation d'une fenêtre glissante pour adaptation aux conditions récentes
                recent_spreads = current_data['return_spread'].tail(self.config['lookback_window']).dropna()
                
                if len(recent_spreads) > 0:
                    spread_mean = recent_spreads.mean()
                    spread_std = recent_spreads.std()
                    
                    if spread_std > 0:
                        # Z-score = (valeur - moyenne) / écart-type
                        z_score = (predicted_spread - spread_mean) / spread_std
                    else:
                        z_score = 0  # Aucune variation = pas de signal
                else:
                    z_score = 0
            else:
                z_score = 0  # Données insuffisantes
            
            # === ÉTAPE 4 : CALCUL DES SEUILS ADAPTATIFS ===
            # Estimation de la volatilité récente du spread
            if 'return_spread' in current_data.columns:
                recent_vol = current_data['return_spread'].tail(6).std()
                if np.isnan(recent_vol) or recent_vol <= 0:
                    recent_vol = 0.02  # Volatilité par défaut (2%)
            else:
                recent_vol = 0.02
                
            # Estimation de la corrélation récente
            if 'correlation_6m' in current_data.columns:
                recent_corr = current_data['correlation_6m'].iloc[-1]
                if np.isnan(recent_corr):
                    recent_corr = 0.7  # Corrélation par défaut
            else:
                recent_corr = 0.7
            
            # Application de l'ajustement dynamique des seuils
            entry_threshold, exit_threshold = self.risk_manager.dynamic_threshold_adjustment(
                recent_vol, recent_corr
            )
            
            # === ÉTAPE 5 : GÉNÉRATION DU SIGNAL FINAL ===
            position = 0
            reason = ""
            
            if z_score > entry_threshold:
                # Spread trop élevé → short spread (vendre actif1, acheter actif2)
                position = -1
                reason = f"Short spread (z-score: {z_score:.2f})"
            elif z_score < -entry_threshold:
                # Spread trop bas → long spread (acheter actif1, vendre actif2)
                position = 1
                reason = f"Long spread (z-score: {z_score:.2f})"
            elif abs(z_score) < exit_threshold:
                # Spread proche de la moyenne → sortir de position
                position = 0
                reason = f"Exit position (z-score: {z_score:.2f})"
            
            # === RETOUR DU SIGNAL STRUCTURÉ ===
            return {
                'position': position,
                'z_score': z_score,
                'predicted_spread': predicted_spread,
                'confidence': predictions['confidence'],
                'entry_threshold': entry_threshold,
                'exit_threshold': exit_threshold,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Error generating signals for {pair_key}: {e}")
            return {'position': 0, 'confidence': 0, 'reason': f'Error: {str(e)}'}
    
    def calculate_position_size(self, pair_key: str, signal_info: Dict, historical_returns: pd.Series) -> float:
        """
        Calcule la taille optimale de position en combinant plusieurs approches de gestion des risques.
        
        Cette méthode sophistiquée détermine la taille de position en intégrant :
        1. Direction du signal de base (long/short/neutre)
        2. Ajustement par la confiance ML (prédictions incertaines = positions réduites)
        3. Critère de Kelly pour optimisation mathématique
        4. Ajustement VaR pour contrôle du risque de perte
        5. Contraintes réglementaires (taille maximale)
        
        L'objectif est de maximiser le rendement ajusté du risque en adaptant
        automatiquement l'exposition selon les conditions et la qualité des signaux.
        
        Args:
            pair_key: Identifiant de la paire
            signal_info: Informations du signal (position, confiance, etc.)
            historical_returns: Historique des rendements pour calculs statistiques
        
        Returns:
            float: Taille de position signée (-1 à +1, où signe = direction)
        """
        try:
            # === ÉTAPE 1 : DIRECTION DE BASE ===
            base_position = signal_info['position']
            
            # Si aucun signal, aucune position
            if base_position == 0:
                return 0.0
            
            # === ÉTAPE 2 : AJUSTEMENT PAR LA CONFIANCE ML ===
            # Les prédictions plus confiantes justifient des positions plus importantes
            confidence_multiplier = max(0.1, min(1.0, signal_info.get('confidence', 0.5)))
            
            # === ÉTAPE 3 : CALCUL DE LA TAILLE KELLY ===
            # Optimisation mathématique basée sur l'historique des performances
            if len(historical_returns) > 10:  # Minimum pour statistiques fiables
                historical_returns_clean = historical_returns.dropna()
                winning_trades = historical_returns_clean[historical_returns_clean > 0]
                losing_trades = historical_returns_clean[historical_returns_clean < 0]
                
                if len(winning_trades) > 0 and len(losing_trades) > 0:
                    # Calcul des paramètres Kelly
                    win_rate = len(winning_trades) / len(historical_returns_clean)
                    avg_win = winning_trades.mean()
                    avg_loss = losing_trades.mean()
                    
                    # Application du critère de Kelly
                    kelly_size = self.risk_manager.kelly_position_size(win_rate, avg_win, avg_loss)
                else:
                    # Fallback si pas assez de données segmentées
                    kelly_size = self.config['max_position_size'] * 0.5
            else:
                # Position conservatrice si historique insuffisant
                kelly_size = self.config['max_position_size'] * 0.5
            
            # === ÉTAPE 4 : AJUSTEMENT VaR (VALUE AT RISK) ===
            # Réduction de la position si le risque historique est élevé
            var = self.risk_manager.calculate_var(historical_returns)
            
            # Si VaR > 2%, réduction proportionnelle de la position
            var_multiplier = min(1.0, 0.02 / abs(var)) if var != 0 else 1.0
            
            # === ÉTAPE 5 : CALCUL DE LA TAILLE FINALE ===
            # Combinaison de tous les facteurs d'ajustement
            final_size = kelly_size * confidence_multiplier * var_multiplier
            
            # === ÉTAPE 6 : APPLICATION DES CONTRAINTES ===
            # Respect de la taille maximale réglementaire
            final_size = min(final_size, self.config['max_position_size'])
            
            # === ÉTAPE 7 : APPLICATION DE LA DIRECTION ===
            # Le signe final correspond à la direction du signal (long/short)
            return np.sign(base_position) * final_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

class AdvancedBacktester:
    """
    Système de backtesting avancé pour stratégies de pairs trading avec deep learning.
    
    Cette classe implémente un moteur de simulation sophistiqué qui reproduit fidèlement
    les conditions de trading réelles pour évaluer la performance historique des stratégies ML.
    
    Le backtester simule :
    - L'exécution séquentielle des trades selon les signaux ML
    - La gestion dynamique des positions avec sizing optimal
    - Le calcul précis des rendements et des coûts de transaction
    - L'évolution temporelle des positions et des PnL
    
    Métriques calculées :
    - Performance absolue et relative (rendements, ratio de Sharpe)
    - Qualité des signaux (taux de réussite, rendement moyen par trade)
    - Gestion des risques (drawdown maximum, volatilité)
    - Analyse détaillée trade par trade
    
    Cette approche rigoureuse est essentielle pour valider la robustesse des modèles ML
    avant déploiement en conditions réelles et identifier les biais potentiels.
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le backtester avec les paramètres de simulation.
        
        Args:
            config: Configuration contenant :
                   - Paramètres de coûts de transaction
                   - Contraintes de positions (tailles min/max)
                   - Fenêtres d'analyse pour métriques
                   - Critères de validation des trades
        """
        self.config = config
    
    def run_backtest(self, trader: DeepLearningPairsTrader, pairs_data: List[Dict]) -> Dict:
        """
        Exécute un backtest complet sur toutes les paires avec simulation réaliste.
        
        Cette méthode constitue le cœur du système de backtesting. Elle simule
        l'exécution de la stratégie de trading de manière séquentielle, en respectant
        les contraintes temporelles et informationnelles réelles :
        
        Processus de simulation :
        1. Pour chaque paire, progression chronologique des données
        2. À chaque timestep, génération de signaux avec données disponibles uniquement
        3. Calcul dynamique des tailles de position basé sur l'historique
        4. Simulation de l'exécution des trades et calcul des PnL
        5. Gestion des entrées/sorties selon les signaux ML
        
        Cette approche évite le look-ahead bias (utilisation d'informations futures)
        et reproduit fidèlement les conditions de trading réelles.
        
        Args:
            trader: Instance du trader ML configuré avec modèles entraînés
            pairs_data: Liste des paires avec leurs données historiques
        
        Returns:
            Dictionnaire contenant toutes les métriques de performance :
            - Statistiques globales (nombre de trades, taux de réussite)
            - Métriques de rendement (total, moyen, Sharpe ratio)
            - Métriques de risque (drawdown maximum)
            - DataFrame détaillé de tous les trades
        """
        logger.info("Running advanced backtest...")
        
        all_trades = []
        all_r2_scores = []# Stockage de tous les trades exécutés
        
        # === BOUCLE PRINCIPALE : SIMULATION POUR CHAQUE PAIRE ===
        for pair_info in pairs_data:
            try:
                # === PRÉPARATION DE LA PAIRE ===
                pair_key = f"{pair_info['ticker1']}_{pair_info['ticker2']}"
                
                # Vérification que le modèle ML est disponible pour cette paire
                if pair_key not in trader.ensemble_models:
                    continue
                
                model_info = None
                for model in trader.ensemble_models[pair_key].models:
                    if 'r2_score' in model:
                        all_r2_scores.append(model['r2_score'])
                        break
                
                # Préparation des données avec tri chronologique (crucial pour simulation)
                pair_data = pair_info['data'].copy()
                if 'date' in pair_data.columns:
                    pair_data = pair_data.sort_values('date')
                
                # === INITIALISATION DES VARIABLES DE TRADING ===
                position = 0        # Position actuelle (0=neutre, +/-=long/short)
                entry_date = None   # Date d'entrée en position
                trade_returns = []  # Historique des rendements pour chaque période
                position_history = [] # Historique des tailles de position
                
                # === SIMULATION TEMPORELLE SÉQUENTIELLE ===
                # Début après avoir suffisamment de données pour les features et séquences
                min_start_idx = max(trader.config['sequence_length'], 12)
                
                for i in range(min_start_idx, len(pair_data)):
                    try:
                        # === ÉTAPE 1 : PRÉPARATION DES DONNÉES DISPONIBLES ===
                        # CRUCIAL : Utiliser uniquement les données jusqu'au timestep i
                        # (évite le look-ahead bias)
                        current_data = pair_data.iloc[:i+1]
                        
                        # === ÉTAPE 2 : GÉNÉRATION DU SIGNAL ML ===
                        # Le trader génère un signal basé uniquement sur les données historiques
                        signal_info = trader.generate_trading_signals(pair_key, current_data)
                        
                        # === ÉTAPE 3 : CALCUL DE LA TAILLE DE POSITION ===
                        # Utilisation de l'historique des rendements (maximum 50 derniers)
                        historical_rets = pd.Series(trade_returns[-50:]) if trade_returns else pd.Series([])
                        position_size = trader.calculate_position_size(pair_key, signal_info, historical_rets)
                        
                        # === ÉTAPE 4 : CALCUL DU RENDEMENT DE LA PÉRIODE ===
                        # Extraction du rendement du spread pour cette période
                        if 'return_spread' in current_data.columns:
                            current_spread_return = current_data['return_spread'].iloc[-1]
                            if np.isnan(current_spread_return):
                                current_spread_return = 0
                        else:
                            current_spread_return = 0
                        
                        # === ÉTAPE 5 : GESTION DES POSITIONS EXISTANTES ===
                        if position != 0:
                            # Position ouverte : calcul du PnL de la période
                            # PnL = position_size × rendement_spread
                            trade_return = position * current_spread_return
                            trade_returns.append(trade_return)
                            position_history.append(position)
                            
                            # === LOGIQUE DE SORTIE DE POSITION ===
                            # Sortie si : signal neutre OU changement de direction du signal
                            if (signal_info['position'] == 0 or 
                                np.sign(signal_info['position']) != np.sign(position)):
                                
                                # Clôture de la position et enregistrement du trade
                                if entry_date is not None:
                                    # Calcul du rendement total du trade
                                    total_return = sum(trade_returns[-len(position_history):]) if position_history else 0
                                    
                                    # Enregistrement du trade complet
                                    all_trades.append({
                                        'pair': pair_key,
                                        'entry_date': entry_date,
                                        'exit_date': current_data['date'].iloc[-1] if 'date' in current_data.columns else i,
                                        'position_size': abs(position),
                                        'total_return': total_return,
                                        'duration': len(position_history)  # Durée en périodes
                                    })
                                
                                # Réinitialisation pour nouvelle position
                                position = 0
                                entry_date = None
                                position_history = []
                        
                        # === ÉTAPE 6 : OUVERTURE DE NOUVELLE POSITION ===
                        if position == 0 and signal_info['position'] != 0:
                            # Entrée en nouvelle position selon le signal ML
                            position = position_size
                            entry_date = current_data['date'].iloc[-1] if 'date' in current_data.columns else i
                            position_history = []  # Nouveau cycle de position
                        
                    except Exception as e:
                        logger.warning(f"Error in backtest step {i} for {pair_key}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error in backtest for pair {pair_info.get('ticker1', 'unknown')}: {e}")
                continue
        
        # === CONSOLIDATION ET CALCUL DES MÉTRIQUES FINALES ===
        if all_trades:
            # Conversion en DataFrame pour analyses avancées
            trades_df = pd.DataFrame(all_trades)
            
            # === MÉTRIQUES DE PERFORMANCE GLOBALES ===
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['total_return'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # === MÉTRIQUES DE RENDEMENT ===
            avg_return = trades_df['total_return'].mean()
            total_return = trades_df['total_return'].sum()
            
            # === MÉTRIQUES DE QUALITÉ AJUSTÉES DU RISQUE ===
            sharpe_ratio = self._calculate_sharpe(trades_df['total_return'])
            max_drawdown = self._calculate_max_drawdown(trades_df['total_return'].cumsum())
            
            
            #Calcul R² moyen
            
            avg_r2 = np.mean(all_r2_scores) if all_r2_scores else 0
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'avg_return_per_trade': avg_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_r2_score': avg_r2,
                'trades_df': trades_df
            }
        else:
            # Cas où aucun trade n'a été généré
            logger.warning("No trades generated in backtest")
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'avg_return_per_trade': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_r2_score': 0,
                'trades_df': pd.DataFrame()
            }
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """
        Calcule le ratio de Sharpe - mesure de performance ajustée du risque.
        
        Le ratio de Sharpe est une métrique fondamentale en finance qui mesure
        l'excès de rendement par unité de risque pris. C'est l'une des mesures
        les plus importantes pour évaluer la qualité d'une stratégie de trading.
        
        Formule : Sharpe = (Rendement_moyen - Taux_sans_risque) / Volatilité ATTENTION PAS AJOUTE A faire dans la version finale
        
        Dans cette implémentation simplifiée, le taux sans risque est considéré
        comme nul (contexte de rendements relatifs en pairs trading).
        
        Le facteur √12 annualise le ratio en supposant des données mensuelles.
        
        
        Args:
            returns: Série des rendements des trades
        
        Returns:
            float: Ratio de Sharpe annualisé
        """
        # Validation des données d'entrée
        if len(returns) == 0 or returns.std() == 0:
            return 0  # Pas de données ou volatilité nulle
        
        # Calcul du ratio de Sharpe avec annualisation
        # √12 suppose des données mensuelles → annualisation
        return returns.mean() / returns.std() * np.sqrt(12)
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """
        Calcule le drawdown maximum - mesure du risque de perte maximale.
        
        Le drawdown maximum représente la perte maximale subie entre un pic
        historique et le creux suivant. C'est une mesure cruciale du risque
        car elle quantifie la perte la plus importante qu'un investisseur
        aurait pu subir en suivant la stratégie.
        
        Le calcul suit cette logique :
        1. Pour chaque point, calculer le maximum historique jusqu'à ce point
        2. Calculer l'écart (négatif) entre la valeur actuelle et ce maximum
        3. Le drawdown maximum est le plus grand de ces écarts négatifs
        
        Cette métrique est particulièrement importante pour :
        - Évaluer la tolérance au risque requise
        - Dimensionner le capital nécessaire
        - Comparer les stratégies sur leur profil de risque
        
        Args:
            cumulative_returns: Série des rendements cumulés
        
        Returns:
            float: Drawdown maximum (valeur négative, ex: -0.15 = -15%)
        """
        # Validation des données d'entrée
        if len(cumulative_returns) == 0:
            return 0
        
        # === ÉTAPE 1 : CALCUL DU MAXIMUM GLISSANT ===
        # Pour chaque point, le maximum historique jusqu'à ce point
        running_max = cumulative_returns.expanding().max()
        
        # === ÉTAPE 2 : CALCUL DES DRAWDOWNS ===
        # Différence entre valeur actuelle et maximum historique
        # Valeurs négatives = périodes de drawdown
        drawdown = cumulative_returns - running_max
        
        # === ÉTAPE 3 : IDENTIFICATION DU DRAWDOWN MAXIMUM ===
        # La valeur la plus négative = pire perte historique
        return drawdown.min()

class AdvancedVisualization:
    """Visualisations avancées pour l'analyse"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
    def plot_model_performance(self, backtest_results: Dict):
        """Graphiques de performance du modèle"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            if backtest_results['total_trades'] > 0:
                trades_df = backtest_results['trades_df']
                
                # Préparation des données temporelles
                if 'exit_date' in trades_df.columns:
                    # Conversion des dates en datetime si nécessaire
                    if not pd.api.types.is_datetime64_any_dtype(trades_df['exit_date']):
                        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'], errors='coerce')
                    
                    # Suppression des lignes avec des dates invalides
                    trades_df = trades_df.dropna(subset=['exit_date'])
                    
                    if len(trades_df) > 0:
                        # Tri par date de sortie
                        trades_df = trades_df.sort_values('exit_date').reset_index(drop=True)
                        
                        # Extraction des années pour l'axe X
                        trade_years = trades_df['exit_date'].dt.year
                        
                        # Création d'un index temporel pour les graphiques
                        time_index = trades_df['exit_date']
 
                
                # 1. Distribution des rendements
                if len(trades_df['total_return']) > 0:
                    axes[0, 0].hist(trades_df['total_return'], bins=min(30, len(trades_df)), alpha=0.7, edgecolor='black')
                    axes[0, 0].set_title('Distribution of Trade Returns')
                    axes[0, 0].set_xlabel('Return')
                    axes[0, 0].set_ylabel('Frequency')
                    mean_return = trades_df['total_return'].mean()
                    axes[0, 0].axvline(mean_return, color='red', linestyle='--', label=f'Mean: {mean_return:.4f}')
                    axes[0, 0].legend()
                    
                if 'exit_date' in trades_df.columns and len(trades_df) > 0:
                    cumulative_returns = trades_df['total_return'].cumsum()
                # 2. Rendements cumulés
                
                    axes[0, 1].plot(cumulative_returns.index, cumulative_returns.values)
                    axes[0, 1].set_title('Cumulative Returns')
                    axes[0, 1].set_xlabel('Year')
                    axes[0, 1].set_ylabel('Cumulative Return')
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    axes[0, 1].tick_params(axis='x', rotation=45)
                
                # 3. Drawdown
                if 'exit_date' in trades_df.columns and len(trades_df) > 0:
                    cumulative_returns = trades_df['total_return'].cumsum()
                    running_max = cumulative_returns.expanding().max()
                    drawdown = cumulative_returns - running_max
                    
                    axes[0, 2].fill_between(time_index, drawdown.values, 0, alpha=0.7, color='red')
                    axes[0, 2].set_title('Drawdown Over Time')
                    axes[0, 2].set_xlabel('Year')
                    axes[0, 2].set_ylabel('Drawdown')
                    axes[0, 2].tick_params(axis='x', rotation=45)
                
                # 4. Durée des trades
                if 'duration' in trades_df.columns:
                    axes[1, 0].hist(trades_df['duration'], bins=min(20, len(trades_df)), alpha=0.7, edgecolor='black')
                    axes[1, 0].set_title('Trade Duration Distribution')
                    axes[1, 0].set_xlabel('Duration (periods)')
                    axes[1, 0].set_ylabel('Frequency')
                
                # 5. Performance par paire
                if 'pair' in trades_df.columns:
                    pair_performance = trades_df.groupby('pair')['total_return'].sum().sort_values(ascending=False)
                    top_pairs = pair_performance.head(10)
                    if len(top_pairs) > 0:
                        y_pos = range(len(top_pairs))
                        axes[1, 1].barh(y_pos, top_pairs.values)
                        axes[1, 1].set_yticks(y_pos)
                        axes[1, 1].set_yticklabels([str(pair)[:15] + '...' if len(str(pair)) > 15 else str(pair) 
                                                   for pair in top_pairs.index], fontsize=8)
                        axes[1, 1].set_title('Top 10 Pairs by Return')
                        axes[1, 1].set_xlabel('Total Return')
                
                # 6. Métriques clés
                metrics_text = f"""Total Trades: {backtest_results['total_trades']}
Win Rate: {backtest_results['win_rate']:.2%}
Total Return: {backtest_results['total_return']:.4f}
Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}
Max Drawdown: {backtest_results['max_drawdown']:.4f}
Avg Return/Trade: {backtest_results['avg_return_per_trade']:.6f}
Avg R² Score: {backtest_results.get('avg_r2_score', 0):.4f}"""
                
                axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                axes[1, 2].set_xlim(0, 1)
                axes[1, 2].set_ylim(0, 1)
                axes[1, 2].axis('off')
                axes[1, 2].set_title('Performance Metrics')
            else:
                # Si pas de trades, afficher un message
                for i in range(2):
                    for j in range(3):
                        axes[i, j].text(0.5, 0.5, 'No trades executed', 
                                       ha='center', va='center', fontsize=16)
                        axes[i, j].set_xlim(0, 1)
                        axes[i, j].set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'deep_learning_performance.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating performance plots: {e}")
    
    def plot_feature_importance(self, trader: DeepLearningPairsTrader):
        """Analyse d'importance des features (approximative)"""
        # Cette fonction nécessiterait une analyse plus poussée des modèles
        # Pour l'instant, on crée un placeholder
        logger.info("Feature importance analysis would require model interpretability tools")
        
    def plot_regime_analysis(self, data: pd.DataFrame):
        """Analyse des régimes de marché"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Volatility regimes
            if 'high_vol_regime' in data.columns and 'MthRet' in data.columns:
                vol_periods = data.groupby('high_vol_regime')['MthRet'].std()
                axes[0, 0].bar(['Low Vol', 'High Vol'], vol_periods.values)
                axes[0, 0].set_title('Volatility by Regime')
                axes[0, 0].set_ylabel('Return Volatility')
            
            # Market regimes
            if all(col in data.columns for col in ['bull_regime', 'bear_regime', 'MthRet']):
                regime_returns = data.groupby(['bull_regime', 'bear_regime'])['MthRet'].mean()
                axes[0, 1].bar(range(len(regime_returns)), regime_returns.values)
                axes[0, 1].set_title('Returns by Market Regime')
                axes[0, 1].set_ylabel('Average Return')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'regime_analysis.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating regime analysis plots: {e}")

def main_deep_learning():
    """Fonction principale pour la stratégie Deep Learning"""
    
    try:
        # Configuration des dossiers
        folder = os.path.expanduser('~/Desktop/Machine Learning/data_ML_Project')
        folder1 = os.path.expanduser('~/Desktop/Machine Learning/data_filtered')
        output_dir = os.path.expanduser('~/Desktop/Machine Learning/deep_learning_results')
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("=== DEEP LEARNING PAIRS TRADING STRATEGY ===")
        
        # Chargement des données
        logger.info("Loading and preprocessing data...")
        try:
            main_csv = os.path.join(folder, 'monthly_crsp.csv')
            jkp_csv = os.path.join(folder1, 'JKP_filtered.csv')
            zimmer_csv = os.path.join(folder1, 'Zimmer_filtered.csv')
            norm_csv = os.path.join(folder1, 'CompFirmCharac_filtred.csv')
            
            data_main = pd.read_csv(main_csv)
            data_jkp = pd.read_csv(jkp_csv)
            data_zimmer = pd.read_csv(zimmer_csv)
            data_comp = pd.read_csv(norm_csv)
            
        except FileNotFoundError as e:
            logger.error(f"Data files not found: {e}")
            logger.info("Please ensure all required CSV files are in the correct directories")
            return None
        
        # Préprocessing
        try:
            data_main['date'] = pd.to_datetime(data_main['MthCalDt'], errors='coerce')
            data_jkp['date'] = pd.to_datetime(data_jkp['date'], errors='coerce')
            data_zimmer['date'] = pd.to_datetime(data_zimmer['date'], errors='coerce')
            data_comp['datadate'] = pd.to_datetime(data_comp['datadate'], errors='coerce')
            
            # Supprimer les lignes avec des dates invalides
            data_main = data_main.dropna(subset=['date'])
            data_jkp = data_jkp.dropna(subset=['date'])
            data_zimmer = data_zimmer.dropna(subset=['date'])
            data_comp = data_comp.dropna(subset=['datadate'])
            
            data_comp['tic'] = data_comp['tic'].astype(str)
            data_comp.rename(columns={'tic': 'Ticker'}, inplace=True)
            
            start_date = pd.to_datetime('2010-01-01')  # Date plus récente pour plus de données
            data_main = data_main[data_main['date'] >= start_date]
            data_jkp = data_jkp[data_jkp['date'] >= start_date]
            data_zimmer = data_zimmer[data_zimmer['date'] >= start_date]
            
            for df in [data_main, data_jkp, data_zimmer]:
                df['year'] = df['date'].dt.year
                df['month'] = df['date'].dt.month
            
            # Fusion des données
            merged = data_main.merge(
                data_jkp.drop(columns=['date'], errors='ignore'),
                on=['year', 'month'],
                how='left',
                suffixes=('', '_jkp')
            ).merge(
                data_zimmer.drop(columns=['date'], errors='ignore'),
                on=['year', 'month'],
                how='left',
                suffixes=('', '_zimmer')
            )
            
            # Filtrer les tickers avec suffisamment de données
            ticker_counts = merged.groupby('Ticker').size()
            valid_tickers = ticker_counts[ticker_counts >= DEEP_LEARNING_CONFIG['min_observations']].index
            merged = merged[merged['Ticker'].isin(valid_tickers)]
            
            # Nettoyage
            columns_to_drop = [
                'PERMNO', 'HdrCUSIP', 'CUSIP', 'TradingSymbol',
                'PERMCO', 'SICCD', 'NAICS', 'year', 'month', 'MthCalDt',
            ]
            merged.drop(columns=[col for col in columns_to_drop if col in merged.columns], inplace=True)
            
            # Fusion avec les caractéristiques firmes (optionnelle)
            if len(data_comp) > 0:
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
            
            # Supprimer les lignes avec trop de NaN et nettoyer les valeurs extrêmes
            merged = merged.dropna(subset=['Ticker', 'date', 'MthRet'])
            
            # Nettoyer toutes les colonnes numériques
            logger.info("Cleaning extreme values in dataset...")
            numeric_cols = merged.select_dtypes(include=[np.number]).columns
            feature_engineer = AdvancedFeatureEngineer()
            
            for col in numeric_cols:
                if col not in ['year', 'month']:  # Préserver les colonnes de date
                    merged[col] = feature_engineer._clean_extreme_values(merged[col])
            
            # Vérification finale qu'il n'y a plus d'infinis
            inf_cols = []
            for col in numeric_cols:
                if np.any(np.isinf(merged[col])) or np.any(merged[col].abs() > 1e10):
                    inf_cols.append(col)
                    merged[col] = np.clip(merged[col].replace([np.inf, -np.inf], np.nan).fillna(0), -100, 100)
            
            if inf_cols:
                logger.warning(f"Cleaned extreme values in columns: {inf_cols}")
            
            logger.info(f"Data loaded: {len(merged)} records, {merged['Ticker'].nunique()} unique tickers")
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {e}")
            return None
        
        merged = merged.sort_values(['Ticker', 'date']).reset_index(drop=True)
        merged['MthRet'] = merged.groupby('Ticker')['MthRet'].shift(1)
        merged = merged.dropna(subset=['MthRet'])
        
        
        # Vérifier qu'on a assez de données
        if len(merged) < 1000 or merged['Ticker'].nunique() < 10:
            logger.error("Insufficient data for analysis")
            return None
        
        
        
        # Initialiser le trader Deep Learning
        trader = DeepLearningPairsTrader(DEEP_LEARNING_CONFIG)
        
        # Sélection de paires avec ML
        logger.info("Selecting pairs using Machine Learning...")
        valid_pairs = trader.select_pairs_ml(merged)
        
        if not valid_pairs:
            logger.error("No valid pairs found with ML approach!")
            return None
        
        logger.info(f"Selected {len(valid_pairs)} pairs for Deep Learning training")
        
        # Sauvegarder les paires sélectionnées
        try:
            pairs_summary = pd.DataFrame([
                {
                    'ticker1': p['ticker1'],
                    'ticker2': p['ticker2'], 
                    'correlation': p['correlation'],
                    'coint_pvalue': p['coint_pvalue'],
                    'ml_score': p['ml_score']
                } for p in valid_pairs
            ])
            pairs_summary.to_csv(os.path.join(output_dir, 'selected_pairs_ml.csv'), index=False)
            logger.info("Pairs summary saved")
        except Exception as e:
            logger.warning(f"Could not save pairs summary: {e}")
        
        # Entraînement des modèles
        logger.info("Training Deep Learning models...")
        success = trader.train_models(valid_pairs)
        
        if not success:
            logger.error("Failed to train any models!")
            return None
        
        # Backtesting
        logger.info("Running backtesting...")
        backtester = AdvancedBacktester(DEEP_LEARNING_CONFIG)
        results = backtester.run_backtest(trader, valid_pairs)
        
        # Affichage des résultats
        logger.info("=== DEEP LEARNING STRATEGY RESULTS ===")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Win Rate: {results['win_rate']:.2%}")
        logger.info(f"Total Return: {results['total_return']:.4f}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.4f}")
        logger.info(f"Average Return per Trade: {results['avg_return_per_trade']:.6f}")
        logger.info(f"Average R² Score: {results['avg_r2_score']:.4f}")  
        
        # Sauvegarder les résultats
        try:
            results_summary = pd.DataFrame([{
                'strategy': 'Deep Learning Pairs Trading',
                'total_trades': results['total_trades'],
                'win_rate': results['win_rate'],
                'total_return': results['total_return'],
                'sharpe_ratio': results['sharpe_ratio'],
                'max_drawdown': results['max_drawdown'],
                'avg_return_per_trade': results['avg_return_per_trade'],
                'avg_r2_score': results['avg_r2_score']
            }])
            results_summary.to_csv(os.path.join(output_dir, 'strategy_results.csv'), index=False)
            
            # Sauvegarder les trades détaillés
            if not results['trades_df'].empty:
                results['trades_df'].to_csv(os.path.join(output_dir, 'detailed_trades.csv'), index=False)
                
        except Exception as e:
            logger.warning(f"Could not save results: {e}")
        
        # Visualisations
        logger.info("Creating visualizations...")
        try:
            viz = AdvancedVisualization(output_dir)
            viz.plot_model_performance(results)
        except Exception as e:
            logger.warning(f"Could not create visualizations: {e}")
        
        # Sauvegarder les modèles
        logger.info("Saving trained models...")
        try:
            with open(os.path.join(output_dir, 'trained_models.pkl'), 'wb') as f:
                pickle.dump(trader.ensemble_models, f)
            logger.info("Models saved successfully")
        except Exception as e:
            logger.warning(f"Could not save models: {e}")
        
        logger.info(f"Deep Learning strategy analysis complete. Results saved to {output_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Critical error in main_deep_learning: {e}")
        return None

if __name__ == "__main__":
    try:
        results = main_deep_learning()
        if results is not None:
            print("\n=== FINAL RESULTS ===")
            print(f"Strategy completed successfully!")
            print(f"Total trades: {results['total_trades']}")
            print(f"Win rate: {results['win_rate']:.2%}")
            print(f"Total return: {results['total_return']:.4f}")
            print(f"Sharpe ratio: {results['sharpe_ratio']:.3f}")
        else:
            print("Strategy execution failed. Check logs for details.")
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()