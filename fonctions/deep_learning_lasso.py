import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.feature_selection import SelectFromModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')


class DeepLearningRegressor(nn.Module):
    """
    Réseau de neurones profond pour la régression.
    """
    def __init__(self, input_dim, hidden_layers, dropout_rate=0.2, activation='relu'):
        super(DeepLearningRegressor, self).__init__()
        
        # Vérifications de sécurité
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim doit être un entier positif, reçu: {input_dim}")
        
        # Fonction d'activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
        
        # Construction des couches
        layers = []
        prev_size = int(input_dim)
        
        for hidden_size in hidden_layers:
            hidden_size = int(hidden_size)
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),  # Remplacé BatchNorm par LayerNorm
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Couche de sortie
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialisation des poids
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class AdaptiveFeatureSelector:
    """
    Sélecteur de features adaptatif utilisant Lasso avec différentes stratégies.
    """
    def __init__(self, 
                 method='lasso', 
                 alpha_range=(1e-4, 1e-1),
                 max_features=None,
                 min_features=5,
                 stability_threshold=0.7,
                 memory_factor=0.3):
        """
        Parameters:
        -----------
        method : str
            Méthode de sélection ('lasso', 'elastic_net', 'adaptive_lasso')
        alpha_range : tuple
            Plage des valeurs alpha pour la régularisation
        max_features : int
            Nombre maximum de features à sélectionner
        min_features : int
            Nombre minimum de features à garder
        stability_threshold : float
            Seuil de stabilité pour la sélection des features
        memory_factor : float
            Facteur de mémoire pour la stabilité des features
        """
        self.method = method
        self.alpha_range = alpha_range
        self.max_features = max_features
        self.min_features = min_features
        self.stability_threshold = stability_threshold
        self.memory_factor = memory_factor
        
        # Historique des sélections
        self.feature_history = []
        self.stability_scores = {}
        
    def _get_selector_model(self, X, y):
        """
        Retourne le modèle de sélection approprié.
        """
        if self.method == 'lasso':
            # Lasso avec validation croisée
            alphas = np.logspace(np.log10(self.alpha_range[0]), 
                               np.log10(self.alpha_range[1]), 50)
            return LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=2000)
        
        elif self.method == 'elastic_net':
            # ElasticNet avec validation croisée
            alphas = np.logspace(np.log10(self.alpha_range[0]), 
                               np.log10(self.alpha_range[1]), 20)
            l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
            return ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, 
                              cv=5, random_state=42, max_iter=2000)
        
        elif self.method == 'adaptive_lasso':
            # Lasso adaptatif (deux étapes)
            # Étape 1: Ridge pour obtenir les poids
            from sklearn.linear_model import RidgeCV
            ridge = RidgeCV(cv=5)
            ridge.fit(X, y)
            
            # Étape 2: Lasso avec poids adaptatifs
            weights = 1 / (np.abs(ridge.coef_) + 1e-8)
            X_weighted = X / weights
            
            alphas = np.logspace(np.log10(self.alpha_range[0]), 
                               np.log10(self.alpha_range[1]), 50)
            lasso = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=2000)
            lasso.fit(X_weighted, y)
            
            # Ajuster les coefficients
            lasso.coef_ = lasso.coef_ / weights
            return lasso
        
        else:
            raise ValueError(f"Méthode {self.method} non supportée")
    
    def select_features(self, X, y, feature_names):
        """
        Sélectionne les features en utilisant la méthode choisie.
        """
        # Obtenir le modèle de sélection
        selector_model = self._get_selector_model(X, y)
        
        # Ajustement si ce n'est pas déjà fait (pour adaptive_lasso)
        if self.method != 'adaptive_lasso':
            selector_model.fit(X, y)
        
        # Sélection basée sur les coefficients non-nuls
        selected_mask = np.abs(selector_model.coef_) > 1e-8
        
        # Limiter le nombre de features si spécifié
        if self.max_features is not None and np.sum(selected_mask) > self.max_features:
            # Garder les features avec les plus gros coefficients
            coef_abs = np.abs(selector_model.coef_)
            top_indices = np.argsort(coef_abs)[-self.max_features:]
            selected_mask = np.zeros_like(selected_mask, dtype=bool)
            selected_mask[top_indices] = True
        
        # S'assurer d'avoir au moins min_features
        if np.sum(selected_mask) < self.min_features:
            coef_abs = np.abs(selector_model.coef_)
            top_indices = np.argsort(coef_abs)[-self.min_features:]
            selected_mask = np.zeros_like(selected_mask, dtype=bool)
            selected_mask[top_indices] = True
        
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        
        # Mise à jour de l'historique et des scores de stabilité
        self._update_stability_scores(selected_features, feature_names)
        
        # Application de la stabilité si on a un historique
        if len(self.feature_history) > 0:
            selected_features = self._apply_stability_filter(selected_features, feature_names)
        
        self.feature_history.append(selected_features)
        
        return selected_features, selector_model.alpha_, selector_model.coef_[selected_mask]
    
    def _update_stability_scores(self, selected_features, all_features):
        """
        Met à jour les scores de stabilité des features.
        """
        for feature in all_features:
            if feature not in self.stability_scores:
                self.stability_scores[feature] = 0.0
            
            # Facteur de décroissance temporelle
            self.stability_scores[feature] *= (1 - self.memory_factor)
            
            # Augmentation si la feature est sélectionnée
            if feature in selected_features:
                self.stability_scores[feature] += self.memory_factor
    
    def _apply_stability_filter(self, current_selection, all_features):
        """
        Applique un filtre de stabilité aux features sélectionnées.
        """
        # Features stables (sélectionnées fréquemment)
        stable_features = [f for f in all_features 
                          if self.stability_scores.get(f, 0) >= self.stability_threshold]
        
        # Combiner features actuelles et features stables
        combined_features = list(set(current_selection + stable_features))
        
        # Limiter si nécessaire
        if self.max_features is not None and len(combined_features) > self.max_features:
            # Prioriser par score de stabilité
            combined_features.sort(key=lambda x: self.stability_scores.get(x, 0), reverse=True)
            combined_features = combined_features[:self.max_features]
        
        return combined_features
    
    def get_feature_stability_report(self):
        """
        Retourne un rapport sur la stabilité des features.
        """
        if not self.stability_scores:
            return pd.DataFrame()
        
        stability_df = pd.DataFrame([
            {'feature': feature, 'stability_score': score}
            for feature, score in self.stability_scores.items()
        ]).sort_values('stability_score', ascending=False)
        
        return stability_df


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                epochs, patience, device):
    """
    Entraîne le modèle avec early stopping - Version robuste.
    """
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Vérification que les loaders ne sont pas vides
    if len(train_loader) == 0 or len(val_loader) == 0:
        return {
            'final_epoch': 0,
            'best_val_loss': float('inf'),
            'train_losses': [],
            'val_losses': []
        }
    
    for epoch in range(epochs):
        try:
            # Phase d'entraînement
            model.train()
            train_loss = 0.0
            train_count = 0
            
            for batch_X, batch_y in train_loader:
                try:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    # Vérification des dimensions
                    if batch_X.numel() == 0 or batch_y.numel() == 0:
                        continue
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    
                    # S'assurer que outputs et batch_y ont la même forme
                    outputs = outputs.squeeze()
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
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()
                    train_count += 1
                    
                except Exception as e:
                    print(f"Erreur dans batch d'entraînement: {e}")
                    continue
            
            if train_count == 0:
                break
                
            # Phase de validation
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
                
            train_loss /= train_count
            val_loss /= val_count
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if scheduler:
                scheduler.step(val_loss)
            
            # Early stopping
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
    
    # Charger le meilleur modèle
    try:
        if 'best_model_state' in locals():
            model.load_state_dict(best_model_state)
    except:
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
    # Paramètres de sélection des features
    feature_selection_method: str = 'lasso',
    feature_selection_frequency: int = 1,
    max_features: int = None,
    min_features: int = 5,
    lasso_alpha_range: Tuple[float, float] = (1e-4, 1e-1),
    stability_threshold: float = 0.7,
    memory_factor: float = 0.3,
    **base_dl_params
) -> Dict[str, Any]:
    """
    Deep Learning avec présélection adaptative des features via Lasso.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame avec les données
    date_column : str
        Nom de la colonne de date
    target_column : str
        Nom de la variable cible
    feature_columns : list
        Liste des features à utiliser
    train_years : int
        Nombre d'années pour l'entraînement
    test_years : int
        Nombre d'années pour le test
    tune_frequency : int
        Fréquence de tuning des hyperparamètres (tous les X fenêtres)
    n_trials : int
        Nombre d'essais pour Optuna
    early_stopping_rounds : int
        Patience pour l'early stopping
    feature_selection_method : str
        Méthode de sélection ('lasso', 'elastic_net', 'adaptive_lasso')
    feature_selection_frequency : int
        Fréquence de re-sélection des features (tous les X fenêtres)
    max_features : int
        Nombre maximum de features à sélectionner
    min_features : int
        Nombre minimum de features à garder
    lasso_alpha_range : tuple
        Plage des valeurs alpha pour la régularisation Lasso
    stability_threshold : float
        Seuil de stabilité pour la sélection des features
    memory_factor : float
        Facteur de mémoire pour la stabilité des features
    **base_dl_params : dict
        Paramètres de base pour le modèle
    
    Returns:
    --------
    Dict contenant les résultats
    """
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device: {device}")
    
    # Préparation des données
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df_sorted = df.sort_values(by=date_column)
    
    # Définir les features initiales
    if feature_columns is None:
        exclude_cols = [target_column, date_column]
        feature_columns = [col for col in df_sorted.columns if col not in exclude_cols]
    
    # Initialisation du sélecteur de features
    feature_selector = AdaptiveFeatureSelector(
        method=feature_selection_method,
        alpha_range=lasso_alpha_range,
        max_features=max_features,
        min_features=min_features,
        stability_threshold=stability_threshold,
        memory_factor=memory_factor
    )
    
    # Paramètres de base pour le Deep Learning
    base_params = {
        'hidden_layers': [512, 256, 128, 64],
        'dropout_rate': 0.3,
        'learning_rate': 0.0001,
        'batch_size': 128,
        'epochs': 500,
        'activation': 'relu',
        'optimizer': 'adam',
        'patience': 15,
        'weight_decay': 1e-4
    }
    base_params.update(base_dl_params)
    
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
        
        # PHASE 1: SÉLECTION DES FEATURES
        if window_count % feature_selection_frequency == 0:
            print(f"Fenêtre {window_count + 1}: Sélection des features avec {feature_selection_method}...")
            
            # Préparation des données pour la sélection
            X_selection = train_data[feature_columns].fillna(train_data[feature_columns].median())
            y_selection = train_data[target_column]
            
            # Normalisation pour la sélection
            scaler_selection = StandardScaler()
            X_selection_scaled = scaler_selection.fit_transform(X_selection)
            
            # Sélection des features
            selected_features, lasso_alpha, lasso_coefs = feature_selector.select_features(
                X_selection_scaled, y_selection.values, feature_columns
            )
            
            current_selected_features = selected_features
            
            print(f"Features sélectionnées: {len(selected_features)}/{len(feature_columns)}")
            print(f"Alpha Lasso optimal: {lasso_alpha:.6f}")
            print(f"Top 5 features: {selected_features[:5]}")
        
        # Préparation des données avec les features sélectionnées
        X_train = train_data[current_selected_features].fillna(train_data[current_selected_features].median())
        y_train = train_data[target_column]
        X_test = test_data[current_selected_features].fillna(X_train.median())
        y_test = test_data[target_column]
        
        # Normalisation des features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # PHASE 2: HYPERPARAMETER TUNING BAYÉSIEN
        if window_count % tune_frequency == 0:
            print(f"Fenêtre {window_count + 1}: Hyperparameter tuning bayésien en cours...")
            
            def objective(trial):
                n_layers = trial.suggest_categorical('n_layers', [3, 4, 5])
                
                layer1 = trial.suggest_categorical('layer1', [64, 128, 256, 512])
                layer2 = trial.suggest_categorical('layer2', [32, 64, 128, 256])
                layer3 = trial.suggest_categorical('layer3', [16, 32, 64, 128])
                layer4 = trial.suggest_categorical('layer4', [8, 16, 32, 64])
                layer5 = trial.suggest_categorical('layer5', [8, 16, 32])
                
                if n_layers == 3:
                    layer_sizes = [layer1, layer2, layer3]
                elif n_layers == 4:
                    layer_sizes = [layer1, layer2, layer3, layer4]
                else:
                    layer_sizes = [layer1, layer2, layer3, layer4, layer5]
                
                params = {
                    'hidden_layers': layer_sizes,
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                    'activation': trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu']),
                    'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop']),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                    'epochs': 50,
                    'patience': 8
                }
                
                # Validation croisée temporelle
                tscv = TimeSeriesSplit(n_splits=3)
                val_scores = []
                
                for train_idx, valid_idx in tscv.split(X_train_scaled):
                    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[valid_idx]
                    y_tr, y_val = y_train.iloc[train_idx].values, y_train.iloc[valid_idx].values
                    
                    X_tr = np.array(X_tr, dtype=np.float32)
                    X_val = np.array(X_val, dtype=np.float32)
                    y_tr = np.array(y_tr, dtype=np.float32).reshape(-1)
                    y_val = np.array(y_val, dtype=np.float32).reshape(-1)
                    
                    if len(y_tr) < 5 or len(y_val) < 2:
                        val_scores.append(1e6)
                        continue
                    
                    try:
                        train_dataset = TensorDataset(
                            torch.FloatTensor(X_tr), 
                            torch.FloatTensor(y_tr)
                        )
                        val_dataset = TensorDataset(
                            torch.FloatTensor(X_val), 
                            torch.FloatTensor(y_val)
                        )
                        
                        effective_batch_size = min(params['batch_size'], len(train_dataset) // 2, 32)
                        
                        train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=effective_batch_size)
                        
                        model = DeepLearningRegressor(
                            input_dim=int(X_tr.shape[1]),
                            hidden_layers=[int(x) for x in params['hidden_layers']],
                            dropout_rate=params['dropout_rate'],
                            activation=params['activation']
                        )
                        
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
                        
                        training_info = train_model(
                            model, train_loader, val_loader, criterion, optimizer, scheduler,
                            params['epochs'], params['patience'], device
                        )
                        
                        val_scores.append(training_info['best_val_loss'])
                        
                    except Exception as e:
                        print(f"Erreur dans validation croisée: {e}")
                        val_scores.append(1e6)
                
                return np.mean(val_scores)
            
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
            
            # Mise à jour des meilleurs paramètres
            best_trial_params = study.best_trial.params
            
            n_layers = best_trial_params['n_layers']
            if n_layers == 3:
                hidden_layers = [
                    best_trial_params['layer1'],
                    best_trial_params['layer2'],
                    best_trial_params['layer3']
                ]
            elif n_layers == 4:
                hidden_layers = [
                    best_trial_params['layer1'],
                    best_trial_params['layer2'],
                    best_trial_params['layer3'],
                    best_trial_params['layer4']
                ]
            else:
                hidden_layers = [
                    best_trial_params['layer1'],
                    best_trial_params['layer2'],
                    best_trial_params['layer3'],
                    best_trial_params['layer4'],
                    best_trial_params['layer5']
                ]
            
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
            
            print(f"Nouveaux meilleurs paramètres: {best_params}")
        
        # PHASE 3: ENTRAÎNEMENT DU MODÈLE FINAL
        try:
            print(f"Dimensions: X_train={X_train_scaled.shape}, y_train={y_train.shape}")
            
            # Conversion sécurisée en numpy arrays
            X_train_array = np.array(X_train_scaled, dtype=np.float32)
            y_train_array = np.array(y_train.values, dtype=np.float32).reshape(-1)
            X_test_array = np.array(X_test_scaled, dtype=np.float32)
            y_test_array = np.array(y_test.values, dtype=np.float32).reshape(-1)
            
            if len(y_train_array) < 10:
                raise ValueError("Pas assez de données d'entraînement")
            
            # Préparation des données finales
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_array), 
                torch.FloatTensor(y_train_array)
            )
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test_array), 
                torch.FloatTensor(y_test_array)
            )
            
            # Split train/validation
            train_size = max(int(0.9 * len(train_dataset)), len(train_dataset) - 10)
            val_size = len(train_dataset) - train_size
            
            if val_size < 5:
                train_subset = train_dataset
                val_subset = train_dataset
            else:
                train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            
            # DataLoaders avec batch_size adaptatif
            effective_batch_size = min(best_params['batch_size'], len(train_subset) // 2, 32)
            
            train_loader = DataLoader(train_subset, batch_size=effective_batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=effective_batch_size)
            test_loader = DataLoader(test_dataset, batch_size=effective_batch_size)
            
            # Modèle final avec les features sélectionnées
            final_model = DeepLearningRegressor(
                input_dim=int(X_train_array.shape[1]),  # Utilise le nombre de features sélectionnées
                hidden_layers=[int(x) for x in best_params['hidden_layers']],
                dropout_rate=best_params['dropout_rate'],
                activation=best_params['activation']
            )
            
            # Optimiseur
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
            
            # Entraînement final
            training_info = train_model(
                final_model, train_loader, val_loader, criterion, optimizer, scheduler,
                best_params['epochs'], best_params['patience'], device
            )
            
            # Prédictions
            final_model.eval()
            test_predictions = []
            
            with torch.no_grad():
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
                        print(f"Erreur dans prédiction batch {batch_idx}: {e}")
                        batch_size = len(batch_X)
                        test_predictions.extend([y_train.mean()] * batch_size)
            
            # Ajustement du nombre de prédictions
            if len(test_predictions) != len(y_test):
                print(f"Ajustement: {len(test_predictions)} prédictions pour {len(y_test)} échantillons")
                if len(test_predictions) > len(y_test):
                    test_predictions = test_predictions[:len(y_test)]
                else:
                    while len(test_predictions) < len(y_test):
                        test_predictions.append(y_train.mean())
            
            test_pred = np.array(test_predictions, dtype=np.float32)
            
        except Exception as e:
            print(f"Erreur lors de l'entraînement: {e}")
            print(f"Type d'erreur: {type(e)}")
            import traceback
            traceback.print_exc()
            
            # Fallback simple
            test_pred = np.full(len(y_test), y_train.mean())
            training_info = {'final_epoch': 0, 'best_val_loss': float('inf')}
        
        # Calcul des métriques
        window_metrics = {
            'window': window_count + 1,
            'train_period': f"{current_start.strftime('%Y-%m')} à {train_end.strftime('%Y-%m')}",
            'test_period': f"{test_start.strftime('%Y-%m')} à {test_end.strftime('%Y-%m')}",
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
        
        # Feature importance via permutation
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
        
        # Informations sur la sélection des features
        feature_selection_info = {
            'window': window_count + 1,
            'selected_features': current_selected_features.copy(),
            'n_selected': len(current_selected_features),
            'n_total': len(feature_columns),
            'selection_ratio': len(current_selected_features) / len(feature_columns)
        }
        
        # Ajout du rapport de stabilité si disponible
        if window_count > 0:
            stability_report = feature_selector.get_feature_stability_report()
            feature_selection_info['stability_report'] = stability_report
        
        # Stockage des résultats
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
        
        print(f"Fenêtre {window_count + 1}: R² = {window_metrics['r2']:.4f}, "
              f"RMSE = {window_metrics['rmse']:.6f}, "
              f"Features = {len(current_selected_features)}, "
              f"Epochs = {training_info['final_epoch']}")
        
        current_start += pd.DateOffset(years=1)
        window_count += 1
    
    # Calcul des métriques globales
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
    
    # Rapport final de stabilité des features
    final_stability_report = feature_selector.get_feature_stability_report()
    results['final_feature_stability'] = final_stability_report
    
    print(f"\n=== RÉSULTATS GLOBAUX DEEP LEARNING avec SÉLECTION LASSO ===")
    print(f"R² global: {overall_metrics['overall_r2']:.4f}")
    print(f"RMSE global: {overall_metrics['overall_rmse']:.6f}")
    print(f"R² moyen par fenêtre: {overall_metrics['avg_window_r2']:.4f}")
    print(f"Nombre de fenêtres: {overall_metrics['n_windows']}")
    print(f"Époques moyennes: {overall_metrics['avg_epochs']:.1f}")
    print(f"Features moyennes sélectionnées: {overall_metrics['avg_features_selected']:.1f}/{len(feature_columns)}")
    print(f"Réduction de features: {overall_metrics['feature_reduction_ratio']:.1%}")
    
    # Top 10 features les plus stables
    if not final_stability_report.empty:
        print(f"\n=== TOP 10 FEATURES LES PLUS STABLES ===")
        for idx, row in final_stability_report.head(10).iterrows():
            print(f"{row['feature']:20}: {row['stability_score']:.3f}")
    
    return results


def calculate_pytorch_feature_importance(model, X_test, y_test, device):
    """
    Calcule l'importance des features via permutation pour PyTorch.
    """
    model.eval()
    X_tensor = torch.FloatTensor(X_test).to(device)
    
    # Score de base
    with torch.no_grad():
        baseline_pred = model(X_tensor).squeeze().cpu().numpy()
    baseline_score = mean_squared_error(y_test, baseline_pred)
    
    importances = []
    
    for i in range(X_test.shape[1]):
        # Permutation de la feature i
        X_permuted = X_test.copy()
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
        X_permuted_tensor = torch.FloatTensor(X_permuted).to(device)
        
        # Score avec permutation
        with torch.no_grad():
            permuted_pred = model(X_permuted_tensor).squeeze().cpu().numpy()
        permuted_score = mean_squared_error(y_test, permuted_pred)
        
        # Importance = augmentation de l'erreur
        importance = permuted_score - baseline_score
        importances.append(max(0, importance))
    
    # Normalisation
    importances = np.array(importances)
    if importances.sum() > 0:
        importances = importances / importances.sum()
    
    return importances


def analyze_feature_selection_results(results):
    """
    Analyse détaillée des résultats de sélection des features.
    """
    print("\n=== ANALYSE DÉTAILLÉE DE LA SÉLECTION DES FEATURES ===")
    
    # Evolution du nombre de features
    n_features_by_window = [info['n_selected'] for info in results['feature_selection_evolution']]
    
    print(f"Nombre de features par fenêtre:")
    print(f"  Minimum: {min(n_features_by_window)}")
    print(f"  Maximum: {max(n_features_by_window)}")
    print(f"  Moyenne: {np.mean(n_features_by_window):.1f}")
    print(f"  Médiane: {np.median(n_features_by_window):.1f}")
    
    # Features les plus souvent sélectionnées
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
    
    # Corrélation entre nombre de features et performance
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


# Exemple d'utilisation complète:
""" 
# Installation nécessaire:
# pip install torch optuna scikit-learn pandas numpy

# Appel de la fonction avec sélection Lasso
dl_lasso_results = sliding_window_dl_prediction_with_lasso(
    df=your_dataframe,
    date_column='date',
    target_column='MthRet',
    feature_columns=None,  # Sera automatiquement défini
    train_years=5,
    test_years=1,
    tune_frequency=4,
    n_trials=15,
    early_stopping_rounds=8,
    # Paramètres spécifiques à la sélection Lasso
    feature_selection_method='lasso',  # 'lasso', 'elastic_net', 'adaptive_lasso'
    feature_selection_frequency=1,     # Re-sélection à chaque fenêtre
    max_features=50,                   # Maximum 50 features
    min_features=10,                   # Minimum 10 features
    lasso_alpha_range=(1e-4, 1e-1),   # Plage alpha pour Lasso
    stability_threshold=0.7,           # Seuil de stabilité
    memory_factor=0.3                  # Facteur de mémoire
)

# Analyse détaillée des résultats de sélection
feature_analysis = analyze_feature_selection_results(dl_lasso_results)

# Comparaison avec la version sans sélection
print("\\n=== COMPARAISON AVEC/SANS SÉLECTION DE FEATURES ===")
print(f"Avec sélection Lasso    : R² = {dl_lasso_results['overall_metrics']['overall_r2']:.4f}")
print(f"Features moyennes       : {dl_lasso_results['overall_metrics']['avg_features_selected']:.1f}")
print(f"Réduction de features   : {dl_lasso_results['overall_metrics']['feature_reduction_ratio']:.1%}")

# Rapport de stabilité final
stability_report = dl_lasso_results['final_feature_stability']
if not stability_report.empty:
    print(f"\\nFeatures très stables (>0.8): {len(stability_report[stability_report['stability_score'] > 0.8])}")
    print(f"Features moyennement stables (0.5-0.8): {len(stability_report[(stability_report['stability_score'] > 0.5) & (stability_report['stability_score'] <= 0.8)])}")
"""

# Exemple d'utilisation complète:
"""
# Installation nécessaire:
# pip install torch optuna scikit-learn pandas numpy

# Appel de la fonction avec sélection Lasso
dl_lasso_results = sliding_window_dl_prediction_with_lasso(
    df=your_dataframe,
    date_column='date',
    target_column='MthRet',
    feature_columns=None,  # Sera automatiquement défini
    train_years=5,
    test_years=1,
    tune_frequency=4,
    n_trials=15,
    early_stopping_rounds=8,
    # Paramètres spécifiques à la sélection Lasso
    feature_selection_method='lasso',  # 'lasso', 'elastic_net', 'adaptive_lasso'
    feature_selection_frequency=1,     # Re-sélection à chaque fenêtre
    max_features=50,                   # Maximum 50 features
    min_features=10,                   # Minimum 10 features
    lasso_alpha_range=(1e-4, 1e-1),   # Plage alpha pour Lasso
    stability_threshold=0.7,           # Seuil de stabilité
    memory_factor=0.3                  # Facteur de mémoire
)

# Analyse détaillée des résultats de sélection
feature_analysis = analyze_feature_selection_results(dl_lasso_results)

# Comparaison avec la version sans sélection
print("\\n=== COMPARAISON AVEC/SANS SÉLECTION DE FEATURES ===")
print(f"Avec sélection Lasso    : R² = {dl_lasso_results['overall_metrics']['overall_r2']:.4f}")
print(f"Features moyennes       : {dl_lasso_results['overall_metrics']['avg_features_selected']:.1f}")
print(f"Réduction de features   : {dl_lasso_results['overall_metrics']['feature_reduction_ratio']:.1%}")

# Rapport de stabilité final
stability_report = dl_lasso_results['final_feature_stability']
if not stability_report.empty:
    print(f"\\nFeatures très stables (>0.8): {len(stability_report[stability_report['stability_score'] > 0.8])}")
    print(f"Features moyennement stables (0.5-0.8): {len(stability_report[(stability_report['stability_score'] > 0.5) & (stability_report['stability_score'] <= 0.8)])}")
    
    #APPEL OPTIMAL
    dl_lasso_results = sliding_window_dl_prediction_with_lasso(
    df=merged,
    date_column='date',
    target_column='target_ret',
    feature_columns=None,  # Sera automatiquement défini
    train_years=40, #essaye 20 30
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
    
    
    
"""



