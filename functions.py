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



import pandas as pd
import numpy as np
from typing import Dict, Any
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
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
        prev_size = input_dim
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
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
                        # Les deux sont des scalaires
                        loss = criterion(outputs.unsqueeze(0), batch_y.unsqueeze(0))
                    elif outputs.dim() == 0:
                        # outputs est scalaire, batch_y est un vecteur
                        outputs = outputs.repeat(batch_y.size(0))
                        loss = criterion(outputs, batch_y)
                    elif batch_y.dim() == 0:
                        # batch_y est scalaire, outputs est un vecteur
                        batch_y = batch_y.repeat(outputs.size(0))
                        loss = criterion(outputs, batch_y)
                    else:
                        # Cas normal
                        loss = criterion(outputs, batch_y)
                    
                    loss.backward()
                    
                    # Gradient clipping
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
                        
                        # Même logique que pour l'entraînement
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
            
            # Learning rate scheduling
            if scheduler:
                scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Sauvegarder le meilleur modèle
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
                    
        except Exception as e:
            print(f"Erreur dans époque {epoch}: {e}")
            break
    
    # Charger le meilleur modèle si on en a un
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


def sliding_window_dl_prediction(
    df: pd.DataFrame,
    date_column: str = 'date',
    target_column: str = 'MthRet',
    feature_columns: list = None,
    train_years: int = 5,
    test_years: int = 1,
    tune_frequency: int = 4,
    n_trials: int = 20,
    early_stopping_rounds: int = 10,
    **base_dl_params
) -> Dict[str, Any]:
    """
    Deep Learning avec PyTorch, fenêtre glissante et hyperparameter tuning bayésien.
    
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
        Fréquence de tuning (tous les X fenêtres)
    n_trials : int
        Nombre d'essais pour Optuna
    early_stopping_rounds : int
        Patience pour l'early stopping
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
    
    # Définir les features
    if feature_columns is None:
        exclude_cols = [target_column, 'date', 'MthRet', 'Ticker', 'sprtrn']
        feature_columns = [col for col in df_sorted.columns if col not in exclude_cols]
    
    # Paramètres de base pour le Deep Learning
    base_params = {
        'hidden_layers': [128, 64, 32],
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 200,
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
        'best_params_evolution': [],
        'dates': [],
        'training_history': []
    }
    
    min_date = df_sorted[date_column].min()
    max_date = df_sorted[date_column].max()
    current_start = min_date
    window_count = 0
    best_params = base_params.copy()
    
    print(f"\n=== DEEP LEARNING (PyTorch) - BAYESIAN TUNING ===")
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
            
        # Préparation des données avec normalisation
        X_train = train_data[feature_columns].fillna(train_data[feature_columns].median())
        y_train = train_data[target_column]
        X_test = test_data[feature_columns].fillna(X_train.median())
        y_test = test_data[target_column]
        
        # Normalisation des features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # HYPERPARAMETER TUNING BAYÉSIEN (Optuna)
        if window_count % tune_frequency == 0:
            print(f"Fenêtre {window_count + 1}: Hyperparameter tuning bayésien en cours...")
            
            def objective(trial):
                # Architecture du réseau - choix fixes pour éviter l'erreur Optuna
                n_layers = trial.suggest_categorical('n_layers', [3, 4, 5])
                
                # Définir toutes les couches avec les mêmes choix possibles
                layer1 = trial.suggest_categorical('layer1', [64, 128, 256, 512])
                layer2 = trial.suggest_categorical('layer2', [32, 64, 128, 256])
                layer3 = trial.suggest_categorical('layer3', [16, 32, 64, 128])
                layer4 = trial.suggest_categorical('layer4', [8, 16, 32, 64])
                layer5 = trial.suggest_categorical('layer5', [8, 16, 32])
                
                # Construire l'architecture selon le nombre de couches
                if n_layers == 3:
                    layer_sizes = [layer1, layer2, layer3]
                elif n_layers == 4:
                    layer_sizes = [layer1, layer2, layer3, layer4]
                else:  # 5 layers
                    layer_sizes = [layer1, layer2, layer3, layer4, layer5]
                
                params = {
                    'hidden_layers': layer_sizes,
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                    'activation': trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu']),
                    'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop']),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                    'epochs': 50,  # Réduit pour le tuning
                    'patience': 8
                }
                
                # Validation croisée temporelle
                tscv = TimeSeriesSplit(n_splits=3)
                val_scores = []
                
                for train_idx, valid_idx in tscv.split(X_train_scaled):
                    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[valid_idx]
                    y_tr, y_val = y_train.iloc[train_idx].values, y_train.iloc[valid_idx].values
                    
                    # Vérification des dimensions et conversion sécurisée
                    X_tr = np.array(X_tr, dtype=np.float32)
                    X_val = np.array(X_val, dtype=np.float32)
                    y_tr = np.array(y_tr, dtype=np.float32).reshape(-1)
                    y_val = np.array(y_val, dtype=np.float32).reshape(-1)
                    
                    if len(y_tr) < 5 or len(y_val) < 2:
                        val_scores.append(1e6)
                        continue
                    
                    try:
                        # Création des datasets PyTorch
                        train_dataset = TensorDataset(
                            torch.FloatTensor(X_tr), 
                            torch.FloatTensor(y_tr)
                        )
                        val_dataset = TensorDataset(
                            torch.FloatTensor(X_val), 
                            torch.FloatTensor(y_val)
                        )
                        
                        # Batch size adaptatif
                        effective_batch_size = min(params['batch_size'], len(train_dataset) // 2, 32)
                        
                        train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=effective_batch_size)
                        
                        # Modèle
                        model = DeepLearningRegressor(
                            input_dim=X_tr.shape[1],
                            hidden_layers=params['hidden_layers'],
                            dropout_rate=params['dropout_rate'],
                            activation=params['activation']
                        )
                        
                        # Optimiseur
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
                        
                        # Entraînement
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
            
            # Early stopping pour Optuna
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
            
            # Construction de l'architecture optimale
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
            else:  # 5 layers
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
                'epochs': 200,  # Rétablir pour l'entraînement final
                'patience': 20
            }
            
            print(f"Nouveaux meilleurs paramètres: {best_params}")
        
        # Entraînement du modèle final avec les meilleurs paramètres
        try:
            # Vérification des dimensions
            print(f"Dimensions: X_train={X_train_scaled.shape}, y_train={y_train.shape}")
            
            # Conversion sécurisée en numpy arrays
            X_train_array = np.array(X_train_scaled, dtype=np.float32)
            y_train_array = np.array(y_train.values, dtype=np.float32).reshape(-1)
            X_test_array = np.array(X_test_scaled, dtype=np.float32)
            y_test_array = np.array(y_test.values, dtype=np.float32).reshape(-1)
            
            # Vérification qu'on a assez de données
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
            
            # Split train/validation avec vérification de taille
            train_size = max(int(0.9 * len(train_dataset)), len(train_dataset) - 10)
            val_size = len(train_dataset) - train_size
            
            if val_size < 5:  # Si pas assez pour validation, pas de split
                train_subset = train_dataset
                val_subset = train_dataset  # Utilise train comme validation
            else:
                train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            
            # DataLoaders avec batch_size adaptatif
            effective_batch_size = min(best_params['batch_size'], len(train_subset) // 2, 32)
            
            train_loader = DataLoader(train_subset, batch_size=effective_batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=effective_batch_size)
            test_loader = DataLoader(test_dataset, batch_size=effective_batch_size)
            
            # Modèle final
            final_model = DeepLearningRegressor(
                input_dim=X_train_array.shape[1],
                hidden_layers=best_params['hidden_layers'],
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
            
            # Prédictions avec gestion des erreurs
            final_model.eval()
            test_predictions = []
            
            with torch.no_grad():
                for batch_idx, (batch_X, _) in enumerate(test_loader):
                    try:
                        batch_X = batch_X.to(device)
                        outputs = final_model(batch_X)
                        
                        # Conversion sécurisée en numpy
                        outputs_np = outputs.squeeze().cpu().numpy()
                        
                        # Gestion des différents cas de dimensions
                        if outputs_np.ndim == 0:
                            # C'est un scalaire
                            test_predictions.append(outputs_np.item())
                        elif outputs_np.ndim == 1:
                            # C'est un vecteur
                            test_predictions.extend(outputs_np.tolist())
                        else:
                            # Autre cas, on flatten
                            test_predictions.extend(outputs_np.flatten().tolist())
                            
                    except Exception as e:
                        print(f"Erreur dans prédiction batch {batch_idx}: {e}")
                        # Prédiction par défaut
                        batch_size = len(batch_X)
                        test_predictions.extend([y_train.mean()] * batch_size)
            
            # S'assurer qu'on a le bon nombre de prédictions
            if len(test_predictions) != len(y_test):
                print(f"Ajustement: {len(test_predictions)} prédictions pour {len(y_test)} échantillons")
                if len(test_predictions) > len(y_test):
                    test_predictions = test_predictions[:len(y_test)]
                else:
                    # Compléter avec la moyenne
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
            'final_epoch': training_info['final_epoch'],
            'best_val_loss': training_info['best_val_loss']
        }
        
        # Feature importance via permutation
        try:
            feature_importance = calculate_pytorch_feature_importance(
                final_model, X_test_scaled, y_test.values, device
            )
        except:
            feature_importance = np.ones(len(feature_columns)) / len(feature_columns)
        
        feature_imp = pd.DataFrame({
            'feature': feature_columns,
            'importance': feature_importance,
            'window': window_count + 1
        }).sort_values('importance', ascending=False)
        
        # Stockage des résultats
        results['predictions'].extend(test_pred)
        results['actual_values'].extend(y_test.values)
        results['metrics_by_window'].append(window_metrics)
        results['feature_importance_evolution'].append(feature_imp)
        results['best_params_evolution'].append({
            'window': window_count + 1,
            'params': best_params.copy()
        })
        results['dates'].extend(test_data[date_column].tolist())
        results['training_history'].append(training_info)
        
        print(f"Fenêtre {window_count + 1}: R² = {window_metrics['r2']:.4f}, RMSE = {window_metrics['rmse']:.6f}, Epochs = {training_info['final_epoch']}")
        
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
        'avg_val_loss': np.mean([m['best_val_loss'] for m in results['metrics_by_window'] if m['best_val_loss'] != float('inf')])
    }
    
    results['overall_metrics'] = overall_metrics
    results['model_type'] = 'PyTorch_DeepLearning'
    
    print(f"\n=== RÉSULTATS GLOBAUX DEEP LEARNING (PyTorch) ===")
    print(f"R² global: {overall_metrics['overall_r2']:.4f}")
    print(f"RMSE global: {overall_metrics['overall_rmse']:.6f}")
    print(f"R² moyen par fenêtre: {overall_metrics['avg_window_r2']:.4f}")
    print(f"Nombre de fenêtres: {overall_metrics['n_windows']}")
    print(f"Époques moyennes: {overall_metrics['avg_epochs']:.1f}")
    
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


# Exemple d'utilisation:
"""
# Installation nécessaire:
# pip install torch optuna scikit-learn pandas numpy

# Appel de la fonction
dl_results = sliding_window_dl_prediction(
    df=your_dataframe,
    date_column='date',
    target_column='MthRet',
    feature_columns=None,
    train_years=5,
    test_years=1,
    tune_frequency=4,
    n_trials=15,  # Moins d'essais car plus lent
    early_stopping_rounds=8
)

# Comparaison finale
print("\\n=== COMPARAISON FINALE DES MODÈLES ===")
models_comparison = [
    ('Random Forest', rf_results['overall_metrics']['overall_r2']),
    ('XGBoost', xgb_results['overall_metrics']['overall_r2']),
    ('Deep Learning', dl_results['overall_metrics']['overall_r2'])
]

for name, r2 in models_comparison:
    print(f"{name:15}: R² = {r2:.4f}")

best_model = max(models_comparison, key=lambda x: x[1])
print(f"\\n🏆 Meilleur modèle: {best_model[0]} (R² = {best_model[1]:.4f})")
"""


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


def implement_long_short_strategy(
    results: Dict[str, Any],
    df: pd.DataFrame,
    date_column: str = 'date',
    target_column: str = 'MthRet',
    ticker_column: str = 'Ticker',
    percentile_threshold: float = 0.1,
    rebalance_frequency: str = 'yearly'  # 'yearly', 'quarterly', 'monthly'
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
    elif rebalance_frequency == 'quarterly':
        strategy_df['period'] = strategy_df['date'].dt.year.astype(str) + '-Q' + strategy_df['date'].dt.quarter.astype(str)
    else:  # monthly
        strategy_df['period'] = strategy_df['date'].dt.to_period('M')
    
    portfolio_results = []
    
    print(f"=== STRATÉGIE LONG-SHORT ({percentile_threshold*100:.0f}% - {percentile_threshold*100:.0f}%) ===")
    print(f"Rebalancement: {rebalance_frequency}")
    print(f"Périodes analysées: {strategy_df['period'].nunique()}")
    
    for period in strategy_df['period'].unique():
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
        
        # Calculer les rendements de la stratégie
        long_return = long_positions['actual_return'].mean()
        short_return = short_positions['actual_return'].mean()
        
        # Rendement de la stratégie (long - short)
        strategy_return = long_return - short_return
        
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
        
        portfolio_results.append(period_result)
        
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
    
    results_dict = {
        'portfolio_results': portfolio_results,
        'overall_metrics': overall_metrics,
        'strategy_config': {
            'percentile_threshold': percentile_threshold,
            'rebalance_frequency': rebalance_frequency,
            'model_type': results.get('model_type', 'Unknown')
        }
    }
    
    print(f"\n=== RÉSULTATS GLOBAUX DE LA STRATÉGIE ===")
    print(f"Rendement moyen par période: {overall_metrics['avg_strategy_return']:.4f}")
    print(f"Rendement cumulé total: {overall_metrics['total_cumulative_return']:.4f}")
    print(f"Volatilité: {overall_metrics['volatility']:.4f}")
    print(f"Ratio de Sharpe: {overall_metrics['sharpe_ratio']:.4f}")
    print(f"Alpha moyen: {overall_metrics['avg_alpha']:.4f}")
    print(f"Taux de réussite: {overall_metrics['win_rate']:.2%}")
    
    return results_dict

def analyze_strategy_performance(strategy_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Analyse détaillée des performances de la stratégie
    """
    portfolio_results = strategy_results['portfolio_results']
    
    # Créer un DataFrame avec tous les résultats
    analysis_df = pd.DataFrame(portfolio_results)
    
    # Statistiques détaillées
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
    Visualise les résultats de la stratégie
    """
    try:
        portfolio_results = strategy_results['portfolio_results']
        
        # Extraire les données pour les graphiques
        periods = [str(r['period']) for r in portfolio_results]
        strategy_returns = [r['strategy_return'] for r in portfolio_results]
        long_returns = [r['long_return'] for r in portfolio_results]
        short_returns = [r['short_return'] for r in portfolio_results]
        alphas = [r['alpha'] for r in portfolio_results]
        
        # Calculer les rendements cumulés
        cumulative_strategy = np.cumprod([1 + r for r in strategy_returns])
        
        plt.figure(figsize=(16, 12))
        
        # Graphique 1: Rendements par période
        plt.subplot(2, 3, 1)
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
        plt.subplot(2, 3, 2)
        plt.plot(periods, cumulative_strategy, 'o-', linewidth=2, markersize=6, color='green')
        plt.title('Performance Cumulée de la Stratégie')
        plt.ylabel('Valeur du Portefeuille (base 1)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Graphique 3: Distribution des rendements
        plt.subplot(2, 3, 3)
        plt.hist(strategy_returns, bins=min(15, len(strategy_returns)//2), alpha=0.7, edgecolor='black')
        plt.axvline(x=np.mean(strategy_returns), color='red', linestyle='--', 
                   label=f'Moyenne: {np.mean(strategy_returns):.3f}')
        plt.title('Distribution des Rendements')
        plt.xlabel('Rendement de la Stratégie')
        plt.ylabel('Fréquence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graphique 4: Alpha par période
        plt.subplot(2, 3, 4)
        colors = ['green' if alpha > 0 else 'red' for alpha in alphas]
        plt.bar(periods, alphas, color=colors, alpha=0.7, edgecolor='black')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.title('Alpha par Période')
        plt.ylabel('Alpha')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Graphique 5: Corrélation Long vs Short
        plt.subplot(2, 3, 5)
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
        plt.subplot(2, 3, 6)
        metrics = ['Rendement\nMoyen', 'Volatilité', 'Ratio Sharpe', 'Win Rate']
        values = [
            np.mean(strategy_returns) * 100,
            np.std(strategy_returns) * 100,
            strategy_results['overall_metrics']['sharpe_ratio'],
            strategy_results['overall_metrics']['win_rate'] * 100
        ]
        
        bars = plt.bar(metrics, values, color=['blue', 'orange', 'green', 'purple'], alpha=0.7)
        plt.title('Métriques Clés (%)')
        plt.ylabel('Valeur')
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
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
