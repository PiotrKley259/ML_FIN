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