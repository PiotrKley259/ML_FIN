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

# Unmodified FlexibleMLP class
class FlexibleMLP(nn.Module):
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

# Unmodified train_model function
def train_model(num_epochs: int,
                train_loader: torch.utils.data.DataLoader,
                criterion,
                optimizer,
                model: torch.nn.Module,
                ridge_penalty: float = 0.001):
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets) + ridge_penalty * sum(p.pow(2.0).sum() for p in model.parameters())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Define predict_returns_for_ticker function
def predict_returns_for_ticker(ticker_data, ticker, epochs=50, batch_size=32, num_seeds=10):
    # Prepare features and target
    X = ticker_data.drop(columns=['MthCalDt', 'sprtrn', 'Ticker', 'MthRet']).fillna(0)
    y = ticker_data['MthRet'].fillna(0)
    dates = ticker_data['MthCalDt']
    sprtrn = ticker_data['sprtrn'].fillna(0)

    # Split data
    X_train, X_test, y_train, y_test, dates_train, dates_test, sprtrn_train, sprtrn_test = train_test_split(
        X, y, dates, sprtrn, test_size=0.2, random_state=42, shuffle=False
    )

    # Debug: Check input shapes
    print(f"Ticker {ticker}: ticker_data shape: {ticker_data.shape}, X_test shape: {X_test.shape}, n_test_samples: {len(y_test)}")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and train with different seeds
    input_size = X_train.shape[1]
    train_predictions_all = []
    test_predictions_all = []
    n_test_samples = X_test.shape[0]
    for seed in range(num_seeds):
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize model
        model = FlexibleMLP(layers=[input_size, 64, 32, 1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train model
        train_model(
            num_epochs=epochs,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            model=model,
            ridge_penalty=0.001
        )

        # Generate predictions
        model.eval()
        with torch.no_grad():
            train_predictions = model(X_train_tensor).squeeze().numpy()
            test_predictions = model(X_test_tensor).squeeze().numpy()
            # Handle single-sample case
            if n_test_samples == 1:
                test_predictions = np.array([test_predictions])
            # Validate prediction shape
            if test_predictions.shape != (n_test_samples,):
                raise ValueError(f"Ticker {ticker}: test_predictions has shape {test_predictions.shape}, expected ({n_test_samples},)")
            train_predictions_all.append(train_predictions)
            test_predictions_all.append(test_predictions)

    # Debug: Check predictions list
    print(f"Ticker {ticker}: test_predictions_all length: {len(test_predictions_all)}, first prediction shape: {test_predictions_all[0].shape if test_predictions_all else 'empty'}")

    # Convert predictions to arrays with shape (n_samples, num_seeds)
    try:
        train_predictions_all = np.stack(train_predictions_all, axis=1)  # Shape: (n_train_samples, 10)
        test_predictions_all = np.stack(test_predictions_all, axis=1)    # Shape: (n_test_samples, 10)
    except ValueError as e:
        print(f"Ticker {ticker}: Error stacking predictions: {e}")
        raise

    # Debug: Check stacked shape
    print(f"Ticker {ticker}: test_predictions_all stacked shape: {test_predictions_all.shape}")

    # Average predictions across seeds
    train_predictions_avg = train_predictions_all.mean(axis=1)
    test_predictions_avg = test_predictions_all.mean(axis=1)

    # Combine train and test predictions
    all_predictions = np.concatenate([train_predictions_avg, test_predictions_avg])
    all_actual = np.concatenate([y_train.values, y_test.values])
    all_dates = pd.concat([dates_train.reset_index(drop=True), dates_test.reset_index(drop=True)])

    # Create DataFrame for predictions
    predictions_df = pd.DataFrame({
        'Date': all_dates,
        'Actual_MthRet': all_actual,
        'Predicted_MthRet': all_predictions
    })

    # Compute timed returns and market comparison using mean predictions
    test_targets = y_test.values
    test_dates = dates_test
    sprtrn = sprtrn_test.values

    # Ensure shapes are correct
    timed_returns = test_targets * test_predictions_avg  # Shape: (n_test_samples,)
    together = pd.DataFrame({
        'Actual_MthRet': test_targets,
        'Timed_Returns': timed_returns,
        'Market_Returns': sprtrn
    }, index=test_dates)

    def sharpe_ratio(returns):
        return np.mean(returns) / np.std(returns) * np.sqrt(12) if np.std(returns) != 0 else 0

    sharpe_ratios = together.apply(sharpe_ratio)
    cumulative_returns = together.cumsum()

    market_sharpe = sharpe_ratios['Market_Returns']
    market_cumulative = cumulative_returns['Market_Returns'].iloc[-1]

    performance_summary = pd.DataFrame({
        'Metric': ['Sharpe_Ratio', 'Cumulative_Return', 'Beats_Market_Sharpe', 'Beats_Market_Cumulative'],
        'Value': [
            sharpe_ratios['Timed_Returns'],
            cumulative_returns['Timed_Returns'].iloc[-1],
            sharpe_ratios['Timed_Returns'] > market_sharpe,
            cumulative_returns['Timed_Returns'].iloc[-1] > market_cumulative
        ]
    })

    (together / together.std()).cumsum().plot(figsize=(10, 6))
    plt.title(f'{ticker} Model vs Market (Market Sharpe: {market_sharpe:.2f})')
    plt.xlabel('Date')
    plt.ylabel('Standardized Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{ticker}_cumulative_returns.png'))
    plt.close()

    # Save performance summary and predictions
    performance_summary.to_csv(os.path.join(output_dir, f'{ticker}_performance_summary.csv'), index=False)
    predictions_df.to_csv(os.path.join(output_dir, f'{ticker}_predictions.csv'), index=False)

    return {
        'ticker': ticker,
        'mse': mean_squared_error(test_targets, test_predictions_avg),
        'performance_summary': performance_summary,
        'predictions_df': predictions_df
    }

# Load and preprocess data
folder = os.path.expanduser('~/Desktop/Machine Learning/data_ML_Project')
folder1 = os.path.expanduser('~/Desktop/Machine Learning/data_filtered')

main_csv = os.path.join(folder, 'monthly_crsp.csv')
jkp_csv = os.path.join(folder1, 'JKP_filtered.csv')
zimmer_csv = os.path.join(folder1, 'Zimmer_filtered.csv')

if os.path.exists(folder):
    data_main = pd.read_csv(main_csv)
    data_jkp = pd.read_csv(jkp_csv)
    data_zimmer = pd.read_csv(zimmer_csv)
else:
    raise FileNotFoundError("Data folder not found!")

# Preprocess data
data_main['date'] = pd.to_datetime(data_main['MthCalDt'])
data_jkp['date'] = pd.to_datetime(data_jkp['date'])
data_zimmer['date'] = pd.to_datetime(data_zimmer['date'], errors='coerce')

start_date = pd.to_datetime('1986-01-01')
data_main = data_main[data_main['date'] >= start_date]
data_jkp = data_jkp[data_jkp['date'] >= start_date]
data_zimmer = data_zimmer[data_zimmer['date'] >= start_date]

for df in [data_main, data_jkp, data_zimmer]:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

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

# Filter tickers with at least 300 records
merged = merged[merged.groupby('Ticker')['Ticker'].transform('count') >= 300]

# Drop unnecessary columns
columns_to_drop = ['PERMNO', 'HdrCUSIP', 'CUSIP', 'TradingSymbol', 'PERMCO', 'SICCD', 'NAICS', 'date', 'year', 'month']
merged.drop(columns=columns_to_drop, inplace=True)

# Shift MthRet and sprtrn
merged[['MthRet', 'sprtrn']] = merged.groupby('Ticker')[['MthRet', 'sprtrn']].shift(1).fillna(0)

# Create output directory for predictions
output_dir = os.path.expanduser('~/Desktop/Machine Learning/predictions')
os.makedirs(output_dir, exist_ok=True)

# Compute correlations
tickers = merged['Ticker'].unique()

# 1. Correlation between MthRet and sprtrn for each ticker
correlation_results = []
for ticker in tqdm(tickers, desc="Computing MthRet-sprtrn correlations"):
    ticker_data = merged[merged['Ticker'] == ticker].copy()
    # Compute correlation on non-null pairs
    valid_data = ticker_data[['MthRet', 'sprtrn']].dropna()
    num_obs = len(valid_data)
    if num_obs < 2:
        correlation = np.nan
    else:
        correlation = valid_data['MthRet'].corr(valid_data['sprtrn'])
    correlation_results.append({
        'Ticker': ticker,
        'Correlation': correlation,
        'Num_Observations': num_obs
    })

# Create MthRet-sprtrn correlation DataFrame and save to CSV
correlation_df = pd.DataFrame(correlation_results)
correlation_df.to_csv(os.path.join(output_dir, 'ticker_sprtrn_correlations.csv'), index=False)
print("\nCorrelation between MthRet and sprtrn saved to ticker_sprtrn_correlations.csv")
print(correlation_df.head())

# 2. Pairwise correlations between stocks' MthRet
# Pivot data to align MthRet by date
pivot_mthret = merged.pivot_table(index='MthCalDt', columns='Ticker', values='MthRet')
# Compute correlation matrix
corr_matrix = pivot_mthret.corr()
# Compute number of overlapping observations for each pair
count_matrix = pivot_mthret.notnull().astype(int).T.dot(pivot_mthret.notnull().astype(int))

pairwise_results = []
for i, ticker1 in enumerate(tickers):
    for j, ticker2 in enumerate(tickers[i+1:], start=i+1):  # Skip self and duplicates
        correlation = corr_matrix.loc[ticker1, ticker2]
        num_obs = count_matrix.loc[ticker1, ticker2]
        if num_obs < 2:
            correlation = np.nan
        pairwise_results.append({
            'Ticker1': ticker1,
            'Ticker2': ticker2,
            'Correlation': correlation,
            'Num_Observations': num_obs
        })

# Create pairwise correlation DataFrame and save to CSV
pairwise_df = pd.DataFrame(pairwise_results)
pairwise_df.to_csv(os.path.join(output_dir, 'stock_pairwise_correlations.csv'), index=False)
print("\nPairwise correlations between stocks' MthRet saved to stock_pairwise_correlations.csv")
print(pairwise_df.head())

# Filter high correlations (|Correlation| >= 0.8)
high_corr_df = pairwise_df[pairwise_df['Correlation'].abs() >= 0.8]
high_corr_df.to_csv(os.path.join(output_dir, 'stock_pairwise_high_correlations.csv'), index=False)
print("\nHigh correlations (|Correlation| >= 0.8) savedけto stock_pairwise_high_correlations.csv")
print(f"Number of high-correlation pairs: {len(high_corr_df)}")
print(high_corr_df)

# Get unique tickers from high-correlation pairs
high_corr_tickers = pd.concat([high_corr_df['Ticker1'], high_corr_df['Ticker2']]).unique()
print(f"\nNumber of unique tickers in high-correlation pairs: {len(high_corr_tickers)}")

# Run predictions only for high-correlation tickers
results = []
if len(high_corr_tickers) > 0:
    for ticker in tqdm(high_corr_tickers, desc="Predicting returns for high-correlation tickers"):
        ticker_data = merged[merged['Ticker'] == ticker].copy()
        if len(ticker_data) < 100:  # Increased threshold to skip problematic tickers
            print(f"Skipping ticker {ticker}: insufficient data ({len(ticker_data)} records)")
            continue
        try:
            result = predict_returns_for_ticker(ticker_data, ticker)
            results.append(result)
        except Exception as e:
            print(f"Error processing ticker {ticker}: {e}")
else:
    print("\nNo high-correlation pairs found. Skipping predictions and trading strategy.")

# Aggregate results
results_df = pd.DataFrame([
    {'ticker': r['ticker'], 'mse': r['mse']}
    for r in results
])

# Summary statistics
if not results_df.empty:
    print("\nSummary Statistics for Test Set MSE:")
    print(results_df['mse'].describe())
else:
    print("\nNo prediction results to summarize.")

# Plot actual vs predicted returns for each ticker
if results:
    for result in results:
        ticker = result['ticker']
        pred_df = result['predictions_df']
        plt.figure(figsize=(10, 6))
        plt.plot(pred_df['Date'], pred_df['Actual_MthRet'], label='Actual MthRet', alpha=0.7)
        plt.plot(pred_df['Date'], pred_df['Predicted_MthRet'], label='Predicted MthRet', alpha=0.7)
        plt.title(f'Actual vs Predicted Monthly Returns for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Monthly Return')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{ticker}_actual_vs_predicted.png'))
        plt.close()

# Pairs trading strategy for high-correlation pairs
# Load high-correlation pairs
high_corr_df = pd.read_csv(os.path.join(output_dir, 'stock_pairwise_high_correlations.csv'))

# Function to compute z-score and trading signals
def pairs_trading_strategy(ticker1, ticker2, predictions_dict, merged_data, window=12, entry_threshold=2, exit_threshold=1):
    # Get predictions for both tickers
    pred_df1 = predictions_dict.get(ticker1)
    pred_df2 = predictions_dict.get(ticker2)
    if pred_df1 is None or pred_df2 is None:
        print(f"Skipping pair {ticker1}-{ticker2}: Predictions not available")
        return []

    # Merge predictions and actual returns
    data1 = merged_data[merged_data['Ticker'] == ticker1][['MthCalDt', 'MthRet']].merge(
        pred_df1[['Date', 'Predicted_MthRet']], left_on='MthCalDt', right_on='Date', how='inner'
    ).rename(columns={'MthRet': 'MthRet1', 'Predicted_MthRet': 'Pred_MthRet1'}).drop(columns='Date')

    data2 = merged_data[merged_data['Ticker'] == ticker2][['MthCalDt', 'MthRet']].merge(
        pred_df2[['Date', 'Predicted_MthRet']], left_on='MthCalDt', right_on='Date', how='inner'
    ).rename(columns={'MthRet': 'MthRet2', 'Predicted_MthRet': 'Pred_MthRet2'}).drop(columns='Date')

    # Align data by date
    pair_data = data1.merge(data2, on='MthCalDt', how='inner')
    if len(pair_data) < window + 1:
        print(f"Skipping pair {ticker1}-{ticker2}: Insufficient overlapping data ({len(pair_data)} records)")
        return []

    # Compute spread
    pair_data['Spread'] = pair_data['MthRet1'] - pair_data['MthRet2']

    # Compute rolling mean and std of spread
    pair_data['Spread_Mean'] = pair_data['Spread'].rolling(window=window, min_periods=window).mean()
    pair_data['Spread_Std'] = pair_data['Spread'].rolling(window=window, min_periods=window).std()

    # Compute z-score
    pair_data['Z_Score'] = (pair_data['Spread'] - pair_data['Spread_Mean']) / pair_data['Spread_Std']

    # Filter for test period (last 20% of dates)
    test_size = 0.2
    test_start_idx = int(len(pair_data) * (1 - test_size))
    pair_data = pair_data.iloc[test_start_idx:].copy()

    # Generate trading signals
    trades = []
    position = None  # (ticker_long, ticker_short)
    entry_date = None
    for idx, row in pair_data.iterrows():
        z_score = row['Z_Score']
        date = row['MthCalDt']
        pred1 = row['Pred_MthRet1']
        pred2 = row['Pred_MthRet2']
        ret1 = row['MthRet1']
        ret2 = row['MthRet2']

        # Skip if z-score is NaN (e.g., insufficient window)
        if pd.isna(z_score):
            continue

        # Close position if z-score returns to normal
        if position and -exit_threshold <= z_score <= exit_threshold:
            ticker_long, ticker_short = position
            trade_return = ret1 if ticker_long == ticker1 else -ret1
            trade_return += -ret2 if ticker_short == ticker2 else ret2
            trades.append({
                'Ticker1': ticker1,
                'Ticker2': ticker2,
                'Trade_Date': date,
                'Z_Score': z_score,
                'Action': 'Close',
                'Return': trade_return
            })
            position = None
            entry_date = None
            continue

        # Open position if z-score exceeds threshold
        if not position:
            if z_score > entry_threshold:
                # Spread is too high: buy stock predicted to go up, short stock predicted to go down
                ticker_long = ticker1 if pred1 > pred2 else ticker2
                ticker_short = ticker2 if pred1 > pred2 else ticker1
                action = 'Open (Buy {}, Short {})'.format(ticker_long, ticker_short)
                position = (ticker_long, ticker_short)
                entry_date = date
                trades.append({
                    'Ticker1': ticker1,
                    'Ticker2': ticker2,
                    'Trade_Date': date,
                    'Z_Score': z_score,
                    'Action': action,
                    'Return': 0.0
                })
            elif z_score < -entry_threshold:
                # Spread is too low: buy stock predicted to go down, short stock predicted to go up
                ticker_long = ticker1 if pred1 < pred2 else ticker2
                ticker_short = ticker2 if pred1 < pred2 else ticker1
                action = 'Open (Buy {}, Short {})'.format(ticker_long, ticker_short)
                position = (ticker_long, ticker_short)
                entry_date = date
                trades.append({
                    'Ticker1': ticker1,
                    'Ticker2': ticker2,
                    'Trade_Date': date,
                    'Z_Score': z_score,
                    'Action': action,
                    'Return': 0.0
                })

    return trades

# Create dictionary of predictions
predictions_dict = {r['ticker']: r['predictions_df'] for r in results}

# Run pairs trading strategy
trading_results = []
for _, row in high_corr_df.iterrows():
    ticker1 = row['Ticker1']
    ticker2 = row['Ticker2']
    trades = pairs_trading_strategy(ticker1, ticker2, predictions_dict, merged, window=12, entry_threshold=2, exit_threshold=1)
    trading_results.extend(trades)

output_dir1 = os.path.expanduser('~/Desktop/Machine Learning/trades')
os.makedirs(output_dir1, exist_ok=True) 
# Save trading results
if trading_results:
    trading_df = pd.DataFrame(trading_results)
    trading_df.to_csv(os.path.join(output_dir1, 'pairs_trading_results.csv'), index=False)
    print("\nPairs trading results saved to pairs_trading_results.csv")
    print(f"Total trades: {len(trading_df)}")
    print(f"Total return: {trading_df['Return'].sum():.4f}")
    print(trading_df.head())

    # Plot cumulative returns
    trading_df['Cumulative_Return'] = trading_df['Return'].cumsum()
    plt.figure(figsize=(10, 6))
    plt.plot(trading_df['Trade_Date'], trading_df['Cumulative_Return'], label='Cumulative Return')
    plt.title('Pairs Trading Strategy Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir1, 'pairs_trading_cumulative_returns.png'))
    plt.close()
else:
    print("\nNo trades generated from pairs trading strategy")