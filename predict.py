# -------------------- Imports --------------------
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
import datetime
warnings.filterwarnings('ignore')
def predict_stock(file_path):
    # Load your CSV
    data = pd.read_csv(file_path, index_col='Date', parse_dates=True)

    # Clean the Volume column
    data['Volume'] = data['Volume'].replace(',', '', regex=True).astype(float)

    # Now safe to compute rolling mean
    data['VMA10'] = data['Volume'].rolling(window=10).mean()
    stock_prices = data.copy()

    # -------------------- Feature Engineering --------------------
    # Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()

    # Volume-Based Features
    data['VMA10'] = data['Volume'].rolling(window=10).mean()
    data['Volume_Ratio'] = data['Volume'] / data['VMA10']

    # Returns and Volatility
    data['Daily_Return'] = data['Close'].pct_change()
    data['Volatility_10'] = data['Daily_Return'].rolling(window=10).std()

    # Momentum
    data['Momentum_10'] = data['Close'] - data['Close'].shift(10)

    # Price Difference from Moving Averages
    data['Price_SMA20_Diff'] = data['Close'] - data['SMA_20']
    data['Price_EMA20_Diff'] = data['Close'] - data['EMA_20']


    # RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
    data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()

    # ATR (Average True Range)
    data['H-L'] = data['High'] - data['Low']
    data['H-PC'] = abs(data['High'] - data['Close'].shift())
    data['L-PC'] = abs(data['Low'] - data['Close'].shift())
    data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)  # Removed the extra space before 'H-PC'
    data['ATR'] = data['TR'].rolling(window=14).mean()

    # OBV (On Balance Volume)
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i - 1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i - 1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    data['OBV'] = obv

    # Stochastic Oscillator
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    data['%K'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
    data['%D'] = data['%K'].rolling(window=3).mean()

    # Drop rows with NaN values created by indicators
    data.dropna(inplace=True)

    # Create the target column (1 for 'Up' and 0 for 'Down')
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    # Add lag features for previous n days' returns
    def add_lag_features(df, lags=[1, 2, 3]):
        df = df.copy()
        df['Return'] = df['Close'].pct_change()

        for lag in lags:
            df[f'Return_Lag{lag}'] = df['Return'].shift(lag)

        return df

    data = add_lag_features(data)



    # Define features (X) and target (y) — use only the columns that exist
    X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_20', 'VMA10',
            'Volume_Ratio', 'Daily_Return', 'Volatility_10', 'Momentum_10', 'Price_SMA20_Diff', 'Price_EMA20_Diff',
            'RSI', 'MACD', 'Signal_Line', 'ATR', 'OBV']].copy()

    y = data['Target'].copy()

    # Display the first few rows to confirm everything looks good
    print(X.head())
    print(y.head())

    X_train, X_test = X.iloc[:int(len(X)*0.8)], X.iloc[int(len(X)*0.8):]
    y_train, y_test = y.iloc[:int(len(y)*0.8)], y.iloc[int(len(y)*0.8):]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Predict on test data
    xgb_model = XGBClassifier( eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train_scaled, y_train)


    # Train LightGBM model
    lgbm_model = LGBMClassifier(random_state=42, verbosity=-1)
    lgbm_model.fit(X_train_scaled, y_train)

    # Predict on test data
    lgbm_preds = lgbm_model.predict(X_test_scaled)


    # Tuned XGBoost model with some better hyperparameters
    tuned_xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    tuned_xgb.fit(X_train_scaled, y_train)

    # Define the LightGBM model
    lgb_model = LGBMClassifier(random_state=42)

    # Fit the model on the training data
    lgb_model.fit(X_train_scaled, y_train)

    # Define individual classifiers
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('lgb', lgb_model)
        ],
        voting='hard'
    )

    # Fit on training data
    voting_clf.fit(X_train_scaled, y_train)

    # Define the parameter grid for XGBoost
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 150]
    }

    # Instantiate XGBoost model
    xgb = XGBClassifier()

    # Perform GridSearchCV for hyperparameter tuning
    xgb_grid = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)

    # Fit the model to the training data
    xgb_grid.fit(X_train_scaled, y_train)

    # Get the best model
    xgb_tuned = xgb_grid.best_estimator_

    # Define base models for stacking
    base_models = [
        ('rf', rf_model),  # Random Forest
        ('xgb', xgb_model),  # XGBoost
        ('lgb', lgb_model)  # LightGBM
    ]

    # Define the final estimator (meta-model) as Logistic Regression
    meta_model = LogisticRegression()

    # Create the stacking classifier
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

    # Fit the stacking model
    stacking_model.fit(X_train_scaled, y_train)

    # Predict using the stacking model
    y_pred = stacking_model.predict(X_test_scaled)

    # Evaluate each model
    models = {
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "LightGBM": lgb_model,
        "Tuned XGBoost": xgb_tuned,
        "Stacking": stacking_model
    }

    # Define a function to calculate strategy return
    def calculate_strategy_return(y_pred, stock_prices):
        # Get price change for next day
        price_change = stock_prices['Close'].shift(-1) - stock_prices['Close']

        # Only consider returns where prediction = 1 (i.e., buy signal)
        strategy_return = (y_pred == 1) * price_change

        # Return sum of strategy return (total profit/loss from all predicted trades)
        return strategy_return.sum()

    # Define thresholds to test
    thresholds = np.arange(0.40, 1.00, 0.05)

    # Store results
    results = []

    # Loop over each model
    for model_name, model in models.items():
        print(f"\n🔍 Evaluating {model_name}...")

        # Predict probabilities for positive class (class = 1)
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

        # Track best threshold and accuracy
        best_threshold = 0
        best_accuracy = 0

        # Loop over thresholds to find the best one
        for threshold in thresholds:
            y_pred = (y_pred_prob >= threshold).astype(int)
            acc = accuracy_score(y_test, y_pred)

            if acc > best_accuracy:
                best_accuracy = acc
                best_threshold = threshold

        # Apply best threshold for final prediction
        y_pred_best = (y_pred_prob >= best_threshold).astype(int)

        # Print classification report
        print(f"✅ Best Threshold for {model_name}: {best_threshold:.2f}")
        print(f"📊 Accuracy at Best Threshold: {best_accuracy:.4f}")
        print("🧾 Classification Report:")
        print(classification_report(y_test, y_pred_best))

        # 🛠 Ensure both y_test and stock_prices have timezone-neutral datetime indexes
        stock_prices.index = stock_prices.index.tz_localize(None)
        y_test.index = y_test.index.tz_localize(None)
        aligned_prices = stock_prices.loc[y_test.index]


        # Check for missing values
        if aligned_prices.isnull().values.any():
            raise ValueError("Stock prices contain NaN values after reindexing. Check your data alignment.")

        # 📊 Calculate strategy return using aligned prices
        strategy_return = calculate_strategy_return(y_pred_best, aligned_prices)
        print(f"💰 Strategy Return: {strategy_return:.2f}")

        # Save results for summary
        results.append({
            "Model": model_name,
            "Best Threshold": best_threshold,
            "Accuracy": best_accuracy,
            "Strategy Return": strategy_return
        })

    # 📋 Print summary table
    summary_df = pd.DataFrame(results)
    print("\n📈 Summary of Model Performances (Sorted by Strategy Return):")
    print(summary_df.sort_values(by="Strategy Return", ascending=False))

    # Identify best model and threshold dynamically from your summary table
    best_model_info = summary_df.sort_values(by='Strategy Return', ascending=False).iloc[0]
    best_model_name = best_model_info['Model']
    best_threshold = best_model_info['Best Threshold']

    # Select the actual trained model object
    model_map = {
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'Tuned XGBoost': xgb_tuned,
        'LightGBM': lgb_model,
        'Stacking': stacking_model
    }
    best_model = model_map[best_model_name]

    if best_model is None or best_threshold is None:
        raise ValueError("Best model or threshold is not defined. Ensure model selection logic is correct.")

    # Adjust the threshold for predictions to a lower value (e.g., 0.4)
    threshold = 0.2  # Experiment with values like 0.4, 0.3, etc.

    # Update the backtesting logic to reflect the new threshold
    final_model = best_model  # Use the best-performing model (e.g., XGBoost, LightGBM)

    # Get predicted probabilities (prob of class 1)
    y_proba_final = final_model.predict_proba(X_test_scaled)[:, 1]

    # Before assigning y_test to test_data, ensure it's 1D
    # y_test = y_test.ravel()  # Flatten y_test if it's 2D



    
    def confidence_weighted_backtest_with_filter(model, X, y, data, thresholds=best_threshold, expected_return_col='Return_Lag1', n_splits=5):
        fold_size = len(X) // n_splits
        results = []

        for fold in range(n_splits):
            start = fold * fold_size
            end = (fold + 1) * fold_size if fold < n_splits - 1 else len(X)

            X_train, y_train = X[:start], y[:start]
            X_test, y_test = X[start:end], y[start:end]
            test_index = y_test.index

            if len(X_train) == 0:  # Skip first fold if no training data
                continue

            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]  # Get predicted probabilities for class 1 (buy signal)

            # Apply confidence thresholding based on provided thresholds
            threshold_low, threshold_high = thresholds
            confident_buy = (probs > threshold_high)  # Confident buy signal when probability > threshold_high
            confident_sell = (probs < threshold_low)  # Confident sell signal when probability < threshold_low

            # Filter: Only act if expected return also agrees with signal
            expected_returns = data.loc[test_index, expected_return_col].values  # Get expected returns for test set
            expected_return_filter = expected_returns > 0  # Expected return filter (buy only when expected return is positive)
            
            # Modify the condition for buy and sell signals based on expected return
            confident_buy &= expected_return_filter  # Only consider buy signals when expected return is positive
            confident_sell &= (expected_returns < 0)  # Only consider sell signals when expected return is negative

            # Calculate daily returns from price (shifted to get next day's return)
            daily_returns = data['Close'].pct_change().shift(-1).loc[test_index].values

            # Initialize signals: 1 for buy, -1 for sell, 0 for no action
            signals = np.zeros_like(probs)
            signals[confident_buy] = 1  # Buy signal where confident_buy is True
            signals[confident_sell] = -1  # Sell signal where confident_sell is True

            # Calculate strategy returns based on signals and actual returns
            strategy_returns = signals * daily_returns
            total_return = np.nansum(strategy_returns) * 100  # Total return in percentage
            sharpe = np.nanmean(strategy_returns) / (np.nanstd(strategy_returns) + 1e-9)  # Sharpe ratio (risk-adjusted return)

            print(f"📅 Fold {fold+1}: Strategy Return = {total_return:.2f}%, Sharpe Ratio = {sharpe:.2f}")

            # Store fold results
            results.append({
                'Fold': fold + 1,
                'Strategy Return': total_return,
                'Sharpe Ratio': sharpe
            })

        # Create DataFrame for results
        summary = pd.DataFrame(results)
        summary['Cumulative Return'] = summary['Strategy Return'].cumsum()  # Cumulative return for all folds

        print("\n✅ Improved Confidence-Filtered Backtesting Complete.\n")
        print("📈 Backtest Summary:")
        print(summary)

        return summary



    confidence_weighted_backtest_with_filter(best_model, X, y, data, thresholds=(0.4, 0.85), expected_return_col='Return_Lag1', n_splits=5)
    # Ensure datetime index
    print(best_model_name)

    print("THIS IS MY MIND")

    # Assuming `best_model` is the trained model

    # Step 1: Get the features used by the best model
    if hasattr(best_model, 'feature_names_'):
        print("Features used in the best model:")
        model_features = best_model.feature_names_

    elif hasattr(best_model, 'feature_importances_'):
        print("Features used in the best model:")
        model_features = X.columns.tolist()

    else:
        print("Feature names used in the best model:")
        model_features = X.columns.tolist()
    print("TILL HERE !")

    # Load data
    # Ensure timezone-aware datetime index if your model/data expects tz-aware
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')  # or your specific tz

    # Normalize today in same timezone
    today = pd.to_datetime('today').normalize().tz_localize(data.index.tz)

    # Exclude today's data — only use up to yesterday
    data = data[data.index < today]

    # Split features and target
    X = data.drop(columns=['Target'])
    y = data['Target']

    # Get the last available date in data (today)
    last_date = data.index[-1]
    tomorrow = last_date + datetime.timedelta(days=1)

    # Assume model_features is a list of feature names your best_model expects
    latest_features = data.iloc[-1][model_features].to_frame().T

    # Add any missing columns as zero
    missing_cols = set(model_features) - set(latest_features.columns)
    for col in missing_cols:
        latest_features[col] = 0

    # Align column order
    latest_features = latest_features[model_features]

    # Predict probability for class 1
    proba = best_model.predict_proba(latest_features)[0][1]

    # Use your best threshold to classify
    label = 1 if proba >= best_threshold else 0
    signal = "Buy" if label == 1 else "Sell"

    print(f"📅 Last Data Date: {last_date.date()} | 📈 Prediction for {tomorrow.date()}: {signal} (Confidence: {proba:.2f})")

    return signal, proba, tomorrow.date(), data
