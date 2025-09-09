        
# --- STEP 1: Generate Synthetic Stock Data ---

def simulated_stock_data(days=365):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime, timedelta

    """
    Generates synthetic OHLCV data with occasional anomalies.
    Returns: DataFrame with columns [Date, Open, High, Low, Close, Volume, Trade_Value]
    """
    np.random.seed(42)

    dates = [datetime.now() - timedelta(days=i) for i in range(days)][::-1]
    base_price = np.cumsum(np.random.normal(0, 1, days)) + 100  # Random walk starting at $100
    
    # Generate OHLC (with some randomness)
    opens = base_price + np.random.uniform(-0.5, 0.5, days)
    highs = opens + np.abs(np.random.normal(0.5, 0.2, days))
    lows = opens - np.abs(np.random.normal(0.5, 0.2, days))
    closes = opens + np.random.normal(0, 0.3, days)
    
    # Volume & Trade Value (correlated with price movements)
    volume = np.random.poisson(1000, days) + (np.abs(np.random.normal(0, 200, days)))
    trade_value = volume * closes + np.random.normal(0, 5000, days)
    
    # Inject 5% anomalies (sudden spikes/drops)
    anomaly_indices = np.random.choice(days, size=int(0.05*days), replace=False)
    closes[anomaly_indices] *= np.random.choice([1.2, 0.8], len(anomaly_indices))  # +/-20%
    volume[anomaly_indices] *= np.random.choice([3, 0.3], len(anomaly_indices))   # 3x or 70% drop
    
    return pd.DataFrame({
        'Date': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volume,
        'Trade_Value': trade_value
        
    })

#________________________________________________________________________________________________________________________________________________________________________________________

# 2. IQR-Based Anomaly Detection
# we'll detect anomalies in Close Price and Volume.
# --- STEP 2: IQR Anomaly Detection ---
def detect_anomalies_iqr(data, feature, threshold=1.5):
    """
    Detects anomalies using Interquartile Range (IQR).
    Args:
        data: DataFrame
        feature: Column name to analyze (e.g., 'Close')
        threshold: Multiplier for IQR (default: 1.5)
    Returns:
        Series with anomaly flags (True = Anomaly)
    """
    q1 = data[feature].quantile(0.25)
    q3 = data[feature].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    anomalies = (data[feature] < lower_bound) | (data[feature] > upper_bound)
    return anomalies
#_______________________________________________________________________________________________________________________________________________________________________________________

def plot_function(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    plt.figure(figsize=(14, 8))

# Plot Close Price with Anomalies
    plt.subplot(2, 1, 1)
    sns.lineplot(data=df, x='Date', y='Close', label='Close Price')
    anomaly_dates = df[df['Close_Anomaly']]['Date']
    anomaly_prices = df[df['Close_Anomaly']]['Close']
    plt.scatter(anomaly_dates, anomaly_prices, color='red', label='Anomaly')
    plt.title("Close Price Anomalies (IQR Method)")
    plt.legend()

    # Plot Volume with Anomalies
    plt.subplot(2, 1, 2)
    sns.lineplot(data=df, x='Date', y='Volume', label='Volume')
    anomaly_dates = df[df['Volume_Anomaly']]['Date']

    anomaly_volumes = df[df['Volume_Anomaly']]['Volume']
    plt.scatter(anomaly_dates, anomaly_volumes, color='red', label='Anomaly')
    plt.title("Volume Anomalies (IQR Method)")
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    
#______________________________________________________________________________________________________________________________________________________________________________________

# --- STEP 4: Risk Scoring ---
def calculate_risk_score(data, window=7):
    """
    Calculates a rolling risk score (0-10) based on recent anomalies.
    Args:
        data: DataFrame with anomaly flags
        window: Lookback window in days
    Returns:
        Series with risk scores
    """
    # Count anomalies in rolling window
    data['Recent_Close_Anomalies'] = data['Close_Anomaly'].rolling(window).sum()
    data['Recent_Volume_Anomalies'] = data['Volume_Anomaly'].rolling(window).sum()
    
    # Normalize to 0-10 scale
    max_anomalies = window  # Worst case: anomalies every day
    data['Risk_Score'] = (
        (data['Recent_Close_Anomalies'] + data['Recent_Volume_Anomalies']) / (2 * max_anomalies)
    ) * 10
    
    return data
#________________________________________________________________________________________________________________________________________________________________________________________
#making function
# --- STEP 5: Export Results ---
# Save to CSV _ or _ SQL

def save_db_csv(df,csv=True,sql=True,summary=True):

    import pandas as pd
           
    if csv:

        df.to_csv("anomaly_detection_results.csv", index=False)
    if sql :
        import pandas as pd
        import sqlite3
        # Create a connection to SQLite database
        conn = sqlite3.connect('anomaly_file.db')
        
        # Convert DataFrame to SQL table
        df.to_sql('anomaly_table', conn, if_exists='replace', index=False)

        # Print summary
    if summary:
        print("\n=== Anomaly Detection Report ===")
        print(f"Total Anomalies Detected: {len(df[df['Close_Anomaly'] | df['Volume_Anomaly']])}")
        print(f"Highest Risk Score: {df['Risk_Score'].max():.2f}/10")
        print(f"Days with Risk > 5: {len(df[df['Risk_Score'] > 5])}")
#_______________________________________________________________________________________________________________________________________________________________________________________
# step 6

  #  Detect anomalies in financial data using Isolation Forest algorithm.
def anomaly_ML(df):
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(contamination=0.05)
    df['ML_Anomaly'] = model.fit_predict(df[['Close', 'Volume']]) == -1
    anomaly_number = df['ML_Anomaly'].sum()
    print('IsolationForest Anomaly predicted:',anomaly_number)




