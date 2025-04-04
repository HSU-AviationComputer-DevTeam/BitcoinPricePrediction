def prepare_data(btc_file='data/btc_ohlcv.csv', fg_file='data/fear_greed_index.csv'):
    """
    Data loading and preprocessing function
    """
    # BTC data loading
    btc_data = pd.read_csv(btc_file)
    btc_data['Date'] = pd.to_datetime(btc_data['Date'])
    btc_data.set_index('Date', inplace=True)

    # Extract closing price data and scaling
    close_prices = btc_data['Close'].values.reshape(-1, 1)

    # MinMax scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)

    # Create sequence data (e.g. 30-day sequence)
    seq_length = 30
    X, y = [], []
    for i in range(len(scaled_prices) - seq_length):
        X.append(scaled_prices[i:i+seq_length])
        y.append(scaled_prices[i+seq_length])

    X = np.array(X)
    y = np.array(y)

    # Data split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return {
        'X_test_seq': X_test,
        'y_test_seq': y_test,
        'scaler_y': scaler,
        'btc_data': btc_data,
        'test_dates': btc_data.index[len(btc_data) - len(y_test):]
    }

def load_and_predict(model_path, data_dict):
    """
    Model loading and prediction function
    """
    # Load model
    model = load_model(model_path)

    # Extract data
    X_test_seq = data_dict['X_test_seq']
    y_test_seq = data_dict['y_test_seq']
    scaler_y = data_dict['scaler_y']
    test_dates = data_dict['test_dates']

    # Prediction
    y_pred = model.predict(X_test_seq)

    # Inverse transform scaling
    y_test_orig = scaler_y.inverse_transform(y_test_seq)
    y_pred_orig = scaler_y.inverse_transform(y_pred)

    return y_test_orig, y_pred_orig, test_dates

def visualize_prediction_results(y_test_orig, y_pred_orig, date_index=None):
    """
    Function to visualize prediction results in various ways
    """
    # Data preparation
    y_test_flat = y_test_orig.flatten()
    y_pred_flat = y_pred_orig.flatten()

    # Calculate residuals
    residuals = y_test_flat - y_pred_flat

    # If no date index is provided, use data point index
    if date_index is None:
        date_index = pd.RangeIndex(start=0, stop=len(y_test_flat), step=1)

    # Create results dataframe
    results_df = pd.DataFrame({
        'Date': date_index,
        'Actual': y_test_flat,
        'Predicted': y_pred_flat,
        'Residual': residuals,
        'Abs_Residual': np.abs(residuals),
        'Squared_Residual': residuals**2,
        'Percent_Error': np.abs(residuals / y_test_flat) * 100
    })

    # 1. Time series plot: Actual vs Predicted Price
    plt.figure(figsize=(15, 8))
    plt.plot(results_df['Date'], results_df['Actual'], label='Actual Price', color='blue')
    plt.plot(results_df['Date'], results_df['Predicted'], label='Predicted Price', color='red', linestyle='--')
    plt.title('Bitcoin Price: Actual vs Predicted', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('time_series_plot.png', dpi=300)
    plt.show()

    # 2. Scatter Plot: Actual Price vs Predicted Price
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Actual', y='Predicted', data=results_df, alpha=0.6)

    # Add regression line
    min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
    max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Ideal Prediction')

    # Calculate and display R² value
    r2 = r2_score(results_df['Actual'], results_df['Predicted'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        results_df['Actual'], results_df['Predicted']
    )
    plt.annotate(f"R² = {r2:.4f}\ny = {slope:.4f}x + {intercept:.2f}",
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.title('Actual Price vs Predicted Price Scatter Plot', fontsize=16)
    plt.xlabel('Actual Price (USD)', fontsize=12)
    plt.ylabel('Predicted Price (USD)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('scatter_plot.png', dpi=300)
    plt.show()

    # 3. Residual Plot
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(results_df['Date'], results_df['Residual'], color='green')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Residuals (Prediction Errors) Over Time', fontsize=16)
    plt.ylabel('Residual (USD)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 4. Residual Histogram
    plt.subplot(2, 1, 2)
    sns.histplot(results_df['Residual'], kde=True, bins=30)
    plt.title('Residual Distribution', fontsize=16)
    plt.xlabel('Residual (USD)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('residual_analysis.png', dpi=300)
    plt.show()

    # 5. Autocorrelation Function (ACF) Plot of Residuals
    plt.figure(figsize=(12, 6))
    plot_acf(results_df['Residual'], lags=40, alpha=0.05)
    plt.title('Autocorrelation Function (ACF) of Residuals', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('residual_acf.png', dpi=300)
    plt.show()

    return results_df

def analyze_prediction_errors(results_df, window_size=7):
    """
    Function to analyze prediction errors and find patterns
    """
    # Copy results
    df = results_df.copy()

    # Calculate rolling volatility
    df['rolling_volatility'] = df['Actual'].pct_change().rolling(window=window_size).std() * 100

    # Analyze relationship between prediction error and volatility
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(df['Date'], df['rolling_volatility'], color='purple', label='Volatility (7-day)')
    plt.plot(df['Date'], df['Abs_Residual'] / df['Actual'] * 100, color='orange', label='Percent Error', alpha=0.6)
    plt.title('Relationship Between Volatility and Prediction Error', fontsize=16)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Check relationship between volatility and error with scatter plot
    plt.subplot(2, 1, 2)
    sns.scatterplot(x='rolling_volatility', y='Percent_Error', data=df, alpha=0.6)

    # Add regression line
    sns.regplot(x='rolling_volatility', y='Percent_Error', data=df, scatter=False, color='red')

    # Calculate and display correlation coefficient
    corr = df['rolling_volatility'].corr(df['Percent_Error'])
    plt.annotate(f"Correlation: {corr:.4f}",
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.title('Volatility vs Percent Error Scatter Plot', fontsize=16)
    plt.xlabel('7-day Rolling Volatility (%)', fontsize=12)
    plt.ylabel('Percent Error (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('volatility_vs_error.png', dpi=300)
    plt.show()

    return df

def interpret_model_performance(y_test_orig, y_pred_orig, results_df=None):
    """
    Function to comprehensively interpret model performance metrics
    """
    # Flatten data
    y_test_flat = y_test_orig.flatten()
    y_pred_flat = y_pred_orig.flatten()

    # Calculate basic evaluation metrics
    mse = mean_squared_error(y_test_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    mape = np.mean(np.abs((y_test_flat - y_pred_flat) / y_test_flat)) * 100
    r2 = r2_score(y_test_flat, y_pred_flat)

    # Direction accuracy
    actual_diff = np.diff(y_test_flat)
    pred_diff = np.diff(y_pred_flat)
    direction_match = (actual_diff > 0) == (pred_diff > 0)
    direction_accuracy = np.mean(direction_match) * 100

    # Additional evaluation metrics
    # 1. Theil's U statistic
    # Create simple prediction model (predicting previous value)
    naive_pred = np.roll(y_test_flat, 1)
    naive_pred[0] = naive_pred[1]  # Handle first value

    # Compare MSE of proposed model and simple model
    model_mse = mean_squared_error(y_test_flat, y_pred_flat)
    naive_mse = mean_squared_error(y_test_flat, naive_pred)
    theil_u = np.sqrt(model_mse) / np.sqrt(naive_mse)

    # 2. Positive/Negative direction accuracy
    up_accuracy = np.mean(direction_match[actual_diff > 0]) * 100 if np.any(actual_diff > 0) else 0
    down_accuracy = np.mean(direction_match[actual_diff < 0]) * 100 if np.any(actual_diff < 0) else 0

    # Interpret and output results
    print("\n===== Model Performance Comprehensive Evaluation =====")
    print(f"1. Basic Prediction Accuracy Metrics:")
    print(f"   - RMSE: ${rmse:.2f}")
    print(f"   - MAE: ${mae:.2f}")
    print(f"   - MAPE: {mape:.2f}%")
    print(f"   - R² (Coefficient of Determination): {r2:.4f}")

    print(f"\n2. Directional Prediction Metrics:")
    print(f"   - Overall Direction Accuracy: {direction_accuracy:.2f}%")
    print(f"   - Upward Direction Accuracy: {up_accuracy:.2f}%")
    print(f"   - Downward Direction Accuracy: {down_accuracy:.2f}%")

    # Interpret Theil's U
    print(f"\n3. Theil's U Statistic: {theil_u:.4f}")
    if theil_u < 0.5:
        print("   - Excellent: More than 50% improvement over simple prediction")
    elif theil_u < 0.8:
        print("   - Good: 20-50% improvement over simple prediction")
    elif theil_u < 1:
        print("   - Satisfactory: Better than simple prediction")
    else:
        print("   - Needs Improvement: Performance lower than simple prediction")

    # Performance interpretation and scoring
    avg_score = 0

    # RMSE score
    avg_price = np.mean(y_test_flat)
    rmse_percent = (rmse / avg_price) * 100
    rmse_score = max(0, min(5, 5 * (1 - rmse_percent/20)))  # 0-5 points

    # MAPE score
    mape_score = max(0, min(5, 5 * (1 - mape/30)))  # 0-5 points

    # Direction accuracy score
    dir_score = max(0, min(5, 5 * (direction_accuracy - 50) / 25))  # 0-5 points

    # R² score
    r2_score_val = max(0, min(5, 5 * r2))  # 0-5 points

    # Theil's U score
    theil_score = max(0, min(5, 5 * (1 - theil_u)))  # 0-5 points

    # Calculate average score
    avg_score = (rmse_score + mape_score + dir_score + r2_score_val + theil_score) / 5

    print("\n===== Overall Evaluation =====")
    if avg_score > 4:
        print("Model Performance: Excellent")
        print("→ Can be directly used in trading strategies")
    elif avg_score > 3:
        print("Model Performance: Good")
        print("→ Can be used as a supplementary indicator")
    elif avg_score > 2:
        print("Model Performance: Satisfactory")
        print("→ Can be referenced for long-term trend analysis")
    else:
        print("Model Performance: Needs Improvement")
        print("→ Model structure and input data need to be reconsidered")

    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'direction_accuracy': direction_accuracy,
        'up_accuracy': up_accuracy,
        'down_accuracy': down_accuracy,
        'theil_u': theil_u,
        'avg_score': avg_score
    }

def evaluate_bitcoin_prediction_model(y_test_orig, y_pred_orig, test_dates):
   """
   Integrated function for evaluating Bitcoin price prediction model
   """
   print("===== Bitcoin Price Prediction Model Evaluation =====")

   # 1. Visualize prediction results
   results_df = visualize_prediction_results(y_test_orig, y_pred_orig, test_dates)

   # 2. Analyze prediction errors
   analyze_prediction_errors(results_df)

   # 3. Interpret model performance
   performance_metrics = interpret_model_performance(y_test_orig, y_pred_orig)

   # Return final results
   return {
       'results_df': results_df,
       'performance_metrics': performance_metrics
   }

# Main execution code
def main():
   # Model path
   model_path = 'optimal_bitcoin_model.h5'

   # Prepare data
   data_dict = prepare_data()

   # Load model and predict
   y_test_orig, y_pred_orig, test_dates = load_and_predict(model_path, data_dict)

   # Evaluate model
   evaluation_summary = evaluate_bitcoin_prediction_model(y_test_orig, y_pred_orig, test_dates)

if __name__ == "__main__":
   main()