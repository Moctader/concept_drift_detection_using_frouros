# Import necessary libraries
import ffn
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

# Fetch stock data from Yahoo Finance
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')

# Remove rows with NaN or infinite values
data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Adj Close'])

# Preprocess data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Adj Close'].values.reshape(-1, 1))

# Create dataset function for LSTM
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# Define time step for LSTM model
time_step = 60
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

early_stop = EarlyStopping(monitor='val_loss', patience=10)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, callbacks=[early_stop], verbose=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions to get actual prices
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Combine train and test predictions
predicted_prices = np.concatenate((train_predict, test_predict), axis=0)
predicted_prices_df = pd.DataFrame(predicted_prices, index=data.index[time_step + 1:], columns=['Predicted Price'])

# Calculate rolling performance metrics
window = 252  # 1 year of trading days
predicted_prices_df['Returns'] = predicted_prices_df['Predicted Price'].pct_change()
predicted_prices_df['Sharpe'] = predicted_prices_df['Returns'].rolling(window).apply(lambda x: x.mean() / x.std() * np.sqrt(252), raw=False)
predicted_prices_df['Sortino'] = predicted_prices_df['Returns'].rolling(window).apply(lambda x: x.mean() / x[x < 0].std() * np.sqrt(252), raw=False)
predicted_prices_df['CAGR'] = predicted_prices_df['Predicted Price'].rolling(window).apply(lambda x: (x.iloc[-1] / x.iloc[0]) ** (252 / len(x)) - 1, raw=False)
predicted_prices_df['Total Return'] = predicted_prices_df['Predicted Price'].rolling(window).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1, raw=False)
predicted_prices_df['Max Drawdown'] = predicted_prices_df['Predicted Price'].rolling(window).apply(lambda x: (x / x.cummax() - 1).min(), raw=False)
predicted_prices_df['YTD'] = predicted_prices_df['Predicted Price'].pct_change().cumsum()

# Generate trading signals based on rolling performance metrics
def generate_signals(df):
    signals = np.zeros(len(df))  # Default signal to 0
    
    # Generate buy signals
    buy_condition = (df['Sharpe'] > 1.0) & (df['Sortino'] > 2.0) & (df['CAGR'] > 0.0) & (df['YTD'] > 0.0)
    signals = np.where(buy_condition, 1, signals)
    
    # Generate sell signals
    sell_condition = (df['Total Return'] < 0) | (df['Max Drawdown'] > 0.2)
    signals = np.where(sell_condition, -1, signals)
    
    return signals

# Create signals based on the metrics
predicted_prices_df['Signal'] = generate_signals(predicted_prices_df)

# Simulate trading strategy
initial_investment = 10000  
cash = initial_investment
stock = 0
portfolio_value = []

for i in range(len(predicted_prices_df)):
    signal = predicted_prices_df['Signal'].iloc[i]
    price = predicted_prices_df['Predicted Price'].iloc[i]
    
    if signal == 1 and cash > 0:  # Buy signal
        stock = cash / price
        cash = 0
    elif signal == -1 and stock > 0:  # Sell signal
        cash = stock * price
        stock = 0
    
    portfolio_value.append(cash + stock * price)

predicted_prices_df['Portfolio Value'] = portfolio_value

# Plot stock prices with trading signals and Max Drawdown
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot actual and predicted stock prices on the primary y-axis
ax1.plot(data['Adj Close'], label='Actual Prices', color='blue')
ax1.plot(predicted_prices_df.index, predicted_prices_df['Predicted Price'], label='Predicted Prices', color='orange')

# Highlight buy/sell signals
buy_signals = predicted_prices_df[predicted_prices_df['Signal'] == 1]
sell_signals = predicted_prices_df[predicted_prices_df['Signal'] == -1]
ax1.scatter(buy_signals.index, buy_signals['Predicted Price'], marker='^', color='green', label='Buy Signal', s=100)
ax1.scatter(sell_signals.index, sell_signals['Predicted Price'], marker='v', color='red', label='Sell Signal', s=100)

# Secondary y-axis for Max Drawdown
ax2 = ax1.twinx()
ax2.plot(predicted_prices_df.index, predicted_prices_df['Max Drawdown'], label='Max Drawdown', color='purple', linestyle='--')
ax2.set_ylabel('Max Drawdown', color='purple')
ax2.tick_params(axis='y', labelcolor='purple')

# Set title and labels
ax1.set_title('AAPL Stock Price with Trading Signals and Max Drawdown')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Add legends for both y-axes
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
fig.tight_layout()

# Show the plot
plt.show()

# Plot the portfolio value
plt.figure(figsize=(14, 7))
plt.plot(predicted_prices_df.index, predicted_prices_df['Portfolio Value'], label='Portfolio Value', color='purple')
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()

############################################################################################################

# Calculate statistics for model predictions
predicted_higher = test_predict > y_test_actual

# 1. Count how many of the predictions are higher than the current value
higher_predictions_count = np.sum(predicted_higher)

# 2. Calculate percentage growth for predictions that are higher than the actual values
percentage_growth = np.where(predicted_higher, (test_predict - y_test_actual) / y_test_actual * 100, 0)

# 3. Calculate average percentage growth of those that are higher
average_growth = np.mean(percentage_growth[predicted_higher])

# Print results
print(f"Number of predictions higher than the current value: {higher_predictions_count}")
print(f"Average percentage growth for higher predictions: {average_growth:.2f}%")

############################################################################################################

# Identify sell and buyback opportunities based on future price predictions
future_predictions = test_predict[1:]  # Future predictions (next time step)
current_predictions = test_predict[:-1]  # Current predictions (current time step)

# Identify sell opportunities (immediate future is lower)
sell_opportunities = future_predictions < current_predictions
buyback_opportunities = future_predictions > current_predictions

# Calculate potential gains from selling now and buying back later
potential_gains = future_predictions[sell_opportunities] - current_predictions[sell_opportunities]

# Print opportunities
print(f"Sell opportunities found: {np.sum(sell_opportunities)}")
print(f"Average potential gain from sell opportunities: {np.mean(potential_gains):.2f}")
############################################################################################################

# Trading Decision Logic Based on Statistics
def calculate_statistics(prices):
    """
    Calculate statistical averages and standard deviations.
    
    Parameters:
    - prices (pd.Series): Historical stock prices.
    
    Returns:
    - mean_price (float): Mean of the historical prices.
    - std_dev (float): Standard deviation of the historical prices.
    """
    mean_price = prices.mean()
    std_dev = prices.std()
    return mean_price, std_dev

# Historical prices from predicted data
historical_prices = predicted_prices_df['Predicted Price'][-30:]  # Last 30 days as an example

# Calculate statistics
mean_price, std_dev = calculate_statistics(historical_prices)

# Decide shares to trade based on statistical conditions
def decide_shares_to_trade(current_cash, stock_price, current_shares, action='buy', risk_management_pct=0.1, min_capital=1000, mean_price=None, std_dev=None):
    """
    Decide how many shares to buy or sell based on cash, price, and risk management,
    while maintaining a minimum capital level and using statistical averages.
    
    Returns:
    - shares_to_trade (int): Number of shares to buy or sell.
    """
    usable_cash = max(0, current_cash - min_capital)  # Cash after maintaining minimum capital
    
    if action == 'buy':
        # Determine if price is significantly below mean
        if stock_price < (mean_price - std_dev):  
            amount_to_invest = usable_cash * risk_management_pct
            shares_to_trade = int(amount_to_invest // stock_price)
        else:
            print("Current price is not low enough to justify a buy.")
            shares_to_trade = 0
            
    elif action == 'sell':
        # Determine if price is significantly above mean
        if stock_price > (mean_price + std_dev):
            shares_to_trade = int(current_shares * risk_management_pct)
        else:
            print("Current price is not high enough to justify a sell.")
            shares_to_trade = 0
    
    return shares_to_trade

current_cash = cash  
current_price = predicted_prices_df['Predicted Price'].iloc[-1]  
current_shares = stock  # Number of shares held

# Decide to buy shares
shares_to_buy = decide_shares_to_trade(current_cash, current_price, current_shares, action='buy', mean_price=mean_price, std_dev=std_dev, min_capital=1000)
print(f"Shares to buy: {shares_to_buy}")

# Decide to sell shares
shares_to_sell = decide_shares_to_trade(current_cash, current_price, current_shares, action='sell', mean_price=mean_price, std_dev=std_dev, min_capital=1000)
print(f"Shares to sell: {shares_to_sell}")