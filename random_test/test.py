import pandas as pd
import datetime as dt
from datetime import date
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import save_model  # Correct import for saving the model

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Define a function to load the dataset
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data = load_data('AAPL')
df = data
print(df.head())

df = df.drop(['Date', 'Adj Close'], axis=1)
print(df.head())

train = pd.DataFrame(data[0:int(len(data) * 0.70)])
test = pd.DataFrame(data[int(len(data) * 0.70): int(len(data))])

print(train.shape)
print(test.shape)

scaler = MinMaxScaler(feature_range=(0, 1))
train_close = train.iloc[:, 4:5].values
test_close = test.iloc[:, 4:5].values

data_training_array = scaler.fit_transform(train_close)
data_testing_array = scaler.transform(test_close)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i - 100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Generate x_test and y_test
x_test = []
y_test = []

for i in range(100, data_testing_array.shape[0]):
    x_test.append(data_testing_array[i - 100: i])
    y_test.append(data_testing_array[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['MAE'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50)

# Save the model using the native Keras format
save_model(model, 'my_model.keras')

test_close.shape
test_close

past_100_days = pd.DataFrame(train_close[-100:])
test_df = pd.DataFrame(test_close)

# Use pd.concat instead of append
final_df = pd.concat([past_100_days, test_df], ignore_index=True)
input_data = scaler.transform(final_df)

y_pred = model.predict(x_test)

scale_factor = 1 / scaler.scale_
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label="Original Price")
plt.plot(y_pred, 'r', label="Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


print(y_pred[0])
print(y_test[0])