#!/usr/bin/env python3


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

prep_data = __import__('preprocess_data').preprocess_data


# Load the dataset
data = prep_data('./data/btc_full_dataset.csv')

# load the model
model = tf.keras.models.load_model('./forecast_btc.h5')

# Select features for prediction
features = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']

# delete last 1000 rows
data = data[-8000:]

# drop first column
timestamp = data[:, 0]
data = np.delete(data, 0, axis=1)  # drop first column (timestamp)

# timestamp to datetime
timestamp = pd.to_datetime(timestamp, unit='s')

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences for training
sequence_length = 24  # 24 hours
sequences = []
target = []
for i in range(len(data_scaled) - sequence_length):
    sequences.append(data_scaled[i:i+sequence_length])
    target.append(data_scaled[i+sequence_length])

sequences = np.array(sequences)
target = np.array(target)

# Make predictions
predictions = model.predict(sequences)
# Inverse transform the predictions and actual values
predictions = scaler.inverse_transform(predictions)
target = scaler.inverse_transform(target)

good_trends = {}
for i in range(1, len(predictions)):
    target_trend = target[i][3] - target[i-1][3]
    prediction_trend = predictions[i][3] - predictions[i-1][3]
    if np.sign(target_trend) != np.sign(prediction_trend):
        good_trends[i] = {
            'success': 0,
            'diff': target_trend - prediction_trend,
        }
    else:
        good_trends[i] = {
            'success': 1,
            'diff': target_trend - prediction_trend,
        }

print(f'Accuracy over trend: {sum([good_trends[i]["success"] for i in good_trends]) / len(good_trends)}')
print(f'Average difference: {sum([good_trends[i]["diff"] for i in good_trends]) / len(good_trends)}')

# save csv predictions and actual values
pd.DataFrame(predictions).to_csv('predictions.csv', index=False)
pd.DataFrame(target).to_csv('y_test_actual.csv', index=False)

# Plot the predicted vs. actual values
plt.figure(figsize=(12, 6))
plt.plot(target[:, 1], label='target')
plt.plot(predictions[:, 1], label='predictions')
# add timestamp
plt.xticks(np.arange(0, len(timestamp), 10), timestamp[::10], rotation=45)
plt.xlabel('Time')
plt.ylabel('BTC Close Price')
plt.legend()
plt.show()