#!/usr/bin/env python3
""" forecast """

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

preprocess_data = __import__('preprocess_data').preprocess_data


def forecast():
    """ forecast """
    # Load the dataset
    data = preprocess_data('./data/btc_full_dataset.csv')

    # Select features for prediction
    features_v1 = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']
    features_v2 = ['Close']
    features_v3 = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']

    # drop first column
    data_v1 = np.delete(data, 0, axis=1)  # drop first column (timestamp)

    # delete last 1000 rows
    data_v1 = data_v1[:-1000]

    data_v2 = data_v1[:, 3]  # Close column
    data_v2 = data_v2[:-1000]

    data_v3 = data_v1

    # Normalize the data
    scaler = MinMaxScaler()
    # data_scaled = scaler.fit_transform(data_v1)
    # data_scaled = scaler.fit_transform(data_v2.reshape(-1, 1))
    data_scaled = scaler.fit_transform(data_v3)

    # Create sequences for training
    sequence_length = 24  # 24 hours
    sequences = []
    target = []
    for i in range(len(data_scaled) - sequence_length):
        sequences.append(data_scaled[i:i+sequence_length])
        target.append(data_scaled[i+sequence_length])

    sequences = np.array(sequences)
    target = np.array(target)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(sequences, target, test_size=0.2, random_state=42)

    # ====================================================================================================
    # V1
    # ====================================================================================================
    # model = Sequential()
    # model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, len(features_v1))))
    # model.add(LSTM(units=50))
    # model.add(Dense(len(features_v1)))  # Output layer has the same number of features

    # ====================================================================================================
    # V2
    # ====================================================================================================
    # model = Sequential()
    # model.add(LSTM(units=200, return_sequences=True, input_shape=(sequence_length, len(features_v2))))
    # model.add(LSTM(units=100, return_sequences=True))
    # model.add(LSTM(units=50)) 
    # model.add(Dense(len(features_v2)))  # Output layer has the same number of features

    # ====================================================================================================
    # V3
    # ====================================================================================================
    model = Sequential()
    model.add(LSTM(units=200, return_sequences=True, input_shape=(sequence_length, len(features_v3))))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(LSTM(units=75, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(len(features_v3)))  # Output layer has the same number of features


    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss')
        ])

    # Save the model
    model.save('forecast_btc.h5')
    print('Model saved')

    # # Make predictions
    # predictions = model.predict(X_test)

    # # Inverse transform the predictions and actual values
    # predictions_actual = scaler.inverse_transform(predictions)
    # y_test_actual = scaler.inverse_transform(y_test)

    # # Plot the predicted vs. actual values
    # plt.figure(figsize=(12, 6))
    # plt.plot(y_test_actual[:, 3], label='Actual')
    # plt.plot(predictions_actual[:, 3], label='Predicted')
    # plt.xlabel('Time')
    # plt.ylabel('BTC Close Price')
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    forecast()
