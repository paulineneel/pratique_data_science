# TP 5 - Deep Learning pour la prévision financière

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# MLP

def build_mlp_model(input_shape, hidden_dims=[64, 32], dropout_rate=0.2, activation='relu', optimizer='adam', learning_rate=0.001):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_shape,)))
    for dim in hidden_dims:
        model.add(tf.keras.layers.Dense(dim, activation=activation))
        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1))
    opt = tf.keras.optimizers.get(optimizer)
    opt.learning_rate = learning_rate
    model.compile(optimizer=opt, loss='mean_squared_error')
    return model

# RNN

def build_rnn_model(input_shape, hidden_dims=[50], dropout_rate=0.2, activation='tanh', optimizer='adam', learning_rate=0.001):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    for i, dim in enumerate(hidden_dims):
        return_sequences = i < len(hidden_dims) - 1
        model.add(tf.keras.layers.SimpleRNN(dim, activation=activation, return_sequences=return_sequences))
        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1))
    opt = tf.keras.optimizers.get(optimizer)
    opt.learning_rate = learning_rate
    model.compile(optimizer=opt, loss='mean_squared_error')
    return model

# LSTM

def build_lstm_model(input_shape, hidden_dims=[50], dropout_rate=0.2, activation='tanh', optimizer='adam', learning_rate=0.001):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    for i, dim in enumerate(hidden_dims):
        return_sequences = i < len(hidden_dims) - 1
        model.add(tf.keras.layers.LSTM(dim, activation=activation, return_sequences=return_sequences))
        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1))
    opt = tf.keras.optimizers.get(optimizer)
    opt.learning_rate = learning_rate
    model.compile(optimizer=opt, loss='mean_squared_error')
    return model

# Entraînement

def train_model(model_type, X_train, y_train, input_shape, hidden_dims=[64], dropout_rate=0.2, activation='relu', optimizer='adam', learning_rate=0.001, epochs=50, batch_size=32):
    if model_type == "MLP":
        model = build_mlp_model(input_shape, hidden_dims, dropout_rate, activation, optimizer, learning_rate)
    elif model_type == "RNN":
        model = build_rnn_model(input_shape, hidden_dims, dropout_rate, activation, optimizer, learning_rate)
    elif model_type == "LSTM":
        model = build_lstm_model(input_shape, hidden_dims, dropout_rate, activation, optimizer, learning_rate)
    else:
        raise ValueError("Invalid model type")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

# Prédiction + évaluation

def predict(model, X_test, y_test, scaler):
    y_pred = model.predict(X_test)
    y_pred_inverse = scaler.inverse_transform(y_pred)
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    rmse = np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_inverse[:10], label='Réelles')
    plt.plot(y_pred_inverse[:10], label='Prédictions')
    plt.legend()
    plt.title("Comparaison des 10 premières prédictions")
    plt.show()
    return mae, rmse, y_test_inverse, y_pred_inverse

# Comparaison des architectures

def compare_models(X_train, y_train, X_test, y_test, X_train_rnn, X_test_rnn, scaler):
    results = {}

    for model_type, archs, input_shape in [
        ("MLP", [[64], [128], [64, 32]], X_train.shape[1]),
        ("RNN", [[50], [100], [50, 30]], (X_train_rnn.shape[1], 1)),
        ("LSTM", [[50], [100], [50, 30]], (X_train_rnn.shape[1], 1))
    ]:
        for arch in archs:
            print(f"Testing {model_type} with architecture: {arch}")
            model = train_model(model_type, X_train_rnn if model_type != "MLP" else X_train, y_train, input_shape, hidden_dims=arch, epochs=50, batch_size=32)
            mae, rmse, _, _ = predict(model, X_test_rnn if model_type != "MLP" else X_test, y_test, scaler)
            results[f"{model_type} - {arch}"] = {"MAE": mae, "RMSE": rmse}
    return results

# Exemple d'appel
# Assurez-vous que X_train, y_train, X_test, y_test, scaler, X_train_rnn, X_test_rnn sont définis
# results = compare_models(X_train, y_train, X_test, y_test, X_train_rnn, X_test_rnn, scaler)
# pd.DataFrame(results).T.sort_values(by='RMSE')
