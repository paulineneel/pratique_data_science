
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

"""### Création des modèles"""

def build_mlp_model(input_shape, hidden_dims=[64, 32], dropout_rate=0.2, activation='relu', optimizer='adam', learning_rate=0.001):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_shape,)))

    for dim in hidden_dims:
        model.add(tf.keras.layers.Dense(dim, activation=activation))
        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(1))  # Output layer
    opt = tf.keras.optimizers.get(optimizer)
    opt.learning_rate = learning_rate

    model.compile(optimizer=opt, loss='mean_squared_error')
    return model

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

# Pour RNN ou LSTM, reshape obligatoire
X_train_rnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_rnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

"""### Entraînement des modèles"""

def train_model(
    model_type,
    X_train,
    y_train,
    input_shape,
    hidden_dims=[64],
    dropout_rate=0.2,
    activation='relu',
    optimizer='adam',
    learning_rate=0.001,
    epochs=50,
    batch_size=32
):
    # Sélection du bon modèle
    if model_type == "MLP":
        model = build_mlp_model(
            input_shape=input_shape,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            optimizer=optimizer,
            learning_rate=learning_rate
        )

    elif model_type == "RNN":
        model = build_rnn_model(
            input_shape=input_shape,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            optimizer=optimizer,
            learning_rate=learning_rate
        )

    elif model_type == "LSTM":
        model = build_lstm_model(
            input_shape=input_shape,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            optimizer=optimizer,
            learning_rate=learning_rate
        )
    else:
        raise ValueError("model_type must be one of: 'MLP', 'RNN', 'LSTM'")

    # Entraînement
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    return model

"""### Prédiction (Test du modèle)"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def predict(model, X_test, y_test, scaler, model_type):

    # Faire des prédictions
    y_pred = model.predict(X_test)

    # Inverser la standardisation
    y_pred_inverse = scaler.inverse_transform(y_pred)
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calcul des erreurs
    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    rmse = np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Affichage des 10 premières valeurs prédites vs réelles
    print("\nComparaison des 10 premières valeurs prédites vs réelles:")
    print("Prédites (y_pred): ", y_pred_inverse[:10].flatten())
    print("Réelles (y_test): ", y_test_inverse[:10].flatten())

    # Affichage graphique des résultats
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_inverse[:10], label='Valeurs réelles')
    plt.plot(y_pred_inverse[:10], label='Prédictions')
    plt.legend()
    plt.title("Comparaison des 10 premières prédictions vs réelles")
    plt.show()

    return y_pred_inverse, y_test_inverse

# Fonction pour entraîner les modèles
def train_model(
    model_type,
    X_train,
    y_train,
    input_shape,
    hidden_dims=[64],
    dropout_rate=0.2,
    activation='relu',
    optimizer='adam',
    learning_rate=0.001,
    epochs=50,
    batch_size=32
):
    # Sélection du bon modèle
    if model_type == "MLP":
        model = build_mlp_model(
            input_shape=input_shape,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            optimizer=optimizer,
            learning_rate=learning_rate
        )

    elif model_type == "RNN":
        model = build_rnn_model(
            input_shape=input_shape,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            optimizer=optimizer,
            learning_rate=learning_rate
        )

    elif model_type == "LSTM":
        model = build_lstm_model(
            input_shape=input_shape,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            activation=activation,
            optimizer=optimizer,
            learning_rate=learning_rate
        )
    else:
        raise ValueError("model_type must be one of: 'MLP', 'RNN', 'LSTM'")

    # Entraînement
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    return model

# Fonction pour effectuer les prédictions et afficher les résultats
def predict(model, X_test, y_test, scaler, model_type):
    y_pred = model.predict(X_test)

    # Inverser la standardisation
    y_pred_inverse = scaler.inverse_transform(y_pred)
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calcul des erreurs
    mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
    rmse = np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Affichage des 10 premières valeurs prédites vs réelles
    print("\nComparaison des 10 premières valeurs prédites vs réelles:")
    print("Prédites (y_pred): ", y_pred_inverse[:10].flatten())
    print("Réelles (y_test): ", y_test_inverse[:10].flatten())

    # Affichage graphique des résultats
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_inverse[:10], label='Valeurs réelles')
    plt.plot(y_pred_inverse[:10], label='Prédictions')
    plt.legend()
    plt.title("Comparaison des 10 premières prédictions vs réelles")
    plt.show()

    return y_pred_inverse, y_test_inverse

# Comparaison des différentes architectures
mlp_results = {}
for hidden_dims in [[64], [128], [64, 32]]:
    print(f"Testing MLP with architecture: {hidden_dims}")
    model = train_model("MLP", X_train, y_train, input_shape=30, hidden_dims=hidden_dims, epochs=50, batch_size=32)
    y_pred, y_test_inverse = predict(model, X_test, y_test, scaler, model_type="MLP")
    mae = mean_absolute_error(y_test_inverse, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_inverse, y_pred))
    mlp_results[str(hidden_dims)] = {'MAE': mae, 'RMSE': rmse}

rnn_results = {}
for hidden_dims in [[50], [100], [50, 30]]:
    print(f"Testing RNN with architecture: {hidden_dims}")
    model = train_model("RNN", X_train_rnn, y_train, input_shape=(30, 1), hidden_dims=hidden_dims, epochs=50, batch_size=32)
    y_pred, y_test_inverse = predict(model, X_test_rnn, y_test, scaler, model_type="RNN")
    mae = mean_absolute_error(y_test_inverse, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_inverse, y_pred))
    rnn_results[str(hidden_dims)] = {'MAE': mae, 'RMSE': rmse}

lstm_results = {}
for hidden_dims in [[50], [100], [50, 30]]:
    print(f"Testing LSTM with architecture: {hidden_dims}")
    model = train_model("LSTM", X_train_rnn, y_train, input_shape=(30, 1), hidden_dims=hidden_dims, epochs=50, batch_size=32)
    y_pred, y_test_inverse = predict(model, X_test_rnn, y_test, scaler, model_type="LSTM")
    mae = mean_absolute_error(y_test_inverse, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_inverse, y_pred))
    lstm_results[str(hidden_dims)] = {'MAE': mae, 'RMSE': rmse}

# Sélection du meilleur modèle MLP
best_mlp_architecture = min(mlp_results, key=lambda k: mlp_results[k]['RMSE'])
best_mlp_model = train_model("MLP", X_train, y_train, input_shape=30, hidden_dims=eval(best_mlp_architecture), epochs=50, batch_size=32)

# Sélection du meilleur modèle RNN
best_rnn_architecture = min(rnn_results, key=lambda k: rnn_results[k]['RMSE'])
best_rnn_model = train_model("RNN", X_train_rnn, y_train, input_shape=(30, 1), hidden_dims=eval(best_rnn_architecture), epochs=50, batch_size=32)

# Sélection du meilleur modèle LSTM
best_lstm_architecture = min(lstm_results, key=lambda k: lstm_results[k]['RMSE'])
best_lstm_model = train_model("LSTM", X_train_rnn, y_train, input_shape=(30, 1), hidden_dims=eval(best_lstm_architecture), epochs=50, batch_size=32)

# Prédictions sur les meilleurs modèles
best_mlp_pred, best_mlp_test_inverse = predict(best_mlp_model, X_test, y_test, scaler, model_type="MLP")
best_rnn_pred, best_rnn_test_inverse = predict(best_rnn_model, X_test_rnn, y_test, scaler, model_type="RNN")
best_lstm_pred, best_lstm_test_inverse = predict(best_lstm_model, X_test_rnn, y_test, scaler, model_type="LSTM")

# Comparaison graphique : MLP
plt.figure(figsize=(10, 6))
plt.plot(best_mlp_test_inverse, label='Réelles')
plt.plot(best_mlp_pred, label='Prédictions')
plt.legend()
plt.title("Comparaison MLP : Valeurs réelles vs prédites")
plt.show()

# Comparaison graphique : RNN
plt.figure(figsize=(10, 6))
plt.plot(best_rnn_test_inverse, label='Réelles')
plt.plot(best_rnn_pred, label='Prédictions')
plt.legend()
plt.title("Comparaison RNN : Valeurs réelles vs prédites")
plt.show()

# Comparaison graphique : LSTM
plt.figure(figsize=(10, 6))
plt.plot(best_lstm_test_inverse, label='Réelles')
plt.plot(best_lstm_pred, label='Prédictions')
plt.legend()
plt.title("Comparaison LSTM : Valeurs réelles vs prédites")
plt.show()

# Tableau des résultats
results = {
    "Modèle": ["MLP", "RNN", "LSTM"],
    "MAE": [mlp_results[best_mlp_architecture]['MAE'], rnn_results[best_rnn_architecture]['MAE'], lstm_results[best_lstm_architecture]['MAE']],
    "RMSE": [mlp_results[best_mlp_architecture]['RMSE'], rnn_results[best_rnn_architecture]['RMSE'], lstm_results[best_lstm_architecture]['RMSE']]
}

results_df = pd.DataFrame(results)
print(results_df)
