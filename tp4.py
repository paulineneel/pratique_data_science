# TP 4 - Régression financière

!pip install ta

import ta
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from google.colab import drive

drive.mount('/content/drive')

# Charger les fichiers CSV
filepaths = glob.glob('/content/drive/MyDrive/Companies_historical_data/*.csv')
liste_dataframes = []

for filepath in filepaths:
    df = pd.read_csv(filepath)[['Date', 'Close']]
    df['Date'] = pd.to_datetime(df['Date'])
    company_name = os.path.basename(filepath).replace('.csv', '').split('_')[0]
    df['Company'] = company_name
    liste_dataframes.append(df)

base_donnees = pd.concat(liste_dataframes, ignore_index=True)
returns = base_donnees.pivot(index='Date', columns='Company', values='Close')
returns = returns.sort_index()
returns.dropna(inplace=True)

scaler = StandardScaler()
scaled_returns = scaler.fit_transform(returns)
scaled_returns_df = pd.DataFrame(scaled_returns, index=returns.index, columns=returns.columns)
split_index = int(len(scaled_returns_df) * 0.8)
X_train_df = scaled_returns_df.iloc[:split_index]
X_test_df = scaled_returns_df.iloc[split_index:]

def create_target_features(df, n_days=30):
    x, y = [], []
    for i in range(n_days, df.shape[0]):
        x.append(df[i - n_days:i, 0])
        y.append(df[i, 0])
    return np.array(x), np.array(y)

# Choix d'une entreprise cible (exemple : Apple)
target_asset = 'Apple'
data = returns[[target_asset]].dropna().values
n_days = 30
X, y = create_target_features(data, n_days=n_days)

print("Shape X:", X.shape)
print("Shape y:", y.shape)

def prepare_dataset_for_regression(close_series, n_days=30, split_ratio=0.8):
    returns = close_series.pct_change(fill_method=None).dropna().values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns)
    X, y = create_target_features(scaled_returns, n_days=n_days)
    split_index = int(len(X) * split_ratio)
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:], scaler

close_series = returns['Apple']
X_train, X_test, y_train, y_test, scaler = prepare_dataset_for_regression(close_series, n_days=30)

def train_xgboost(X_train, y_train):
    model = XGBRegressor(objective='reg:squarederror')
    params = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1]
    }
    grid = GridSearchCV(model, params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_score_, grid.best_params_

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor()
    params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    grid = GridSearchCV(model, params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_score_, grid.best_params_

def train_knn(X_train, y_train):
    model = KNeighborsRegressor()
    params = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
    grid = GridSearchCV(model, params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_score_, grid.best_params_

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    params = {
        'fit_intercept': [True, False],
        'positive': [False, True]
    }
    grid = GridSearchCV(model, params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_score_, grid.best_params_

def evaluate_model(model, X_test, y_test, scaler=None):
    y_pred = model.predict(X_test)
    if scaler:
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("MSE:", mse)
    print("RMSE:", rmse)
    return mse, rmse, y_test, y_pred

def run_all_models(X_train, y_train, X_test, y_test, scaler=None):
    model_functions = {
        "XGBoost": train_xgboost,
        "Random Forest": train_random_forest,
        "KNN": train_knn,
        "Linear Regression": train_linear_regression
    }
    results = {}
    for name, train_func in model_functions.items():
        print(f"\nModel: {name}")
        model, score, params = train_func(X_train, y_train)
        mse, rmse, y_test_inv, y_pred_inv = evaluate_model(model, X_test, y_test, scaler)
        results[name] = {
            "model": model, "mse": mse, "rmse": rmse,
            "params": params, "y_test": y_test_inv, "y_pred": y_pred_inv
        }
    best_model_name = min(results, key=lambda k: results[k]['rmse'])
    best = results[best_model_name]
    print(f"\nBest model: {best_model_name}\nRMSE: {best['rmse']:.4f}\nMSE: {best['mse']:.4f}")
    return best_model_name, best['model'], best

best_model_name, best_model, best_result = run_all_models(X_train, y_train, X_test, y_test, scaler)

def plot_predictions_vs_real(real_series, y_train_len, y_pred, label='Prédictions', n_days=30):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(real_series)), real_series, color='red', label='Valeurs réelles')
    ax.plot(range(y_train_len + n_days, y_train_len + n_days + len(y_pred)), y_pred,
            color='blue', label=label)
    ax.set_title('Comparaison : Prédictions vs Réel')
    ax.set_xlabel('Temps')
    ax.set_ylabel('Prix ou rendement')
    ax.legend()
    plt.tight_layout()
    plt.show()

real_returns = returns['Apple'].pct_change().dropna().values[n_days:]
plot_predictions_vs_real(real_series=real_returns, y_train_len=len(y_train),
                         y_pred=best_result['y_pred'], label=f'Prédictions {best_model_name}', n_days=30)
