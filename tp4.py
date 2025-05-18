
import ta
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from google.colab import drive
drive.mount('/content/drive')

filepaths = glob.glob('/content/drive/MyDrive/Companies_historical_data/*.csv')

# Liste pour stocker les dataframes des entreprises
liste_dataframes = []

for filepath in filepaths:
    # Lecture des colonnes nécessaires
    df = pd.read_csv(filepath)[['Date', 'Close']]

    # Conversion de la date en format datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Extraction du nom de l'entreprise depuis le nom du fichier
    company_name = os.path.basename(filepath).replace('.csv', '').split('_')[0]

    # Ajout d'une colonne "Company"
    df['Company'] = company_name

    # Ajout du DataFrame à la liste
    liste_dataframes.append(df)

# Concaténation de tous les DataFrames
base_donnees = pd.concat(liste_dataframes, ignore_index=True)

# Pivot pour obtenir une colonne par entreprise avec les prix de clôture
returns = base_donnees.pivot(index='Date', columns='Company', values='Close')

# Tri des dates
returns = returns.sort_index()
returns.dropna(inplace=True)

# Affichage des premières lignes
returns.head()

scaler = StandardScaler()
scaled_returns = scaler.fit_transform(returns)

scaled_returns_df = pd.DataFrame(scaled_returns, index=returns.index, columns=returns.columns)

split_index = int(len(scaled_returns_df) * 0.8)

X_train = scaled_returns_df.iloc[:split_index]
X_test = scaled_returns_df.iloc[split_index:]

def create_target_features(df, n_days=30):
    x = []
    y = []
    for i in range(n_days, df.shape[0]):
        x.append(df[i - n_days:i, 0])  # les 30 derniers jours (features)
        y.append(df[i, 0])             # le jour suivant (label)
    x = np.array(x)
    y = np.array(y)
    return x, y

# On choisit un actif — ici Apple — pour l'exemple
target_asset = 'Apple'

# On récupère sa colonne "Close" et on transforme en tableau NumPy
data = close_prices[[target_asset]].dropna().values  # shape (n_samples, 1)

# Nombre de jours pour les features (fenêtre glissante)
n_days = 30

# Création des datasets
X, y = create_target_features(data, n_days=n_days)

# Affichage très utile pour vérifier
print("Shape des features X :", X.shape)  # Devrait être (nb_samples, 30)
print("Shape des labels y :", y.shape)    # Devrait être (nb_samples,)

def prepare_dataset_for_regression(close_series, n_days=30, split_ratio=0.8):
    # Étape 1: Calcul des rendements
    returns = close_series.pct_change(fill_method=None).dropna().values.reshape(-1, 1)

    # Étape 2: Standardisation
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns)

    # Étape 3: Création des features et labels
    X, y = create_target_features(scaled_returns, n_days=n_days)

    # Étape 4: Séparation train/test
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test, scaler

#Exemple
close_series = close_prices['Apple']
X_train, X_test, y_train, y_test, scaler = prepare_dataset_for_regression(close_series, n_days=30)

"""## **1.2 Algorithmes de régression**

##XG Boost
"""

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

"""##Random Forest"""

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

"""##K-Nearest Neighbors Regressor"""

def train_knn(X_train, y_train):
    model = KNeighborsRegressor()
    params = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # distance de Manhattan (p=1) ou euclidienne (p=2)
    }
    grid = GridSearchCV(model, params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_score_, grid.best_params_

"""##Régression linéaire"""

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    params = {
        'fit_intercept': [True, False],
        'positive': [False, True]
    }
    grid = GridSearchCV(model, params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_score_, grid.best_params_

model, score, params = train_xgboost(X_train, y_train)
print("Best score:", score)
print("Best params:", params)

def evaluate_model(model, X_test, y_test, scaler=None):
    # Prédictions
    y_pred = model.predict(X_test)

    # Si les données étaient standardisées, on les "déscale"
    if scaler:
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    # Calcul des erreurs
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("MSE :", mse)
    print("RMSE :", rmse)

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
        print(f"\nEntraînement du modèle : {name}")
        model, score, params = train_func(X_train, y_train)

        # Appel de evaluate_model
        mse, rmse, y_test_inv, y_pred_inv = evaluate_model(model, X_test, y_test, scaler)

        print(f"{name} → MSE : {mse:.4f}, RMSE : {rmse:.4f}")

        results[name] = {
            "model": model,
            "mse": mse,
            "rmse": rmse,
            "params": params,
            "y_test": y_test_inv,
            "y_pred": y_pred_inv
        }

    # Trouver le meilleur modèle (basé sur RMSE)
    best_model_name = min(results, key=lambda k: results[k]['rmse'])
    best = results[best_model_name]

    print(f"\nMeilleur modèle : {best_model_name}")
    print(f"RMSE : {best['rmse']:.4f}, MSE : {best['mse']:.4f}")
    print(f"Meilleurs paramètres : {best['params']}")

    return best_model_name, best['model'], best

best_model_name, best_model, best_result = run_all_models(X_train, y_train, X_test, y_test, scaler)

def plot_predictions_vs_real(real_series, y_train_len, y_pred, label='Prédictions', n_days=30):
    """
    real_series : toutes les vraies valeurs (ex : close ou returns, non-scalées)
    y_train_len : longueur du jeu d'entraînement (pour placer la prédiction dans le temps)
    y_pred      : les prédictions du modèle à afficher
    label       : nom du modèle pour la légende
    n_days      : taille de la fenêtre utilisée dans create_target_features (important pour aligner)
    """
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

# Exemple avec Apple, modèle gagnant issu de run_all_models
real_returns = close_prices['Apple'].pct_change().dropna().values[n_days:]  # pour être aligné avec X et y

plot_predictions_vs_real(
    real_series=real_returns,
    y_train_len=len(y_train),
    y_pred=best_result['y_pred'],
    label=f'Prédictions {best_model_name}',
    n_days=30
)
