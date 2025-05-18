
# TP 3
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import shap
import pandas as pd
import yfinance as yf
import glob
import os

!pip install ta

import ta

filepaths = glob.glob("Companies_historical_data/*.csv")

# Liste pour stocker les dataframes des entreprises
liste_dataframes = []

for filepath in filepaths:

  company_returns = pd.read_csv(filepath)[['Close']]
  company_returns['Close Horizon'] = company_returns['Close'].shift(-20)
  company_returns['Horizon Return'] = (company_returns['Close Horizon'] - company_returns['Close']) / company_returns['Close']
  # > 5% → 2 (buy), < -5% → 0 (sell), sinon → 1 (hold)
  company_returns['label'] = company_returns['Horizon Return'].apply(
      lambda x: 2 if x > 0.05 else (0 if x < -0.05 else 1)
  )

  # Ajoute les indicateurs techniques avec ta
  company_returns['SMA 20'] = ta.trend.sma_indicator(company_returns['Close'], window=20)
  company_returns['EMA 20'] = ta.trend.ema_indicator(company_returns['Close'], window=20)
  company_returns['RSI 14'] = ta.momentum.rsi(company_returns['Close'], window=14)

  macd = ta.trend.macd(company_returns['Close'])
  macd_signal = ta.trend.macd_signal(company_returns['Close'])
  company_returns['MACD'] = macd
  company_returns['MACD Signal'] = macd_signal

  bollinger_high = ta.volatility.bollinger_hband(company_returns['Close'], window=20)
  bollinger_low = ta.volatility.bollinger_lband(company_returns['Close'], window=20)
  company_returns['Bollinger High'] = bollinger_high
  company_returns['Bollinger Low'] = bollinger_low

  company_returns['Rolling Volatility 20'] = company_returns['Close'].rolling(window=20).std()
  company_returns['ROC 10'] = ta.momentum.roc(company_returns['Close'], window=10)

  # Garde trace de l'entreprise via le nom du fichier
  #company_returns['Company'] = os.path.basename(filepath).replace('.csv', '').split('_')[0]

  # Ajoute ce dataframe à la liste
  liste_dataframes.append(company_returns)

# Concatène tous les dataframes en un seul grand dataframe
base_donnees = pd.concat(liste_dataframes, ignore_index=True)

# Affiche les dernières lignes pour vérifier
print(base_donnees.tail())

"""Il est normal d'avoir des NaN pour Close Horizon et Horizon Return car on a affiché des valeurs qui font partie des 20 dernières. On a cependant choisi de faire comme ça car sinon c'est pour les autres colonnes que les premières valeurs sont NaN"""

# y : la cible (label)
y = base_donnees['label']

# X : toutes les colonnes sauf la cible et les colonnes à exclure
X = base_donnees.drop(columns=['label', 'Close Horizon', 'Weekly return', 'Next Day Close'], errors='ignore')

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Colonnes de X :", X.columns.tolist())

# Supprimer les NaN
X.dropna(inplace=True)
y = y[X.index]

# Standardiser les données de X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser en données d'entraînement et de test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

"""Verification

"""

print("X_train shape :", X_train.shape)
print("X_test shape  :", X_test.shape)
print("y_train distribution :")
print(y_train.value_counts(normalize=True))
print("y_test distribution :")
print(y_test.value_counts(normalize=True))

results = []

"""**XGBoost**"""

def xgboost_classifier(X, X_train, y_train, X_test, y_test):

    # Définir le modèle XGBoost
    xgb_model = XGBClassifier(random_state=42)

    # Paramètres à tester pour le GridSearch
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    # Initialiser GridSearchCV
    grid_search = GridSearchCV(xgb_model, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

    # Effectuer la recherche sur les hyperparamètres
    grid_search.fit(X_train, y_train)

    # Obtenir le meilleur modèle
    best_model = grid_search.best_estimator_
    print("Meilleurs hyperparamètres : ", grid_search.best_params_)

    # Prédictions sur l'ensemble de test
    y_pred = best_model.predict(X_test)

    # Afficher le classification report
    print("Classification Report :\n")
    print(classification_report(y_test, y_pred))

    # Afficher les shap.summary plot
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    # Prediction "buy" :
    shap.summary_plot(shap_values[:, :, 2], X_test, feature_names=X.columns)
    # Prediction "Sell" :
    shap.summary_plot(shap_values[:, :, 0], X_test, feature_names=X.columns)

    # Calculer l'accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculer le Macro F1
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    # Calculer la Precision, Recall et F1 pour la classe "Buy" (Supposons que "Buy" soit la classe 2)
    precision_buy = precision_score(y_test, y_pred, labels=[2], average='macro', zero_division=0)
    recall_buy = recall_score(y_test, y_pred, labels=[2], average='macro', zero_division=0)
    f1_buy = f1_score(y_test, y_pred, labels=[2], average='macro')

    results.append({
        'Modèle': 'XGBoost',
        'Accuracy': round(accuracy, 3),
        'Macro F1': round(macro_f1, 3),
        'Precision (Buy)': round(precision_buy, 3),
        'Recall (Buy)': round(recall_buy, 3),
        'F1 (Buy)': round(f1_buy, 3)
    })

xgboost_classifier(X, X_train, y_train, X_test, y_test)

"""**Random Forest**"""

def random_forest_classifier(X, X_train, y_train, X_test, y_test):

    # Définir le modèle Random Forest
    rf_model = RandomForestClassifier(random_state=42)

    # Paramètres à tester pour le GridSearch
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
    }

    # Initialiser GridSearchCV
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)

    # Effectuer la recherche sur les hyperparamètres
    grid_search.fit(X_train, y_train)

    # Obtenir le meilleur modèle
    best_model = grid_search.best_estimator_
    print("Meilleurs hyperparamètres : ", grid_search.best_params_)

    # Prédictions sur l'ensemble de test
    y_pred = best_model.predict(X_test)

    # Afficher le classification report
    print("Classification Report :\n")
    print(classification_report(y_test, y_pred))

    # Afficher les shap.summary plot
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    # Prediction "buy" :
    shap.summary_plot(shap_values[:, :, 2], X_test, feature_names=X.columns)
    # Prediction "Sell" :
    shap.summary_plot(shap_values[:, :, 0], X_test, feature_names=X.columns)

    # Calculer l'accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculer le Macro F1
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    # Calculer la Precision, Recall et F1 pour la classe "Buy" (Supposons que "Buy" soit la classe 2)
    precision_buy = precision_score(y_test, y_pred, labels=[2], average='macro', zero_division=0)
    recall_buy = recall_score(y_test, y_pred, labels=[2], average='macro', zero_division=0)
    f1_buy = f1_score(y_test, y_pred, labels=[2], average='macro')

    results.append({
        'Modèle': 'Random Forest',
        'Accuracy': round(accuracy, 3),
        'Macro F1': round(macro_f1, 3),
        'Precision (Buy)': round(precision_buy, 3),
        'Recall (Buy)': round(recall_buy, 3),
        'F1 (Buy)': round(f1_buy, 3)
    })

# temps d'exécution : environ 5min
random_forest_classifier(X, X_train, y_train, X_test, y_test)

"""**K-Nearest Neighbors (KNN)**"""

import shap
shap.initjs()

def knn_classifier(X, X_train, y_train, X_test, y_test):

    # Définir le modèle KNN
    knn_model = KNeighborsClassifier()

    # Paramètres à tester pour le GridSearch
    param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
    }

    # Initialiser GridSearchCV
    grid_search = GridSearchCV(estimator=knn_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)

    # Effectuer la recherche sur les hyperparamètres
    grid_search.fit(X_train, y_train)

    # Obtenir le meilleur modèle
    best_model = grid_search.best_estimator_
    print("Meilleurs hyperparamètres : ", grid_search.best_params_)

    # Prédictions sur l'ensemble de test
    y_pred = best_model.predict(X_test)

    # Afficher le classification report
    print("Classification Report :\n")
    print(classification_report(y_test, y_pred))

    # Afficher les shap.summary plot
    explainer = shap.KernelExplainer(best_model.predict_proba, X_train) # Use predict_proba for multi-class
    shap_values = explainer.shap_values(X_test)
    # Prediction "buy" :
    shap.summary_plot(shap_values[:, :, 2], X_test, feature_names=X.columns)
    # Prediction "Sell" :
    shap.summary_plot(shap_values[:, :, 0], X_test, feature_names=X.columns)

    # Calculer l'accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculer le Macro F1
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    # Calculer la Precision, Recall et F1 pour la classe "Buy" (Supposons que "Buy" soit la classe 2)
    precision_buy = precision_score(y_test, y_pred, labels=[2], average='macro', zero_division=0)
    recall_buy = recall_score(y_test, y_pred, labels=[2], average='macro', zero_division=0)
    f1_buy = f1_score(y_test, y_pred, labels=[2], average='macro')

    results.append({
        'Modèle': 'KNN',
        'Accuracy': round(accuracy, 3),
        'Macro F1': round(macro_f1, 3),
        'Precision (Buy)': round(precision_buy, 3),
        'Recall (Buy)': round(recall_buy, 3),
        'F1 (Buy)': round(f1_buy, 3)
    })

knn_classifier(X, X_train, y_train, X_test, y_test)

#####################################################

"""**Régression Logistique**"""

def logistic_regression_classifier(X_train, y_train, X_test, y_test):

    # Définir le modèle Logistic Regression
    logreg_model = LogisticRegression(max_iter=1000, random_state=42)

    # Paramètres à tester pour le GridSearch
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'solver': ['saga']
    }

    # Initialiser GridSearchCV
    grid_search = GridSearchCV(estimator=logreg_model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

    # Effectuer la recherche sur les hyperparamètres
    grid_search.fit(X_train, y_train)

    # Obtenir le meilleur modèle
    best_model = grid_search.best_estimator_
    print("Meilleurs hyperparamètres : ", grid_search.best_params_)

    # Prédictions sur l'ensemble de test
    y_pred = best_model.predict(X_test)

    # Afficher le classification report
    print("\nClassification Report :\n")
    print(classification_report(y_test, y_pred))

    # Afficher les shap.summary plot
    explainer = shap.LinearExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    # Prediction "buy" :
    shap.summary_plot(shap_values[:, :, 2], X_test, feature_names=X_test.columns)
    # Prediction "Sell" :
    shap.summary_plot(shap_values[:, :, 0], X_test, feature_names=X_test.columns)

    results.append({
        'Modèle': 'Logistic Regression',
        'Accuracy': round(accuracy, 3),
        'Macro F1': round(macro_f1, 3),
        'Precision (Buy)': round(precision_buy, 3),
        'Recall (Buy)': round(recall_buy, 3),
        'F1 (Buy)': round(f1_buy, 3)
    })

logistic_regression_classifier(X_train, y_train, X_test, y_test)

"""**SVM**"""

def svm_classifier(X_train, y_train, X_test, y_test):

    # Définir le modèle SVM
    svm_model = SVC(probability=True, random_state=42)

    # Paramètres à tester pour le GridSearch
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
    }

    # Initialiser GridSearchCV
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

    # Effectuer la recherche sur les hyperparamètres
    grid_search.fit(X_train, y_train)

    # Obtenir le meilleur modèle
    best_model = grid_search.best_estimator_
    print("Meilleurs hyperparamètres : ", grid_search.best_params_)

    # Prédictions sur l'ensemble de test
    y_pred = best_model.predict(X_test)

    # Afficher le classification report
    print("\nClassification Report :\n")
    print(classification_report(y_test, y_pred))

    # Afficher les shap.summary plot
    explainer = shap.KernelExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    # Prediction "buy" :
    shap.summary_plot(shap_values[:, :, 2], X_test, feature_names=X_test.columns)
    # Prediction "Sell" :
    shap.summary_plot(shap_values[:, :, 0], X_test, feature_names=X_test.columns)

    results.append({
        'Modèle': 'SVM',
        'Accuracy': round(accuracy, 3),
        'Macro F1': round(macro_f1, 3),
        'Precision (Buy)': round(precision_buy, 3),
        'Recall (Buy)': round(recall_buy, 3),
        'F1 (Buy)': round(f1_buy, 3)
    })

"""**Comparaisons**"""

random_forest_classifier(X_train, y_train, X_test, y_test)
knn_classifier(X_train, y_train, X_test, y_test)
logistic_regression_classifier(X_train, y_train, X_test, y_test)
svm_classifier(X_train, y_train, X_test, y_test)

def display_performance_table():
    df_perf = pd.DataFrame(performances)
    display(df_perf.sort_values(by="Macro F1", ascending=False))

def run_model(name, model, param_grid, X_train, y_train, X_test, y_test, shap_type='tree'):
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import shap
    import pandas as pd
    import matplotlib.pyplot as plt

    print(f"\nTraining model: {name}")

    clf = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    clf.fit(X_train, y_train)
    best_model = clf.best_estimator_

    y_pred = best_model.predict(X_test)

    print(f"\n {name} - Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    # SHAP
    print(f"SHAP Summary Plots for {name}")
    shap.initjs()

    if shap_type == 'tree':
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)

        shap.summary_plot(shap_values[2], X_test, feature_names=X_test.columns, show=False)
        plt.title(f"{name} - SHAP Summary (Buy)")
        plt.show()

        shap.summary_plot(shap_values[0], X_test, feature_names=X_test.columns, show=False)
        plt.title(f"{name} - SHAP Summary (Sell)")
        plt.show()

    elif shap_type == 'kernel':
        # Échantillon test et fond
        sample = X_test.sample(100, random_state=0)
        sample = sample.reset_index(drop=True)  # pour éviter les indices décalés
        background = X_train.sample(100, random_state=0)
        background = background.reset_index(drop=True)

        # Explainer kernel
        explainer = shap.KernelExplainer(best_model.predict_proba, background)
        shap_values = explainer.shap_values(sample)

        # Vérification de forme
        assert shap_values[2].shape == sample.shape, f"SHAP shape mismatch: {shap_values[2].shape} vs {sample.shape}"
        print(f"SHAP values shape: {shap_values[2].shape}")
        print(f"Sample shape: {sample.shape}")

        # Plots SHAP
        shap.summary_plot(shap_values[2], sample, feature_names=sample.columns, show=False)
        plt.title(f"{name} - SHAP Summary (Buy)")
        plt.show()

        shap.summary_plot(shap_values[0], sample, feature_names=sample.columns, show=False)
        plt.title(f"{name} - SHAP Summary (Sell)")
        plt.show()


    else:
        print(f"SHAP type '{shap_type}' not supported.")
        return

    # Résultats stockés dans le tableau global
    results.append({
        'Model': name,
        'Best Params': clf.best_params_,
        'Train Score': best_model.score(X_train, y_train),
        'Test Score': best_model.score(X_test, y_test),
        'Precision (Buy)': report.get('2', {}).get('precision'),
        'Recall (Buy)': report.get('2', {}).get('recall'),
        'F1 (Buy)': report.get('2', {}).get('f1-score'),
        'Precision (Hold)': report.get('1', {}).get('precision'),
        'Recall (Hold)': report.get('1', {}).get('recall'),
        'F1 (Hold)': report.get('1', {}).get('f1-score'),
        'Precision (Sell)': report.get('0', {}).get('precision'),
        'Recall (Sell)': report.get('0', {}).get('recall'),
        'F1 (Sell)': report.get('0', {}).get('f1-score'),
        'F1 (Macro Avg)': report.get('macro avg', {}).get('f1-score')
    })

def run_all_models(X_train, y_train, X_test, y_test):
    # Random Forest
    run_model(
        name='Random Forest',
        model=RandomForestClassifier(),
        param_grid={
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        shap_type='tree'
    )

    # XGBoost
    run_model(
        name='XGBoost',
        model=XGBClassifier(eval_metric='mlogloss', use_label_encoder=False),
        param_grid={
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1]
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        shap_type='tree'
    )

    # KNN
    run_model(
        name='KNN',
        model=KNeighborsClassifier(),
        param_grid={
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        shap_type='kernel'
    )

    # Logistic Regression
    run_model(
        name='Logistic Regression',
        model=LogisticRegression(max_iter=1000),
        param_grid={
            'C': [0.01, 0.1, 1, 10],
            'solver': ['lbfgs'],
            'multi_class': ['multinomial']
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        shap_type='kernel'
    )

    # SVM
    run_model(
        name='SVM',
        model=SVC(probability=True),
        param_grid={
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        shap_type='kernel'
    )

run_all_models(X_train, y_train, X_test, y_test)

# Résumé des performances
results_df = pd.DataFrame(results)
print("\nRésumé des performances :")
print(results_df)
