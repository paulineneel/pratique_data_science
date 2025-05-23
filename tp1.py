
import pandas as pd
import yfinance as yf
import glob
import os

ratios = {
"forwardPE": [],
"beta": [],
"priceToBook": [],
"priceToSales": [],
"dividendYield": [],
"trailingEps": [],
"debtToEquity": [],
"currentRatio": [],
"quickRatio": [],
"returnOnEquity": [],
"returnOnAssets": [],
"operatingMargins": [],
"profitMargins": []
}

companies = {
"Baidu": "BIDU",
"JD.com": "JD",
"BYD": "BYDDY",
"ICBC": "1398.HK",
"Toyota": "TM",
"SoftBank": "9984.T",
"Nintendo": "NTDOY",
"Hyundai": "HYMTF",
"Reliance Industries": "RELIANCE.NS",
"Tata Consultancy Services": "TCS.NS",
"Apple": "AAPL",
"Microsoft": "MSFT",
"Amazon": "AMZN",
"Alphabet": "GOOGL",
"Meta": "META",
"Tesla": "TSLA",
"NVIDIA": "NVDA",
"Samsung": "005930.KS",
"Tencent": "TCEHY",
"Alibaba": "BABA",
"IBM": "IBM",
"Intel": "INTC",
"Oracle": "ORCL",
"Sony": "SONY",
"Adobe": "ADBE",
"Netflix": "NFLX",
"AMD": "AMD",
"Qualcomm": "QCOM",
"Cisco": "CSCO",
"JP Morgan": "JPM",
"Goldman Sachs": "GS",
"Visa": "V",
"Johnson & Johnson": "JNJ",
"Pfizer": "PFE",
"ExxonMobil": "XOM",
"ASML": "ASML.AS",
"SAP": "SAP.DE",
"Siemens": "SIE.DE",
"Louis Vuitton (LVMH)": "MC.PA",
"TotalEnergies": "TTE.PA",
"Shell": "SHEL.L"
}

for c in companies:
  ticker = yf.Ticker(companies[c])
  for r in ratios:
    ratios[r].append(ticker.info.get(r))

df_ratios = pd.DataFrame(ratios)
df_ratios

# on met le nom des companies en index
df_ratios.index = companies.keys()
df_ratios.head()

# Exporter en CSV
df_ratios.to_csv('df_ratios.csv', index=False)

from google.colab import drive

# Monter Google Drive
drive.mount('/content/drive')

def historique(symbol, start, end):

  company_data = yf.download(symbol, start=start, end=end, progress=False)

  # Extraire la colonne 'Close'
  company_data.columns=company_data.columns.get_level_values(0)
  df = company_data[['Close']].copy()

  # Créer la colonne 'Next Day Close' (décalage des valeurs de 'Close')
  df.loc[:, 'Next Day Close'] = df['Close'].shift(-1)

  # Calculer le rendement : (Next Day Close - Close) / Close
  df.loc[:, 'Rendement'] = (df['Next Day Close'] - df['Close']) / df['Close']

  # Supprimer la dernière ligne qui contient un NaN dans 'Next Day Close'
  df.dropna(inplace=True)

  # Export CSV individuel
  filename = f"/content/drive/MyDrive/Companies_historical_data/{name.replace(' ', '_')}_historical_data.csv"
  df.to_csv(filename)

# Spécifier le chemin du dossier sur Google Drive
data_folder = "/content/drive/MyDrive/Companies_historical_data"

# Créer le dossier sur Google Drive s'il n'existe pas déjà
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

for name, symbol in companies.items():
  historique(symbol, '2019-01-01', '2024-01-01')

# Liste tous les fichiers CSV dans le dossier
csv_files = glob.glob(f"{data_folder}/*_historical_data.csv")

all_data = []

# Boucle pour les lire et ajouter une colonne "Company"
for file in csv_files:
    df = pd.read_csv(file)
    company_name = os.path.basename(file).split('_historical_data.csv')[0].replace('_', ' ')
    df["Company"] = company_name
    all_data.append(df)

# Concatène tous les DataFrames
df_all = pd.concat(all_data, ignore_index=True)

# Sauvegarde dans un seul fichier CSV
df_all.to_csv("Companies_historical_data.csv", index=False)

df_all.head()

"""# TP 2"""

from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform, pdist
import seaborn as sns

"""### Financial profiles clustering : K-MEANS"""

def preprocess_for_financial_clustering(file_path):
  # Charger le fichier des ratios financiers
  df = pd.read_csv(file_path)

  # Sélection des colonnes pertinentes pour le clustering
  selected_columns = ['forwardPE', 'beta', 'priceToBook', 'returnOnEquity']

  # Création du dataframe
  df_selected = df[selected_columns].copy()

  # Nettoyage des NA
  df_selected.dropna(inplace=True)

  # Standardiser les données
  scaler = StandardScaler()
  df_scaled = scaler.fit_transform(df_selected.T)

  return df_selected,df_scaled

def elbow_method(data):
  # Liste pour stocker les valeurs d'inertie
  inertias = []

  # Tester différents nombres de clusters (par exemple de 1 à 10)
  for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data.T)  # Entraînement du modèle KMeans
    inertias.append(kmeans.inertia_)  # Enregistrer l'inertie

  # Tracer l'inertie en fonction du nombre de clusters
  plt.figure(figsize=(8, 6))
  plt.plot(range(1, 11), inertias, marker='o', color='b')
  plt.title("Méthode du coude pour déterminer le nombre de clusters")
  plt.xlabel("Nombre de clusters")
  plt.ylabel("Inertie")
  plt.xticks(range(1, 11))
  plt.grid(True)
  plt.show()

elbow_method(preprocess_for_financial_clustering("/content/df_ratios.csv")[1])

"""On peut dire que le "coude" se situe à k=3"""

def do_kmeans_clustering(data, n_clusters, affichage):
    df = data[0]
    df_scaled = data[1].T

    # Appliquer KMeans avec le nombre de clusters choisi
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(df_scaled)  # On récupère les indices des clusters

    # Ajouter la colonne 'clusters' au DataFrame
    data_with_clusters = df.copy()
    data_with_clusters.dropna(inplace=True)
    # Réinitialiser l'index après avoir supprimé les NaN
    data_with_clusters.reset_index(drop=True, inplace=True)
    data_with_clusters.loc[:,'Clusters'] = clusters

    if(affichage):

      # Afficher les caractéristiques des clusters
      print("Caractéristiques des clusters (moyennes des ratios pour chaque cluster) :")
      cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=list(df.columns))
      print(cluster_centers)

      # Visualisation avec t-SNE pour une représentation 2D
      tsne = TSNE(n_components=2, random_state=0)
      tsne_results = tsne.fit_transform(df_scaled)  # Réduction de dimension

      # Créer un DataFrame avec les résultats de t-SNE
      tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
      tsne_df['clusters'] = clusters

      # Tracer les résultats
      plt.figure(figsize=(10, 8))
      plt.scatter(tsne_df['TSNE1'], tsne_df['TSNE2'], c=tsne_df['clusters'], cmap='viridis', s=100)
      plt.title(f"Visualisation t-SNE des clusters avec KMeans ({n_clusters} clusters)")
      plt.xlabel("t-SNE 1")
      plt.ylabel("t-SNE 2")
      plt.colorbar(label='Cluster')
      plt.show()

    return clusters

do_kmeans_clustering(preprocess_for_financial_clustering("/content/df_ratios.csv"), 3, True)

"""### Risk profiles clustering : Hierarchical Clustering"""

def preprocess_for_risk_clustering(file_path):
  # Charger le fichier des ratios financiers
  df = pd.read_csv(file_path)

  # Sélection des colonnes pertinentes pour le clustering
  selected_columns = ['forwardPE', 'beta', 'priceToBook', 'returnOnEquity', 'debtToEquity',
                        'currentRatio', 'quickRatio', 'operatingMargins', 'profitMargins']

  # Création du dataframe
  df_selected = df[selected_columns].copy()

  # Nettoyage des NA
  df_selected.dropna(inplace=True)

  # Standardiser les données
  scaler = StandardScaler()
  df_scaled = scaler.fit_transform(df_selected.T)

  return df_selected,df_scaled

def do_hierarchical_clustering(data, n_clusters, affichage):
    df = data[0]
    df.dropna(inplace=True)
    data_scaled = data[1].T

    # Appliquer le clustering hiérarchique
    hierarchical_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    clusters = hierarchical_clustering.fit_predict(data_scaled)

    # Ajouter la colonne 'clusters' au DataFrame
    data_with_clusters = df.copy()
    data_with_clusters.dropna(inplace=True)
    data_with_clusters['Clusters'] = clusters

    if(affichage):

      # Afficher les caractéristiques des clusters
      print("Caractéristiques des clusters (moyennes des ratios pour chaque cluster) :")
      cluster_summary = data_with_clusters.groupby('Clusters').mean()
      print(cluster_summary)

    return clusters

def plot_dendrogram(data_scaled):
    # Calculer les distances (linkage)
    linked = linkage(data_scaled, method='ward')

    # Tracer le dendrogramme
    plt.figure(figsize=(10, 7))
    dendrogram(linked)
    plt.title('Dendrogramme pour le Clustering Hiérarchique')
    plt.xlabel('Index des entreprises')
    plt.ylabel('Distance')
    plt.show()

do_hierarchical_clustering(preprocess_for_risk_clustering("/content/df_ratios.csv"), 3, True)

plot_dendrogram(preprocess_for_risk_clustering("/content/df_ratios.csv")[1])

"""### Daily returns correlations clustering : Hierarchical Clustering"""

def preprocess_for_daily_returns(file_path):

  rendements = {}

  # Accéder à tous les fichiers CSV dans le répertoire 'companieshistoric'
  filepaths = glob.glob(file_path)

  for filepath in filepaths:
      # Extraire le nom de l'entreprise à partir du nom du fichier
      company_name = filepath.split('/')[-1].split('_')[0]
      # Charger les données CSV
      df = pd.read_csv(filepath)
      rendements[company_name] = df['Rendement']

  returns_df = pd.DataFrame(rendements)
  # Nettoyer les valeurs manquantes (avec la moyenne de chaque colonne)
  returns_df.fillna(returns_df.mean(), inplace=True)

  returns_df = pd.DataFrame(returns_df)

  return returns_df

returns_df = preprocess_for_daily_returns("Companies_historical_data/*.csv")

corr_matrix = returns_df.corr()

do_hierarchical_clustering((corr_matrix, corr_matrix.values), 3, True)

plot_dendrogram(corr_matrix)

def do_dbscan_clustering(data, eps, min_samples, affichage):
    returns_df = data[0]
    data_scaled = data[1].T

    # Appliquer DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data_scaled)

    if(affichage):

      # Réduction dimensionnelle pour affichage
      tsne = TSNE(n_components=2, random_state=0)
      tsne_results = tsne.fit_transform(data_scaled)

      # DataFrame des résultats
      cluster_df = pd.DataFrame({
          'Company': returns_df.columns,
          'TSNE1': tsne_results[:,0],
          'TSNE2': tsne_results[:,1],
          'Cluster': clusters
      })

      # Plot
      plt.figure(figsize=(10,8))
      sns.scatterplot(data=cluster_df, x='TSNE1', y='TSNE2', hue='Cluster', palette='viridis', s=100)
      plt.title(f"Clusters DBSCAN (eps={eps}, min_samples={min_samples})")
      plt.legend(title='Cluster')
      plt.show()

    return(clusters)

returns_df = preprocess_for_daily_returns("Companies_historical_data/*.csv")
do_dbscan_clustering([returns_df,StandardScaler().fit_transform(returns_df)], 45, 3, True)

from sklearn.metrics import silhouette_score

# Dictionnaire contenant les DataFrames des différentes données déjà standardisées
returns_df = preprocess_for_daily_returns("Companies_historical_data/*.csv").T
data_dict = {
    'Finance': preprocess_for_financial_clustering("/content/df_ratios.csv"),
    'Risk': preprocess_for_risk_clustering("/content/df_ratios.csv"),
    'Returns': [returns_df,StandardScaler().fit_transform(returns_df).T]
}

silhouette_scores = []
n_clusters = 3
eps = 45
min_samples = 3

# Appliquer les trois algorithmes sur chaque jeu de données
for data_name, data in data_dict.items():
    # Appliquer KMeans et calculer le silhouette score
    kmeans_clusters = do_kmeans_clustering(data, n_clusters, False)
    kmeans_score = silhouette_score(data[0], kmeans_clusters) if len(set(kmeans_clusters)) > 1 else -1

    # Appliquer Hierarchical Clustering et calculer le silhouette score
    hierarchical_clusters = do_hierarchical_clustering(data, n_clusters, False)
    hierarchical_score = silhouette_score(data[0], hierarchical_clusters) if len(set(hierarchical_clusters)) > 1 else -1

    # Appliquer DBSCAN et calculer le silhouette score
    dbscan_clusters = do_dbscan_clustering(data, eps, min_samples, False)
    dbscan_score = silhouette_score(data[0], dbscan_clusters) if len(set(dbscan_clusters)) > 1 else -1

    # Ajouter les résultats au tableau des scores de silhouette
    silhouette_scores.append({
          'Data': data_name,
          'KMeans': kmeans_score,
          'Hierarchical': hierarchical_score,
          'DBSCAN': dbscan_score
      })

# Convertir les résultats en DataFrame
silhouette_df = pd.DataFrame(silhouette_scores)

print(silhouette_df)

"""Les résultats montrent que pour les données financières et de risque, tous les algorithmes donnent des scores de silhouette faibles, avec DBSCAN produisant un score de -1, indiquant que tous les points sont considérés comme des anomalies (normal étant donné que les paramètres ont été choisis pour fitter les rendements). En revanche, pour les rendements, DBSCAN a un meilleur score 0.19, suggérant une séparation plus efficace des clusters par rapport à KMeans et Hierarchical.

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
