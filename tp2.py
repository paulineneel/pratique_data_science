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
