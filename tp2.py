from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def preprocess_for_financial_clustering(file_path):
    df = pd.read_csv(file_path)
    selected_columns = ['forwardPE', 'beta', 'priceToBook', 'returnOnEquity']
    df_selected = df[selected_columns].copy()
    df_selected.dropna(inplace=True)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_selected.T)
    return df_selected, df_scaled

def elbow_method(data):
    inertias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data.T)
        inertias.append(kmeans.inertia_)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), inertias, marker='o', color='b')
    plt.title("Méthode du coude pour déterminer le nombre de clusters")
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Inertie")
    plt.grid(True)
    plt.show()

def do_kmeans_clustering(data, n_clusters, affichage):
    df = data[0]
    df_scaled = data[1].T
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(df_scaled)
    data_with_clusters = df.copy()
    data_with_clusters.dropna(inplace=True)
    data_with_clusters.reset_index(drop=True, inplace=True)
    data_with_clusters.loc[:, 'Clusters'] = clusters
    if affichage:
        print("Caractéristiques des clusters :")
        cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=list(df.columns))
        print(cluster_centers)
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(df_scaled)
        tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
        tsne_df['clusters'] = clusters
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_df['TSNE1'], tsne_df['TSNE2'], c=tsne_df['clusters'], cmap='viridis', s=100)
        plt.title(f"Visualisation t-SNE des clusters avec KMeans ({n_clusters} clusters)")
        plt.colorbar(label='Cluster')
        plt.show()
    return clusters

def preprocess_for_risk_clustering(file_path):
    df = pd.read_csv(file_path)
    selected_columns = ['forwardPE', 'beta', 'priceToBook', 'returnOnEquity', 'debtToEquity',
                        'currentRatio', 'quickRatio', 'operatingMargins', 'profitMargins']
    df_selected = df[selected_columns].copy()
    df_selected.dropna(inplace=True)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_selected.T)
    return df_selected, df_scaled

def do_hierarchical_clustering(data, n_clusters, affichage):
    df = data[0]
    df.dropna(inplace=True)
    data_scaled = data[1].T
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    clusters = clustering.fit_predict(data_scaled)
    data_with_clusters = df.copy()
    data_with_clusters['Clusters'] = clusters
    if affichage:
        print("Caractéristiques des clusters :")
        print(data_with_clusters.groupby('Clusters').mean())
    return clusters

def plot_dendrogram(data_scaled):
    linked = linkage(data_scaled, method='ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linked)
    plt.title('Dendrogramme pour le Clustering Hiérarchique')
    plt.xlabel('Index des entreprises')
    plt.ylabel('Distance')
    plt.show()

def preprocess_for_daily_returns(file_path):
    rendements = {}
    filepaths = glob.glob(file_path)
    for filepath in filepaths:
        company_name = filepath.split('/')[-1].split('_')[0]
        df = pd.read_csv(filepath)
        rendements[company_name] = df['Rendement']
    returns_df = pd.DataFrame(rendements)
    returns_df.fillna(returns_df.mean(), inplace=True)
    return returns_df

def do_dbscan_clustering(data, eps, min_samples, affichage):
    returns_df = data[0]
    data_scaled = data[1].T
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data_scaled)
    if affichage:
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(data_scaled)
        cluster_df = pd.DataFrame({
            'Company': returns_df.columns,
            'TSNE1': tsne_results[:, 0],
            'TSNE2': tsne_results[:, 1],
            'Cluster': clusters
        })
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=cluster_df, x='TSNE1', y='TSNE2', hue='Cluster', palette='viridis', s=100)
        plt.title(f"Clusters DBSCAN (eps={eps}, min_samples={min_samples})")
        plt.legend(title='Cluster')
        plt.show()
    return clusters

def evaluate_clustering_algorithms():
    returns_df = preprocess_for_daily_returns("Companies_historical_data/*.csv").T
    data_dict = {
        'Finance': preprocess_for_financial_clustering("/content/df_ratios.csv"),
        'Risk': preprocess_for_risk_clustering("/content/df_ratios.csv"),
        'Returns': [returns_df, StandardScaler().fit_transform(returns_df).T]
    }
    silhouette_scores = []
    for data_name, data in data_dict.items():
        kmeans_clusters = do_kmeans_clustering(data, 3, False)
        kmeans_score = silhouette_score(data[0], kmeans_clusters) if len(set(kmeans_clusters)) > 1 else -1
        hierarchical_clusters = do_hierarchical_clustering(data, 3, False)
        hierarchical_score = silhouette_score(data[0], hierarchical_clusters) if len(set(hierarchical_clusters)) > 1 else -1
        dbscan_clusters = do_dbscan_clustering(data, 45, 3, False)
        dbscan_score = silhouette_score(data[0], dbscan_clusters) if len(set(dbscan_clusters)) > 1 else -1
        silhouette_scores.append({
            'Data': data_name,
            'KMeans': kmeans_score,
            'Hierarchical': hierarchical_score,
            'DBSCAN': dbscan_score
        })
    silhouette_df = pd.DataFrame(silhouette_scores)
    print(silhouette_df)
    return silhouette_df
