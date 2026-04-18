"""
src/clustering.py
-----------------
Reusable K-Means clustering pipeline for RFM segmentation.
Handles feature preparation, model training, evaluation, and prediction.
"""

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA


# ── Feature Preparation ───────────────────────────────────────────────────────

def prepare_features(rfm: pd.DataFrame,
                     features: list = None,
                     log_transform: bool = True) -> tuple:
    """
    Prepare and scale RFM features for clustering.

    Steps:
        1. Select features (default: Recency, Frequency, Monetary)
        2. Apply log1p transform to reduce skewness (optional)
        3. Standardize with StandardScaler (mean=0, std=1)

    Parameters
    ----------
    rfm           : DataFrame with RFM columns
    features      : List of column names to use (default: R, F, M raw values)
    log_transform : Whether to apply log1p before scaling (default: True)

    Returns
    -------
    X_scaled  : np.ndarray of scaled features
    scaler    : Fitted StandardScaler instance
    features  : List of feature names used
    """
    if features is None:
        features = ['Recency', 'Frequency', 'Monetary']

    X = rfm[features].copy().values

    if log_transform:
        X = np.log1p(X)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler, features


# ── Optimal K Selection ───────────────────────────────────────────────────────

def find_optimal_k(X_scaled: np.ndarray,
                   k_range: range = range(2, 11),
                   random_state: int = 42,
                   plot: bool = True,
                   save_path: str = None) -> dict:
    """
    Find the optimal number of clusters using Elbow + Silhouette methods.

    Parameters
    ----------
    X_scaled     : Scaled feature matrix
    k_range      : Range of K values to evaluate
    random_state : Random seed for reproducibility
    plot         : Whether to display the evaluation plots
    save_path    : Optional path to save the plot image

    Returns
    -------
    dict with keys:
        'inertias'    : list of inertia values per K
        'silhouettes' : list of silhouette scores per K
        'best_k'      : K with the highest silhouette score
        'k_range'     : list of K values tested
    """
    inertias    = []
    silhouettes = []
    k_list      = list(k_range)

    print('Evaluating K values...')
    for k in k_list:
        km     = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouettes.append(sil)
        print(f'  K={k} | Inertia: {km.inertia_:,.1f} | Silhouette: {sil:.4f}')

    best_k = k_list[int(np.argmax(silhouettes))]
    print(f'\nBest K by Silhouette: {best_k} (score={max(silhouettes):.4f})')

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle('Optimal K Selection', fontsize=14, fontweight='bold')

        axes[0].plot(k_list, inertias, 'o-', color='#1D9E75', linewidth=2.5, markersize=8)
        axes[0].fill_between(k_list, inertias, alpha=0.1, color='#1D9E75')
        axes[0].set_title('Elbow Method')
        axes[0].set_xlabel('K')
        axes[0].set_ylabel('Inertia (WCSS)')
        axes[0].set_xticks(k_list)

        bars = axes[1].bar(k_list, silhouettes, color='#378ADD', alpha=0.85, edgecolor='white')
        best_idx = int(np.argmax(silhouettes))
        bars[best_idx].set_color('#D85A30')
        axes[1].set_title('Silhouette Score')
        axes[1].set_xlabel('K')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_xticks(k_list)
        axes[1].annotate(
            f'Best K={best_k}',
            xy=(best_k, silhouettes[best_idx]),
            xytext=(best_k + 0.4, silhouettes[best_idx] + 0.005),
            fontsize=10, color='#D85A30', fontweight='bold'
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    return {
        'inertias':    inertias,
        'silhouettes': silhouettes,
        'best_k':      best_k,
        'k_range':     k_list,
    }


# ── Model Training ────────────────────────────────────────────────────────────

def train_kmeans(X_scaled: np.ndarray,
                 n_clusters: int,
                 random_state: int = 42,
                 n_init: int = 20,
                 max_iter: int = 500) -> KMeans:
    """
    Train a K-Means model on scaled features.

    Parameters
    ----------
    X_scaled    : Scaled feature matrix
    n_clusters  : Number of clusters (K)
    random_state: Random seed
    n_init      : Number of initializations (higher = more stable)
    max_iter    : Maximum iterations

    Returns
    -------
    Fitted KMeans instance
    """
    km = KMeans(
        n_clusters   = n_clusters,
        random_state = random_state,
        n_init       = n_init,
        max_iter     = max_iter,
    )
    km.fit(X_scaled)

    sil = silhouette_score(X_scaled, km.labels_)
    print(f'K-Means trained | K={n_clusters} | Inertia={km.inertia_:,.2f} | Silhouette={sil:.4f}')

    return km


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_cluster(new_customers: pd.DataFrame,
                    scaler: StandardScaler,
                    kmeans: KMeans,
                    features: list = None,
                    log_transform: bool = True) -> np.ndarray:
    """
    Predict cluster assignment for new customers.

    Parameters
    ----------
    new_customers : DataFrame with same RFM columns used in training
    scaler        : Fitted StandardScaler from prepare_features()
    kmeans        : Fitted KMeans model from train_kmeans()
    features      : Feature column names (default: Recency, Frequency, Monetary)
    log_transform : Apply log1p before scaling (must match training)

    Returns
    -------
    np.ndarray of cluster labels
    """
    if features is None:
        features = ['Recency', 'Frequency', 'Monetary']

    X = new_customers[features].copy().values

    if log_transform:
        X = np.log1p(X)

    X_scaled = scaler.transform(X)
    return kmeans.predict(X_scaled)


# ── Cluster Profiling ─────────────────────────────────────────────────────────

def cluster_profile(rfm: pd.DataFrame,
                    cluster_col: str = 'Cluster') -> pd.DataFrame:
    """
    Generate a summary profile table for each cluster.

    Returns
    -------
    DataFrame sorted by Avg_Monetary descending with columns:
        Customers, Avg_Recency, Avg_Frequency, Avg_Monetary,
        Total_Revenue, Revenue_Share_%
    """
    total_rev = rfm['Monetary'].sum()

    profile = (
        rfm.groupby(cluster_col)
        .agg(
            Customers     = ('Monetary',  'count'),
            Avg_Recency   = ('Recency',   'mean'),
            Avg_Frequency = ('Frequency', 'mean'),
            Avg_Monetary  = ('Monetary',  'mean'),
            Total_Revenue = ('Monetary',  'sum'),
        )
        .round(1)
        .sort_values('Avg_Monetary', ascending=False)
    )

    profile['Revenue_Share_%'] = (
        profile['Total_Revenue'] / total_rev * 100
    ).round(1)

    return profile


# ── PCA Visualization ─────────────────────────────────────────────────────────

def plot_clusters_pca(X_scaled: np.ndarray,
                      labels: np.ndarray,
                      kmeans: KMeans,
                      colors: list = None,
                      save_path: str = None):
    """
    Plot clusters projected onto the first two principal components.

    Parameters
    ----------
    X_scaled  : Scaled feature matrix
    labels    : Cluster label array
    kmeans    : Fitted KMeans model (for centroid plotting)
    colors    : List of colors per cluster
    save_path : Optional path to save the plot
    """
    if colors is None:
        colors = ['#1D9E75', '#378ADD', '#D85A30', '#7F77DD', '#BA7517']

    pca    = PCA(n_components=2, random_state=42)
    X_pca  = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    n_clusters = len(np.unique(labels))

    fig, ax = plt.subplots(figsize=(10, 7))

    for i in range(n_clusters):
        mask = labels == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   s=25, alpha=0.5, color=colors[i % len(colors)],
                   label=f'Cluster {i}')

    centroids_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
               s=200, marker='X', color='black', zorder=5, label='Centroids')

    ax.set_title(f'Cluster Visualization — PCA (K={n_clusters})', fontweight='bold')
    ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}% variance)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ── Model Persistence ─────────────────────────────────────────────────────────

def save_model(kmeans: KMeans,
               scaler: StandardScaler,
               output_dir: str = '../outputs/models'):
    """
    Save trained KMeans model and scaler to disk as pickle files.

    Parameters
    ----------
    kmeans     : Fitted KMeans model
    scaler     : Fitted StandardScaler
    output_dir : Directory to save files (created if not exists)
    """
    os.makedirs(output_dir, exist_ok=True)

    km_path     = os.path.join(output_dir, 'kmeans_model.pkl')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')

    with open(km_path, 'wb') as f:
        pickle.dump(kmeans, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f'Model saved  → {km_path}')
    print(f'Scaler saved → {scaler_path}')


def load_model(model_dir: str = '../outputs/models') -> tuple:
    """
    Load saved KMeans model and scaler from disk.

    Parameters
    ----------
    model_dir : Directory containing kmeans_model.pkl and scaler.pkl

    Returns
    -------
    (kmeans, scaler) tuple
    """
    km_path     = os.path.join(model_dir, 'kmeans_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')

    with open(km_path, 'rb') as f:
        kmeans = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    print(f'Loaded KMeans  (K={kmeans.n_clusters})')
    print(f'Loaded Scaler  (features={scaler.n_features_in_})')

    return kmeans, scaler
