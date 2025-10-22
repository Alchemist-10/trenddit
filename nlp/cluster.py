# nlp/cluster.py
"""
Small clustering utility using k-means. Given embeddings array (n x d),
returns cluster labels and optionally cluster centers.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def cluster_embeddings(embeddings, n_clusters=5):
    """
    embeddings: numpy array shape (n, d)
    returns labels (n,) and centers (n_clusters, d)
    """
    if embeddings.shape[0] < n_clusters:
        n_clusters = max(1, embeddings.shape[0] // 2)
    if n_clusters < 1:
        labels = np.zeros(embeddings.shape[0], dtype=int)
        centers = np.zeros((1, embeddings.shape[1]))
        return labels, centers

    # optional PCA to speed up
    if embeddings.shape[1] > 50:
        pca = PCA(n_components=50)
        reduced = pca.fit_transform(embeddings)
    else:
        reduced = embeddings
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(reduced)
    return km.labels_, km.cluster_centers_
