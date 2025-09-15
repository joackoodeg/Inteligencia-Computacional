import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

def kmeans_readData(path):
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :-3].values   # primeras 4 columnas 
    Y_bin = df.iloc[:, -3:].values   # últimas 3 columnas 

    # transformar [-1,-1,1] -> 2, [-1,1,-1] -> 1, [1,-1,-1] -> 0
    y = np.argmax(Y_bin, axis=1)

    return X, y

def kmeans(X, k, max_iter=100, seed=0):
    np.random.seed(seed)
    n, d = X.shape

    # elegir k centroides iniciales al azar
    indices = np.random.choice(n, k, replace=False)
    centroids = X[indices]

    for _ in range(max_iter):
        # asignación de clusters
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            min_dist = float("inf")
            best = 0
            for j in range(k):
                dist = 0.0
                for dim in range(d):
                    diff = X[i, dim] - centroids[j, dim]
                    dist += diff * diff
                if dist < min_dist:
                    min_dist = dist
                    best = j
            labels[i] = best

        # actualización de centroides
        new_centroids = np.zeros((k, d))
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                for dim in range(d):
                    new_centroids[j, dim] = cluster_points[:, dim].mean()
            else:
                new_centroids[j] = centroids[j]  # si queda vacío

        # condición de convergencia
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids

def contingency_matrix(true_labels, pred_labels):
    uniq_true = np.unique(true_labels)
    uniq_pred = np.unique(pred_labels)
    M = np.zeros((len(uniq_true), len(uniq_pred)), dtype=int)
    for i, t in enumerate(uniq_true):
        for j, p in enumerate(uniq_pred):
            M[i, j] = np.sum((true_labels == t) & (pred_labels == p))
    return M

if __name__ == "__main__":
    X_train, y_train = kmeans_readData("iris81_trn.csv")
    labels_train, centroids = kmeans(X_train, k=3)

    print("Matriz entrenamiento (K-means vs clases):")
    print(contingency_matrix(y_train, labels_train))

    # Todas las combinaciones posibles de pares de dimensiones
    dims_list = list(itertools.combinations(range(X_train.shape[1]), 2))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, (d1, d2) in enumerate(dims_list):
        ax = axes[idx]
        ax.scatter(
            X_train[:, d1], 
            X_train[:, d2], 
            c=labels_train,         # <- acá van los clusters
            cmap="viridis", 
            s=30
        )
        ax.set_xlabel(f"Dim {d1}")
        ax.set_ylabel(f"Dim {d2}")
        ax.set_title(f"Clustering K-means ({d1} vs {d2})")

    plt.tight_layout()
    plt.show()
