import numpy as np
from kmeans import kmeans, kmeans_readData
from som import SOM
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    X_kmeans, y_kmeans = kmeans_readData("iris81_trn.csv")
    labels_kmeans, centroids = kmeans(X_kmeans, k=3)

    X_som = pd.read_csv("iris81_trn.csv", header=None).values   # 2 columnas (x,y)
    som = SOM(rows=3, cols=1, dim=X_som.shape[1], lr=0.01, r_init=1, seed=1)
    som.routine(X_som)
    
    labels_som = []
    for x in X_som:
        row, col = som.find(x)
        labels_som.append(row * som.cols + col)
    labels_som = np.array(labels_som)

    dims = [0, 2]  # ejemplo: sepal_length vs petal_length

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.scatter(X_kmeans[:,dims[0]], X_kmeans[:,dims[1]], c=labels_kmeans, s=30)
    plt.title("Clustering con K-means")

    plt.subplot(1,2,2)
    plt.scatter(X_som[:,dims[0]], X_som[:,dims[1]], c=labels_som, s=30)
    plt.title("Clustering con SOM")
    plt.show()
