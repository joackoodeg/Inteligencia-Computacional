import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from kmeans import kmeans, kmeans_readData, contingency_matrix
from som import SOM

if __name__ == "__main__":
    # ---------------- DATOS ----------------
    X, y = kmeans_readData("iris81_trn.csv")

    # ---------------- K-MEANS ----------------
    labels_kmeans, centroids = kmeans(X, k=3)
    print("Matriz (K-means vs clases):")
    print(contingency_matrix(y, labels_kmeans))

    # ---------------- SOM ----------------
    som = SOM(rows=2, cols=3, dim=X.shape[1], lr=0.1, r_init=3, seed=1)
    som.routine(X)
    labels_som, activations, neuron_class = som.evaluate(X, y)

    # ---------------- PLOT A: datos en 2D ----------------
    dims = [0, 2]  # sepal_length vs petal_length

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.scatter(X[:,dims[0]], X[:,dims[1]], c=labels_kmeans, cmap="viridis", s=30)
    plt.title("Clustering con K-means")

    plt.subplot(1,2,2)
    plt.scatter(X[:,dims[0]], X[:,dims[1]], c=labels_som, cmap="viridis", s=30)
    plt.title("Clustering con SOM")
    plt.show()

    # ---------------- PLOT B: neuronas SOM (frecuencias) ----------------
    iris_names = ["Setosa", "Versicolor", "Virginica"]

    plt.figure(figsize=(6,6))
    plt.scatter(
        som.W[:,:,0], som.W[:,:,1],
        c=activations.flatten(), cmap="YlOrRd", s=400, marker="s", edgecolors="k"
    )
    for i in range(som.rows):
        for j in range(som.cols):
            if neuron_class[i,j] != -1:
                plt.text(
                    som.W[i,j,0], som.W[i,j,1],
                    iris_names[neuron_class[i,j]],
                    ha="center", va="center", color="black", fontsize=9,
                )

    plt.title("SOM")
    plt.colorbar(label="Frecuencia de activación")
    plt.axis("equal")
    plt.show()

    # ---------------- MATRIZ SOM ----------------
    print("Matriz (SOM vs clases):")
    print(contingency_matrix(y, labels_som))
