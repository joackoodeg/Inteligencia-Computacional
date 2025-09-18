import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score #probar otros scores
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import davies_bouldin_score

if __name__ == "__main__":
    
    df = pd.read_csv("iris81_trn.csv")
    X = df.iloc[:,:4]
    X = X.values
   
    Y = df.iloc[:,4:-1]
    y_true = np.argmax(Y.values, axis=1)


    r = [2,20] # Rango de K a probar
    
    puntuaciones  = []
    puntuaciones2 = []
    puntuaciones3 = []

    for k in range(r[0],r[1]):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)

        silhouette_k = silhouette_score(X,kmeans.labels_)
        puntuaciones.append(silhouette_k)
      
        rand_k = adjusted_rand_score(y_true, kmeans.labels_)
        puntuaciones2.append(rand_k)
        
        davies_k = davies_bouldin_score(X, kmeans.labels_)
        puntuaciones3.append(davies_k)

    # Crear figura con 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    axs[0].plot(range(r[0], r[1]), puntuaciones, marker='o')
    axs[0].set_title("Silhouette vs K")
    axs[0].set_xlabel("K")
    axs[0].set_ylabel("Silhouette")

    axs[1].plot(range(r[0], r[1]), puntuaciones3, marker='o', color='orange')
    axs[1].set_title("Davies–Bouldin vs K")
    axs[1].set_xlabel("K")
    axs[1].set_ylabel("Davies–Bouldin")

    axs[2].plot(range(r[0], r[1]), puntuaciones2, marker='o', color='green')
    axs[2].set_title("Adjusted Rand Index vs K")
    axs[2].set_xlabel("K")
    axs[2].set_ylabel("ARI")

    plt.tight_layout()  # Ajusta espacios
    plt.show()

    input("Presione Enter para continuar...")

#| Métrica             | Supervisada | Lo que indica                     | Mejores valores |
#| ------------------- | ----------- | --------------------------------- | --------------- |
#| Silhouette          | No          | Cohesión y separación             | Cercano a 1     |
#| Davies–Bouldin      | No          | Compactación vs separación        | Cercano a 0     |
#| Adjusted Rand (ARI) | Sí          | Coincidencia con etiquetas reales | Cercano a 1     |
