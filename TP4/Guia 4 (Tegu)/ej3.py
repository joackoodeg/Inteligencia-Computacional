import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score #probar otros scores

if __name__ == "__main__":
    
    df = pd.read_csv("iris81_trn.csv")
    X = df.iloc[:,:4]
    X = X.values
    X = np.unique(X, axis=0) # Eliminar duplicados, el score daba valores raros

    Y = df.iloc[:,4:-1]
    Y = Y.values

    r = [2,X.shape[0]] # Rango de K a probar
    
    puntuaciones = []
    for k in range(r[0],r[1]):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        silhouette_k = silhouette_score(X,kmeans.labels_)
        print("K = ",k," -> Puntacion silhoutte = ", silhouette_k)
        puntuaciones.append(silhouette_k)

    plt.figure()
    plt.plot(range(r[0],r[1]),puntuaciones, marker='o')
    plt.title("Puntuacion Silhouette vs K")
    plt.show()
    input("Presione Enter para finalizar...")