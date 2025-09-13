import matplotlib.pyplot as plt
import numpy as np

def graficar_voronoi_som(W, X, resolucion=300):
    """
    W: matriz de pesos (n_neuronas x 2) -> coordenadas de neuronas
    X: datos de entrada (n_datos x 2)
    resolucion: cantidad de puntos de grilla en cada dimensión
    """
    # Rango del espacio de datos
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5

    # Crear malla de puntos
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolucion),
                         np.linspace(y_min, y_max, resolucion))
    grid = np.c_[xx.ravel(), yy.ravel()]  # (resolucion^2, 2)

    # Calcular distancia de cada punto de la grilla a cada neurona
    distancias = np.linalg.norm(grid[:, None, :] - W[None, :, :], axis=2)  # (n_puntos, n_neuronas)

    # Neurona más cercana
    nearest = np.argmin(distancias, axis=1)

    # Dibujar regiones
    plt.figure(figsize=(8,8))
    plt.scatter(grid[:,0], grid[:,1], c=nearest, cmap="tab20", s=1, alpha=0.2)

    # Dibujar datos originales
    plt.scatter(X[:,0], X[:,1], c='blue', s=20, label="Datos X")

    # Dibujar neuronas
    plt.scatter(W[:,0], W[:,1], c='red', s=60, marker='X', label="Neuronas")

    plt.title("Mapa de Voronoi del SOM")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.show()

def graficar_som(W, mapa_neuronas, X, titulo="SOM"):
    """
    W: matriz de pesos (n_neuronas x dim_entrada)
    mapa_neuronas: matriz de índices de neuronas (filas x columnas)
    X: datos de entrada (n_datos x dim_entrada)
    """
    filas, columnas = mapa_neuronas.shape
    
    plt.clf()             # Limpiar la figura para la nueva trama
    plt.title(titulo)
    
    # Conexiones entre vecinos
    for i in range(filas):
        for j in range(columnas):
            idx = mapa_neuronas[i,j]
            for di in [-1,0,1]:
                for dj in [-1,0,1]:
                    ni, nj = i+di, j+dj
                    if (0 <= ni < filas) and (0 <= nj < columnas):
                        vecino_idx = mapa_neuronas[ni,nj]
                        if vecino_idx > idx:  # evitar doble línea
                            plt.plot([W[idx,0], W[vecino_idx,0]],
                                     [W[idx,1], W[vecino_idx,1]], 'k-', lw=0.8)

    # Graficar neuronas
    plt.scatter(W[:,0], W[:,1], c='red', s=50, label='Neurona')
    # Graficar datos de entrada
    plt.scatter(X[:,0], X[:,1], c='blue', s=20, alpha=0.6, label='Datos X')
    
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.legend()
    plt.pause(0.5)       # Espera 0.5 segundos
    plt.draw()           # Dibuja la actualización

# Uso interactivo
plt.ion()  # Modo interactivo

import matplotlib.pyplot as plt

# activamos modo interactivo
plt.ion()

def graficar_clusters(X, kmeans):
    """
    X: datos (n x d)
    kmeans: objeto K_means entrenado
    """
    plt.clf()  # limpia la ventana actual
    
    # colores por cluster
    colores = plt.cm.tab10.colors  
    
    # graficar puntos
    for i in range(X.shape[0]):
        cluster = int(kmeans.pertenece[i])
        plt.scatter(X[i,0], X[i,1], c=[colores[cluster]], marker='o', s=30)
    
    # graficar centroides
    for j, centroide in enumerate(kmeans.medias):
        plt.scatter(centroide[0], centroide[1], 
                    c=[colores[j]], marker='x', s=200, linewidths=3)
    
    plt.title("Clusters K-means")
    plt.draw()
    plt.pause(0.5)  # deja respirar el plot
