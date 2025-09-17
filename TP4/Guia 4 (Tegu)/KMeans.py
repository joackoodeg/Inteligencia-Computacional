import numpy as np
import pandas as pd
from graficacion import graficar_clusters

def loadData(route):
    data = pd.read_csv(route)
    x = data.values
    return x

class K_means:
    def __init__(self):
        self.medias = []
        self.pertenece = [] #vector que indica a que cluster pertenece cada dato utilizado en el entrenamiento
    
    def entrenamiento(self,k,X):
        for i in range(k): #inicializar medias con los primeros k datos 
            self.medias.append(X[i,:])

        self.pertenece = np.zeros(X.shape[0])
        cambios = -1
        while(cambios!=0): #Hasta que no haya cambios
            cambios = 0
            for i in range(X.shape[0]): #para cada dato
                dist_min = float('inf')
                ind_cluster = -1
                for j in range(k): #buscar el cluster mas cercano
                    dist = np.linalg.norm(X[i,:]-self.medias[j])
                    if dist < dist_min:
                        dist_min = dist
                        ind_cluster = j
                
                if self.pertenece[i] != ind_cluster: #asignar el dato al cluster
                    cambios +=1
                    self.pertenece[i] = ind_cluster
            
            for j in range(k): #actualizar medias
                sum = np.zeros(X.shape[1])
                cant = 0
                for i in range(self.pertenece.shape[0]):
                    if self.pertenece[i]==j:
                        sum +=X[i,:]
                        cant+=1
                self.medias[j] = sum/cant
            
            graficar_clusters(X, self)
            print("Cambios: ",cambios)
    
    def test(self,X):
        pertenece = np.zeros(X.shape[0])
        for i in range(X.shape[0]): #para cada dato
            dist_min = float('inf')
            ind_cluster = -1
            for j in range(len(self.medias)): #buscar el cluster mas cercano
                dist = np.linalg.norm(X[i,:]-self.medias[j])
                if dist < dist_min:
                    dist_min = dist
                    ind_cluster = j
            pertenece[i] = ind_cluster
        return pertenece


if __name__ == "__main__":
    ruta = "te.csv"
    #ruta = 'circulo.csv'
    X = loadData(ruta)
    k = 4
    kmeans = K_means()
    kmeans.entrenamiento(k,X)
    input("Fin del programa. Presione Enter para salir")
    