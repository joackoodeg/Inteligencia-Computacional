import numpy as np
import pandas as pd
from graficacion import graficar_som, graficar_voronoi_som

def loadData(route):
    data = pd.read_csv(route)
    x = data.values
    return x

class SOM:
    def __init__(self, cant_entradas, filas_mapa, columnas_mapa):
        #Cada fila es el vector de pesos de una neurona, cada columna son los pesos asociados a una entrada
        self.W = np.random.uniform(-0.5,0.5,size=(filas_mapa*columnas_mapa,cant_entradas))

        self.filas = filas_mapa
        self.columnas = columnas_mapa
        self.mapa_neuronas = np.arange(0, filas_mapa * columnas_mapa ).reshape(filas_mapa, columnas_mapa) #Matriz que indica la posicion de cada neurona en el mapa
    
    def determinar_ganador(self,x): #Determina el indice de la neurona ganadora (la mas cercana al dato x)
        dist_min = float('inf')
        ind = -1
        for i in range(self.W.shape[0]):
            pesos_Ni = self.W[i,:]  
            dist_i = np.linalg.norm(x-pesos_Ni)
            if dist_i < dist_min:
                dist_min = dist_i
                ind = i
        return ind

    def adaptacion(self,x,indice_g,radio_vecindad,velocidad_aprendizaje): 
        indices = np.where(self.mapa_neuronas==indice_g)
        fil = indices[0][0]
        col = indices[1][0]
        
        #La submatriz va de las filas fV1 a fV2-1 y de las columnas cV1 a cV2-1
        fV1 = max(0,fil-radio_vecindad) 
        fV2 = min(self.filas,fil+radio_vecindad+1)

        cV1 = max(0,col-radio_vecindad)
        cV2 = min(self.columnas,col+radio_vecindad+1)
        
        vecindad = self.mapa_neuronas[fV1:fV2,cV1:cV2] #Submatriz del mapa de neuronas que representa la vecindad y cuyo centro es la neurona ganadora

        for i, j in np.ndindex(vecindad.shape):  
            k = vecindad[i, j]
            dW = velocidad_aprendizaje*(x-self.W[k,:])  
            self.W[k,:] = self.W[k,:] + dW
        #print("Indice ganador: ",indice_g, "W actualizada: ",self.W)
    
    def entrenamiento(self,X,epoca_max = 1000):
        paso = 10

        #Ordenamiento global: al principio los datos se mueven mucho hacia la distribucion de los datos y luego se estabilizan?
        for i in range(0,epoca_max): 
            if(i%paso==0):
                graficar_som(self.W,self.mapa_neuronas,X,titulo="SOM - Epoca "+str(i))
            
            radio_vecindad = int(max(self.columnas,self.filas)/2)
            velocidad_aprendizaje = 0.9
            
            for j in range(X.shape[0]):
                indG = self.determinar_ganador(X[j,:])
                self.adaptacion(X[j,:],indG,radio_vecindad,velocidad_aprendizaje)
            
            print("Ord global: Epoca ",i)


        #Transicion        
        rv = int(max(self.columnas,self.filas)/2)
        print("Radio vecindad inicial: ",rv)
        for i in range(0,epoca_max):
            radio_vecindad = int(np.round(rv + (1 - rv) * i / epoca_max))
            velocidad_aprendizaje = 0.9 - 0.8 * i / epoca_max
            for j in range(X.shape[0]):
                indG = self.determinar_ganador(X[j,:])
                self.adaptacion(X[j,:],indG,radio_vecindad,velocidad_aprendizaje)
            
            print("Transicion: Epoca ",i," Radio vecindad: ",radio_vecindad," Velocidad aprendizaje: ",velocidad_aprendizaje)
            if(i%paso==0):
                graficar_som(self.W,self.mapa_neuronas,X,titulo="SOM - Epoca "+str(i))

        #Ajuste fino
        for i in range(0,epoca_max):
            #Want = self.W

            radio_vecindad = 0
            velocidad_aprendizaje = 0.01
            for j in range(X.shape[0]):
                indG = self.determinar_ganador(X[j,:])
                self.adaptacion(X[j,:],indG,radio_vecindad,velocidad_aprendizaje)
            print("Ajuste Fino: Epoca ",i)

            #diferencia = np.linalg.norm(Want - self.W)
            #if(diferencia<1e-9): break #Convergencia
            if(i%paso==0):
                graficar_som(self.W,self.mapa_neuronas,X,titulo="SOM - Epoca "+str(i))

        #graficar_voronoi_som(self.W,X)
    
    def test(self,X):
        pertenece = np.zeros(X.shape[0])
        for i in range(X.shape[0]): #para cada dato
            ind_cluster = self.determinar_ganador(X[i,:])
            pertenece[i] = ind_cluster
        return pertenece

if __name__ == "__main__":
    route = 'te.csv'
    #route = 'circulo.csv'

    cant_entradas = 2
    som = SOM(cant_entradas,1,5)
    X = loadData(route)  # Cargar datos de entrada
    som.entrenamiento(X,250)
    input("Presione Enter para finalizar...")