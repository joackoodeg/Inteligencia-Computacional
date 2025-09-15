#Implemente el m ́etodo de clustering k-medias sobre el conjunto de
#datos Iris (GTP2) y compare las soluciones obtenidas con las de un SOM.
#Para esto obtenga las matrices de contingencia entre ambos m ́etodos y entre
#cada m ́etodo y las clases de referencia.
#Seleccione 2 dimensiones y grafique los datos coloreando cada punto seg ́un el
#grupo al que pertenece en la soluci ́on de k-medias y en la del SOM.
#Grafique las neuronas del SOM en 2D con una escala de colores seg ́un las
#frecuencias de activacion para los datos de Iris. Indique adem ́as cu ́al es la clase
#de Iris que cada neurona.

from sklearn.metrics.cluster import contingency_matrix
from KMeans import K_means, loadData
from SOM import SOM

if __name__ == "__main__":
    ruta = "iris81_trn.csv"
    X = loadData(ruta)
    X_trn = X[:,:4] 
    
    ruta = "iris81_tst.csv"
    X = loadData(ruta)
    X_tst = X[:,:4] 

    #K-MEANS
    k = 3
    kmeans = K_means()
    kmeans.entrenamiento(k,X_trn)
    kmeans_test = kmeans.test(X_tst)  
    input("Presione Enter para continuar con SOM...")
    
    #SOM
    som = SOM(4,5,5)
    som.entrenamiento(X_trn,epoca_max=500)
    som_test = som.test(X_tst) 
    input("Presione Enter para continuar con la matriz de contingencia...")


    print("Matriz de contingencia entre K-means y SOM:")
    print(contingency_matrix(kmeans_test,som_test)) 
    
    input("Fin del programa. Presione Enter para salir")