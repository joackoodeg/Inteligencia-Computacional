#Ejercicio 2: Dise ̃ne e implemente un algoritmo gen ́etico que busque el mejor sub-
#conjunto de caracter ́ısticas para la clasificaci ́on de c ́ancer (leucemia linfoc ́ıtica
#aguda y leucemia miel ́ogena aguda) a partir de datos gen ́omicos. Se proveen
#38 muestras en el conjunto de entrenamiento y 34 en el conjunto de prueba
#(leukemia train.csv y leukemia test.csv, respectivamente). Cada muestra
#se compone de un total de 7129 caracter ́ısticas, que corresponden a valores de
#expresi ́on g ́enica.

import numpy as np
import bitstring as bs
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df_train = pd.read_csv('leukemia_train.csv')
x_train = df_train.iloc[:, :-1]
y_train = df_train.iloc[:, -1]

df_test = pd.read_csv('leukemia_test.csv')
x_test = df_test.iloc[:, :-1]
y_test = df_test.iloc[:, -1]

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def funcion_aptitud(individuo):
    if sum(individuo.genotipo) == 0: #Si no selecciona ninguna caracteristica, aptitud 0
        return 0
    
    caracteristicas_seleccionadas = individuo.fenotipo()
    #print("Caracteristicas seleccionadas: ", caracteristicas_seleccionadas)
    #print("Cantidad de caracteristicas seleccionadas: ", len(caracteristicas_seleccionadas))

    x_train_sel = x_train.iloc[:, caracteristicas_seleccionadas].to_numpy()
    x_test_sel = x_test.iloc[:, caracteristicas_seleccionadas].to_numpy()

    clasificador = MLPClassifier(hidden_layer_sizes=(10,), max_iter=100)
    clasificador.fit(x_train_sel, y_train)
    y_pred = clasificador.predict(x_test_sel)

    exactitud = accuracy_score(y_test, y_pred)
    
    #Aptitud
    alpha = 0.1
    total_caracteristicas = x_train.shape[1]
    f = exactitud - alpha * (len(caracteristicas_seleccionadas) / total_caracteristicas) #AJUSTAR 

    return  f

def fenotipo(individuo): #Devuelve columnas de caracteristicas seleccionadas
    fenotipo = []
    for i in range(individuo.genSize):
        if individuo.genotipo[i]:
            fenotipo.append(i)
    return fenotipo

def random_gen(tam=32):
    bits = bs.BitArray(tam) #cadena de 0's de tamaño tam

    #Selecciono 20 posiciones al azar para poner un 1
    k=20
    posiciones = random.sample(range(7129), k)
    for pos in posiciones:
        bits[pos] = 1

    return  bits

###################################################################################

def graficar_A(mA,pA):
    plt.plot(mA, label='Mejor Aptitud')
    plt.plot(pA, label='Promedio Aptitud')
    plt.xlabel('Generación')
    plt.ylabel('Aptitud')
    plt.title('Evolución de la Aptitud en el Algoritmo Genético')
    plt.legend()
    plt.grid()
    plt.show()


class individuo:
    def __init__(self, gen, size=32):
        self.genotipo = gen
        self.genSize = size
        self.aptitud = None
        self.fen = None

    def fenotipo(self): #Decodificacion
        if self.fen is None: #Evita recalcular fenotipo si ya fue calculado
            self.fen = fenotipo(self)
        return self.fen

    def evaluar(self):
        if self.aptitud is None: #Evita recalcular aptitud si ya fue calculada
            self.aptitud = funcion_aptitud(self)
        return self.aptitud

class poblacion:
    def __init__(self,cant_individuos,sizeGen=32): #Genera cantidad de individuos con genotipo aleatorio
        self.actualGeneracion = []
        for _ in range(cant_individuos):
            gen = random_gen(sizeGen)
            self.actualGeneracion.append(individuo(gen,sizeGen))
    
    def mejorAptitud(self): #Devuelve indice del individuo con mejor aptitud
        aptitudes = [self.actualGeneracion[i].evaluar() for i in range(len(self.actualGeneracion))]
        return np.argmax(aptitudes)

    def seleccion(self): #Seleccion por ventana deslizante N? padres
        progenitores = []
        aptitudes = [self.actualGeneracion[i].evaluar() for i in range(len(self.actualGeneracion))]
        generacionOrdenada = [x for _, x in sorted(zip(aptitudes, self.actualGeneracion), key=lambda t: t[0], reverse=True)]

        for extremoVentana in range(1,len(generacionOrdenada)): #Ventanas de seleccion
            indPadre = np.random.randint(0,extremoVentana)
            progenitores.append(generacionOrdenada[indPadre]) #devuelve N o N-1?
        
        indPadre = np.random.randint(0,len(generacionOrdenada)) 
        progenitores.append(generacionOrdenada[indPadre])
        #print("Cantidad de progenitores seleccionados: ", len(progenitores))
        return progenitores

    def cruza(self,progenitores): #Corta en dos partes cada padre y genera dos hijos
        self.proxGeneracion = []
        for i in range(0,len(progenitores),2):
            padre1 = progenitores[i]
            padre2 = progenitores[i+1]
            
            tam_gen = padre1.genSize #Se asume que ambos padres tienen el mismo tamaño de genotipo
            puntoCruza = np.random.randint(1,tam_gen-1) ###################################################

            hijo1_gen = padre1.genotipo[:puntoCruza] + padre2.genotipo[puntoCruza:]
            hijo2_gen = padre2.genotipo[:puntoCruza] + padre1.genotipo[puntoCruza:]
            
            hijo1 = individuo(hijo1_gen,tam_gen)
            hijo2 = individuo(hijo2_gen,tam_gen)
            
            self.proxGeneracion.append(hijo1)
            self.proxGeneracion.append(hijo2)
        #print("Cantidad de hijos generados: ", len(self.proxGeneracion))

    def mutacion(self,prob_mut=0.1): #Cada bit del genotipo tiene prob_mut de mutar
        for indHijo in range(len(self.proxGeneracion)):
            for indGen in range(self.proxGeneracion[indHijo].genSize):
                if random.random() < prob_mut:
                    self.proxGeneracion[indHijo].genotipo[indGen] = not self.proxGeneracion[indHijo].genotipo[indGen]
        

    def nuevaGeneracion(self): #Reemplaza la generacion actual por la nueva generacion
        self.actualGeneracion = self.proxGeneracion #Reemplazo total

def algoritmo_genetico(cant_individuos, MaxGen, aptitudRequerida, sizeGen=32):
        P = poblacion(cant_individuos,sizeGen)

        mejorAptitud = P.actualGeneracion[(P.mejorAptitud())].evaluar()
        iGen = 0
        promedioAptitud = np.mean([P.actualGeneracion[i].evaluar() for i in range(len(P.actualGeneracion))])
        print("Generacion: ", iGen, " Mejor aptitud: ", mejorAptitud, " Promedio aptitud: ", promedioAptitud)

        mA = [mejorAptitud]
        pA = [promedioAptitud]

        while(mejorAptitud<aptitudRequerida and iGen < MaxGen):
            
            progenitores = P.seleccion()

            P.cruza(progenitores)
            P.mutacion(prob_mut=0.01)

            P.nuevaGeneracion()
            
            mejorAptitud = P.actualGeneracion[(P.mejorAptitud())].evaluar()    
            iGen = iGen + 1
            promedioAptitud = np.mean([P.actualGeneracion[i].evaluar() for i in range(len(P.actualGeneracion))])
            print("Generacion: ", iGen, " Mejor aptitud: ", mejorAptitud, " Promedio aptitud: ", promedioAptitud)
            mA.append(mejorAptitud)
            pA.append(promedioAptitud)
        
        graficar_A(mA,pA)

        return P.actualGeneracion[(P.mejorAptitud())]

####################################################################################################################

if __name__ == "__main__":
    solucion = algoritmo_genetico(cant_individuos=100,MaxGen=50,aptitudRequerida=float('inf'),sizeGen=7129)
    print("Mejor solucion encontrada: Columnas = ", solucion.fenotipo(), " con aptitud: ", solucion.evaluar())