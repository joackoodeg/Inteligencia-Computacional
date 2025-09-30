#Ejercicio 1: Implemente las estructuras de datos y algoritmos basicos para la solucion
#de un problema mediante algoritmos gen ́eticos. Pruebe estas rutinas y
#compare los resultados con un metodo de gradiente descendiente para buscar
#el m ́ınimo global de las siguientes funciones:

# a) f(x) = −x sin(|x|)   con x ∈ [−512 . . . 512]

import numpy as np
import bitstring as bs
import random
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def f(x):
    return (-x * np.sin(np.abs(x)))

def funcion_aptitud(individuo):
    fenotipo = individuo.fenotipo()
    return  -1*(-fenotipo * np.sin(np.abs(fenotipo))) + 1000

def fenotipo(individuo):
    max_val = (2**individuo.genSize) - 1
    norm = individuo.genotipo.uint / max_val  #Normalizo a [0,1]
    fenotipo = norm * (512 - (-512)) + (-512) #Mapeo a [-512,512]
    return fenotipo

def random_gen(tam=32):
    gen = ''.join(str(random.randint(0,1)) for _ in range(tam))
    return bs.BitArray(bin=gen)

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
    
    def fenotipo(self): #Decodificacion
        return fenotipo(self)

    def evaluar(self):
        return funcion_aptitud(self)

class poblacion:
    def __init__(self,cant_individuos,sizeGen=32): #Genera cantidad de individuos con genotipo aleatorio
        self.actualGeneracion = []
        for _ in range(cant_individuos):
            gen = random_gen()
            self.actualGeneracion.append(individuo(gen,sizeGen))
    
    def mejorAptitud(self): #Devuelve indice del individuo con mejor aptitud
        aptitudes = [self.actualGeneracion[i].evaluar() for i in range(len(self.actualGeneracion))]
        return np.argmax(aptitudes)

    def seleccion(self): #Seleccion por ventana deslizante de N padres
        progenitores = []
        aptitudes = [self.actualGeneracion[i].evaluar() for i in range(len(self.actualGeneracion))]
        generacionOrdenada = [x for _, x in sorted(zip(aptitudes, self.actualGeneracion), key=lambda t: t[0], reverse=True)]

        for extremoVentana in range(1,len(generacionOrdenada)): #Ventanas de seleccion
            indPadre = np.random.randint(0,extremoVentana)
            progenitores.append(generacionOrdenada[indPadre]) #devuelve N-1
        
        indPadre = np.random.randint(0,len(generacionOrdenada)) 
        progenitores.append(generacionOrdenada[indPadre]) #por eso tome otro padre aca
        print("Cantidad de progenitores seleccionados: ", len(progenitores))
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
        print("Cantidad de hijos generados: ", len(self.proxGeneracion))

        

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
    solucion = algoritmo_genetico(cant_individuos=100,MaxGen=500,aptitudRequerida=float('inf'),sizeGen=32)
    print("Mejor solucion encontrada: ", solucion.fenotipo(), " con aptitud: ", solucion.evaluar(),"\n")
    res = minimize_scalar(f, bounds=(-512, 512), method='bounded')
    print("Solucion por gradiente descendiente: x=", res.x, " , y=",res.x , " con valor: ", res.fun)