from MLP import MLP, loadData

if __name__ == "__main__":
    lista_capas = [2,1]  # 2 neuronas en la capa oculta, 1 en la capa de salida
    cant_entradas = 2
    mlp = MLP(lista_capas, cant_entradas,0.1,10)
    [x,y] = loadData("XOR_trn.csv",cant_entradas)  # Cargar datos de entrada
    
    mlp.entrenamiento(x,y)

    [x,y] = loadData("XOR_tst.csv",cant_entradas)  # Cargar datos de entrada
    
    mlp.test(x,y)
    print("Tasa de error en test:", mlp.test(x,y)*100,"%")