from MLP import MLP, loadData


if  __name__ == "__main__":
    lista_capas = [3,3]  # [2,3] a veces converge, a veces no
    cant_entradas = 4
    [x,y] = loadData("iris81_trn.csv",cant_entradas)  # Cargar datos de entrada
    mlp = MLP(lista_capas, cant_entradas,0.01,100)
    mlp.entrenamiento(x,y)

    [x,y] = loadData("iris81_tst.csv",cant_entradas)  # Cargar datos de entrada
    mlp.test(x,y)
    print("Tasa de error en test:", mlp.test(x,y))