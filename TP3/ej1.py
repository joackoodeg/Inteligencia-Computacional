import numpy as np
from sklearn import model_selection 
from sklearn import datasets
from sklearn import neural_network

if __name__ == "__main__":
    print("-------------------------------------")
    neuronas_ocultas = 20

    X, y = datasets.load_digits(n_class=10, return_X_y=True)
    #A
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.20,shuffle=True)

    MLP = neural_network.MLPClassifier(hidden_layer_sizes=(neuronas_ocultas),max_iter=500)
    MLP.fit(X_train,y_train)
    exactitud = MLP.score(X_test,y_test)
    print("Exactitud medida con una sola division train/test 80/20:",exactitud,"\n")
    
    #B
    indices_folds = model_selection.KFold(n_splits=5,shuffle=True)
    exactitudes = []
    for (train_ind,test_ind) in indices_folds.split(X,y):
        MLP = neural_network.MLPClassifier(hidden_layer_sizes=(neuronas_ocultas),max_iter=500)
        MLP.fit(X[train_ind],y[train_ind])
        exactitudes.append(MLP.score(X[test_ind],y[test_ind]))
    promedio = np.mean(exactitudes)
    varianza = np.var(exactitudes)
    print("Capacidad de generalizacion del modelo aproximada con 5-folds\n      Promedio exactitud de test:",promedio,"   Varianza:",varianza,"\n")
    
    #C
    indices_folds = model_selection.KFold(n_splits=10,shuffle=True)
    exactitudes = []
    for (train_ind,test_ind) in indices_folds.split(X,y):
        MLP = neural_network.MLPClassifier(hidden_layer_sizes=(neuronas_ocultas),max_iter=500)
        MLP.fit(X[train_ind],y[train_ind])
        exactitudes.append(MLP.score(X[test_ind],y[test_ind]))
    promedio = np.mean(exactitudes)
    varianza = np.var(exactitudes)
    print("Capacidad de generalizacion del modelo aproximada con 10-folds\n      Promedio exactitud de test:",promedio,"   Varianza:",varianza,"\n")
    print("-------------------------------------")