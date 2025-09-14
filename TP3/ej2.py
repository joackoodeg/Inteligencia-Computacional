import numpy as np
from sklearn import model_selection 
from sklearn import datasets
# Clasificadores
from sklearn.naive_bayes import GaussianNB                   # Naive Bayes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # LDA
from sklearn.neighbors import KNeighborsClassifier           # KNN
from sklearn.tree import DecisionTreeClassifier              # Árbol de decisión
from sklearn.svm import SVC                                  # SVM
from sklearn.neural_network import MLPClassifier             # Perceptrón multicapa


if __name__ == "__main__":
    X,y = datasets.load_digits(return_X_y=True)

    kf = model_selection.KFold(n_splits=5,shuffle=True)

    exactitudes = np.zeros((5,6))
    for i,(train_i,test_i) in enumerate(kf.split(X)):
        
        mlp = MLPClassifier(hidden_layer_sizes=20,max_iter=400)
        mlp.fit(X[train_i],y[train_i])
        exactitudes[i,0] = mlp.score(X[test_i],y[test_i])

        nb = GaussianNB()
        nb.fit(X[train_i],y[train_i])
        exactitudes[i,1] = nb.score(X[test_i],y[test_i])

        lda = LinearDiscriminantAnalysis()
        lda.fit(X[train_i],y[train_i])
        exactitudes[i,2] = lda.score(X[test_i],y[test_i])

        knn = KNeighborsClassifier()
        knn.fit(X[train_i],y[train_i])
        exactitudes[i,3] = knn.score(X[test_i],y[test_i])

        svm = SVC()
        svm.fit(X[train_i],y[train_i])
        exactitudes[i,4] = svm.score(X[test_i],y[test_i])

        tree = DecisionTreeClassifier()
        tree.fit(X[train_i],y[train_i])
        exactitudes[i,5] = tree.score(X[test_i],y[test_i])


    print("MLP: Promedio: ", np.mean(exactitudes[:,0])," Varianza: ", np.var(exactitudes[:,0]))
    print("NBY: Promedio: ", np.mean(exactitudes[:,1])," Varianza: ", np.var(exactitudes[:,1]))
    print("LDA: Promedio: ", np.mean(exactitudes[:,2])," Varianza: ", np.var(exactitudes[:,2]))
    print("KNN: Promedio: ", np.mean(exactitudes[:,3])," Varianza: ", np.var(exactitudes[:,3]))
    print("SVM: Promedio: ", np.mean(exactitudes[:,4])," Varianza: ", np.var(exactitudes[:,4]))
    print("TRE: Promedio: ", np.mean(exactitudes[:,5])," Varianza: ", np.var(exactitudes[:,5]))