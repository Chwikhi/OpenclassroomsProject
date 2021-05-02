#LogisticRegression and naivebayesclassifier used for 2 labels data (0, 1)
#we use Knn RandomForest SVC and MLPClassifier
from sklearn.datasets import fetch_rcv1
import numpy as np
dataset = fetch_rcv1(shuffle=True, random_state=1)

X = dataset.data[0:1000]
ym = dataset.target[0:1000]
topics = dataset.target_names
y = []

for i in range(1000):
    t = False
    for j in range(103):
        if (t==False) and (ym[i, j] == 1):
            y.append(topics[j])
            t=True
                
print(y[:5])
print(len(y))
y = np.array(y)
y = y.reshape(y.shape[0], 1)

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

mlp_classifier = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
mlp_classifier.fit(Xtrain, ytrain)
ypred_mlp = mlp_classifier.predict(Xtest)

print(accuracy_score(ytest, ypred_mlp))
print("MLP_Classifier accuracy {:.2f}".format(accuracy_score(ytest, ypred_mlp)))




from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rfc = RandomForestClassifier(n_estimators=500, oob_score=True)
model = rfc.fit(Xtrain, ytrain)
ypred = rfc.predict(Xtest)
print("RandomForest accuracy {:.2f}".format(accuracy_score(ytest, ypred)))



from sklearn import neighbors, metrics
from sklearn import model_selection

param_grid = {'n_neighbors':[3, 5, 7, 9, 11, 13, 15]}
score = 'accuracy'
clf = model_selection.GridSearchCV(neighbors.KNeighborsClassifier(), param_grid, cv=5, scoring=score)
clf.fit(Xtrain, ytrain)

print("Knn  hyperparamètre(s) sur le jeu d'entraînement:")
print(clf.best_params_)

ypred_knn = clf.predict(Xtest)
print("Knn Sur le jeu de test : {:.3f}".format(accuracy_score(ytest, ypred_knn)))


from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
score = 'accuracy'
params = {'SVC__C': np.logspace(-3, 3, 7), 'SVC__penalty':['l1','l2'] }

svm = OneVsRestClassifier(SVC())
gs_svm = GridSearchCV(svm, param_grid=params, cv=5, scoring=score)

gs_svm.fit(Xtrain, ytrain)
print(gs_svm.best_params_)

ypred_svm = gs_svm.predict(Xtest)
print("\nSur le jeu de test : {:.3f}".format(metrics.accuracy_score(ytest, ypred_svm)))
