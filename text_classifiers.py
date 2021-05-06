
### chargement des données
from sklearn.datasets import fetch_rcv1
dataset = fetch_rcv1(shuffle=True, random_state=1)


### Prétraitement des données
import numpy as np
from sklearn.model_selection import train_test_split

X = dataset.data[0:10000]
ym = dataset.target[0:10000]
topics = dataset.target_names
y = []

for i in range(10000):
    t = False
    for j in range(103):
        if (t==False) and (ym[i, j] == 1):
            y.append(topics[j])
            t=True
                
print(y[:5])
print(len(y))
y = np.array(y)
y = y.reshape(y.shape[0], 1)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)


### Préparation des modèles
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import LatentDirichletAllocation, PCA


preprocessor = make_pipeline(FunctionTransformer((lambda e : e*150)),
                             LatentDirichletAllocation(max_iter=5, 
                                                       learning_method='online',
                                                       random_state=0))


Hyper_params_SVC = {'svc__gamma':[1e-3,1e-4], 'svc__C':[1, 10, 100, 1000, 10000]}
Hyper_params_MLP = {'mlpclassifier__beta_1':np.linspace(1e-3, 0.99, 3), 
                    'mlpclassifier__beta_2':np.linspace(0.5, 0.99, 3),
                    'mlpclassifier__alpha':[1e-2, 1e-3]}
Hyper_params_knn = {'kneighborsclassifier__n_neighbors':[1, 3, 5, 7, 9]}
Hyper_params_RF = {'randomforestclassifier__n_estimators':[200, 300, 400]}
Hyper_params_Tree = {'decisiontreeclassifier__criterion':['gini', 'entropy']}


MLPClassifier = make_pipeline(preprocessor, MLPClassifier(random_state=0, max_iter=400,
                                                          solver='adam', shuffle=True))
RandomForestClassifier = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))
KNN = make_pipeline(preprocessor, KNeighborsClassifier())
Tree = make_pipeline(preprocessor, DecisionTreeClassifier(random_state=0))
SVM = make_pipeline(preprocessor, OneVsRestClassifier(SVC()))

# Modèle optimisés
TreeCV = GridSearchCV(Tree, param_grid=Hyper_params_Tree, cv=5, scoring='accuracy')
MLPClassifierCV = RandomizedSearchCV(MLPClassifier, param_distributions=Hyper_params_MLP,
                                     cv=5, scoring='accuracy', n_iter=12)
RandomForestClassifierCV = GridSearchCV(RandomForestClassifier, param_grid=Hyper_params_RF,
                                        cv=5, scoring='accuracy')
KNNCV = GridSearchCV(KNN, param_grid=Hyper_params_knn, cv= 5, scoring='accuracy')
SVMCV = RandomizedSearchCV(SVM, param_distributions=Hyper_params_SVC,
                           cv= 5, scoring='accuracy', n_iter=4)


## Evaluation des différents modéles

def evaluation(model):
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    print(model)
    print('accuracy = ',accuracy_score(ytest, ypred))
    print(classification_report(ytest, ypred))
    
def evaluationCV(model):
    model.fit(Xtrain, ytrain)
    ypred = model.best_estimator_.predict(Xtest)
    print(model)
    print(model.best_params_)
    print('accuracy = ',accuracy_score(ytest, ypred))
    print(classification_report(ytest, ypred))
    
Model_List = [Tree, MLPClassifier, RandomForestClassifier, KNN, SVM]
Model_List_CV = [TreeCV, MLPClassifierCV, RandomForestClassifierCV, KNNCV, SVMCV]

for model in Model_List:
    evaluation(model)
for model in Model_List_CV:
    evaluationCV(model)
    
#########

