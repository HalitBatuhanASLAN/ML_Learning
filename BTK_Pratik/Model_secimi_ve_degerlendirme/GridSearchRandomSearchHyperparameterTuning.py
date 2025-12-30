from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.2,random_state=42)

# KNN
knn = KNeighborsClassifier()
knn_param_grid = {"n_neighbors": np.arange(2,20)}

# gridsearch
knn_grid_search = GridSearchCV(knn, knn_param_grid)
knn_grid_search.fit(X_train,y_train)

print("KNN Grid Search best parameters :",knn_grid_search.best_params_)
print("KNN Grid Search best accurancy :",knn_grid_search.best_score_)


# random search -> parametre daha fazla ise daha etkili olabilir
knn_random_search = RandomizedSearchCV(knn, knn_param_grid)
knn_random_search.fit(X_train,y_train)

print("KNN Random Search best parameters :",knn_random_search.best_params_)
print("KNN Random Search best accurancy :",knn_random_search.best_score_)

print("")

# Tree
tree = DecisionTreeClassifier()
tree_param_grid = {"max_depth":[3,5,7],
                   "max_leaf_nodes":[None,5,10,20,30,50]}

# gridsearch
tree_grid_search = GridSearchCV(tree, tree_param_grid)
tree_grid_search.fit(X_train,y_train)

print("Tree Grid Search best parameters :",tree_grid_search.best_params_)
print("Tree Grid Search best accurancy :",tree_grid_search.best_score_)


# random search -> parametre daha fazla ise daha etkili olabilir
tree_random_search = RandomizedSearchCV(tree, tree_param_grid)
tree_random_search.fit(X_train,y_train)

print("Tree Random Search best parameters :",tree_random_search.best_params_)
print("Tree Random Search best accurancy :",tree_random_search.best_score_)

print("")
# SVM
svm = SVC()

svm_param_grid = {"C":[0.1,1,10,100],
                  "gamma":[0.1,0.01,0.001,0.0001]}

# gridsearch
svm_grid_search = GridSearchCV(svm, svm_param_grid)
svm_grid_search.fit(X_train,y_train)

print("svm Grid Search best parameters :",svm_grid_search.best_params_)
print("svm Grid Search best accurancy :",svm_grid_search.best_score_)


# random search -> parametre daha fazla ise daha etkili olabilir
svm_random_search = RandomizedSearchCV(svm, svm_param_grid)
svm_random_search.fit(X_train,y_train)

print("svm Random Search best parameters :",svm_random_search.best_params_)
print("svm Random Search best accurancy :",svm_random_search.best_score_)

print("")




