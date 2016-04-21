from sklearn import grid_search
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn import cross_validation
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier





def fit_model_knn(X, y):
    # Fit the best combination of parameters
    classifier = KNeighborsClassifier()

    # Set up the parameters we wish to tune
    parameters = {'n_neighbors': [5, 10, 15, 20, 25]}

    f1_scorer = make_scorer(f1_score, average="micro")
    sss  = cross_validation.StratifiedShuffleSplit(y, n_iter=10, test_size=0.3)
    gs = grid_search.GridSearchCV(classifier, parameters, cv=sss, scoring=f1_scorer)
    gs.fit(X, y)
    return gs.best_estimator_


def fit_model_svm(X, y):
    # Fit the best combination of parameters
    classifier = svm.SVC()

    # Set up the parameters we wish to tune
    parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]}]

    f1_scorer = make_scorer(f1_score, average="micro")
    sss  = cross_validation.StratifiedShuffleSplit(y, n_iter=10, test_size=0.3)
    gs = grid_search.GridSearchCV(classifier, parameters, cv=sss, scoring=f1_scorer)
    gs.fit(X, y)
    return gs.best_estimator_


def fit_model_decision_tree(X, y):
    # Fit the best combination of parameters
    classifier = DecisionTreeClassifier()

    # Set up the parameters we wish to tune
    dtc_parameters = {"criterion": ["entropy", "gini"],
                      "max_depth": [3, 4, 5, 6, 7],
                      "max_leaf_nodes": [5, 10, 15, 20, 25, 30]
                      }

    f1_scorer = make_scorer(f1_score, average="micro")
    sss  = cross_validation.StratifiedShuffleSplit(y, n_iter=10, test_size=0.3)
    gs = grid_search.GridSearchCV(classifier, dtc_parameters, cv=sss, scoring=f1_scorer)
    gs.fit(X, y)
    return gs.best_estimator_


def unopt_decision_tree(X, y):
    # create an optimised DecisionTreeClassifier
    classifier = DecisionTreeClassifier(max_depth=5)
    classifier.fit(X, y)

    # code to produce visualisations
    # print_decision_tree(classifier, X.columns)
    # visualize_tree(classifier, X.columns)

    return classifier