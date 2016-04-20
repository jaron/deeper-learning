from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn import grid_search
from sklearn.metrics import mean_squared_error, r2_score, make_scorer



def fit_model_decision_tree(X, y):
    """ Tunes a decision tree regressor using GridSearchCV on the input data X and target labels y and returns the optimal model. """

    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # Set up the parameters we wish to tune
    parameters = { "max_depth": [None, 3, 4, 5, 6, 7],
                   "max_leaf_nodes": [None, 5, 10, 20, 30]
                 }

    # Make an appropriate scoring function
    scoring_function = make_scorer(mean_squared_error, greater_is_better=False)

    # Make the GridSearchCV object
    reg = grid_search.GridSearchCV(regressor, parameters, scoring_function)

    # Fit the learner to the data to obtain the optimal model with tuned parameters
    reg.fit(X, y)

    # Return the optimal model
    return reg.best_estimator_



def fit_model_svm(X, y):
    """ Tunes a SVM regressor using GridSearchCV on input data X and target labels y, returning the optimal model. """

    # Create SVM regressor object
    regressor = svm.SVR()

    # Set up the parameters we wish to tune
    parameters = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'degree' : [2], 'kernel': ['poly']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001], 'kernel': ['rbf']},
    ]

    # Make an appropriate scoring function
    scoring_function = make_scorer(mean_squared_error, greater_is_better=False)

    # Make the GridSearchCV object
    reg = grid_search.GridSearchCV(regressor, parameters, scoring_function)

    # Fit the learner to the data to obtain the optimal model with tuned parameters
    reg.fit(X, y)

    # Return the optimal model
    return reg.best_estimator_


def fit_model_knn(X, y):
    """ Tunes a KNN regressor using GridSearchCV on input data X and target labels y, returning the optimal model. """

    # Create KNN regressor object
    regressor = KNeighborsRegressor()

    # Set up the parameters we wish to tune
    parameters = { "n_neighbors" : [4, 8, 12, 16]}

    # Make an appropriate scoring function
    scoring_function = make_scorer(mean_squared_error, greater_is_better=False)

    # Make the GridSearchCV object
    reg = grid_search.GridSearchCV(regressor, parameters, scoring_function)

    # Fit the learner to the data to obtain the optimal model with tuned parameters
    reg.fit(X, y)

    # Return the optimal model
    return reg.best_estimator_