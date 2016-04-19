# coding=utf-8

import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn import grid_search
from sklearn.tree import DecisionTreeRegressor
import visualise


def info(details, prices):
    # Number of houses in the dataset
    total_houses = len(details)

    # Number of features in the dataset
    total_features = details.shape[1]

    # Minimum housing value in the dataset
    minimum_price = np.min(prices)

    # Maximum housing value in the dataset
    maximum_price = np.max(prices)

    # Mean house value of the dataset
    mean_price = np.mean(prices)

    # Median house value of the dataset
    median_price = np.median(prices)

    # Standard deviation of housing values of the dataset
    std_dev = np.std(prices)

    # Show the calculated statistics
    print "\n*** Dataset stats ***"
    print "Total num of properties :\t", total_houses
    print "Total num of features   :\t", total_features
    print "Minimum house price     :\t£", minimum_price
    print "Maximum house price     :\t£", maximum_price
    print "Mean house price        :\t£ {0:.2f}".format(mean_price)
    print "Median house price      :\t£ {0:.2f}".format(median_price)
    print "Std Dev of house price  :\t£ {0:.2f}".format(std_dev)


# Shows some sample data so we can inspect it
def show_data(X_all, y_all):
    print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))
    print "\nFeatures..."
    print X_all.head()
    print "\nTop 5 Prices..."
    print y_all.head()

# Preprocess feature columns to ensure all are continuous values
def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column for non-numeric columns
    for col, col_data in X.iteritems():
        # If data type is non-numeric, replace any yes/no values with an int equivalents (1 or 0)
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0]) # will change the data type of column to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'type' => 'type_Detached', 'type_Flat'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX


def fit_model(X, y):
    """ Tunes a decision tree regressor model using GridSearchCV on the input data X
        and target labels y and returns this optimal model. """

    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # Set up the parameters we wish to tune
    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}

    # Make an appropriate scoring function
    scoring_function = make_scorer(mean_squared_error, greater_is_better=False)

    # Make the GridSearchCV object
    reg = grid_search.GridSearchCV(regressor, parameters, scoring_function)

    # Fit the learner to the data to obtain the optimal model with tuned parameters
    reg.fit(X, y)

    # Return the optimal model
    return reg.best_estimator_


def main():
    filename = "../data/IP5-properties.csv"
    property_data = pd.read_csv(filename)
    print "Dataset has been loaded with {0} rows".format(len(property_data))

    feature_cols = list(property_data.columns[1:])  # all columns but first are features
    target_col = property_data.columns[0]  # first column is the target/label
    print "Feature column(s):-\n{}".format(feature_cols)
    print "Target column: {}".format(target_col)

    X_all = property_data[feature_cols]  # feature values for all students
    y_all = property_data[target_col]  # corresponding targets/labels

    X_all = preprocess_features(X_all)

    show_data(X_all, y_all)
    info(X_all,  y_all)

    # Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_all, y_all, test_size=0.3)

    print "Training set: {} samples".format(X_train.shape[0])
    print "Test set: {} samples".format(X_test.shape[0])

    # Fine-tune with full training set
    best = fit_model(X_train, y_train)
    print best.get_params()
    predictions = best.predict(X_test)

    # Find the performance on the training set
    training_error = mean_squared_error(y_train, best.predict(X_train))

    # Find the performance on the testing set
    test_error = mean_squared_error(y_test, best.predict(X_test))

    print "Training Error = ", training_error, "\n"
    print "Test Set Error = ", test_error, "\n"

    #print "Test Data:\n", y_test
    #print "Predictions shape: ", predictions.shape

    for i in range(len(predictions)):
        print y_test.iloc[i], " ->", predictions[i]

    #print "Actually: ", y_test.iloc[0]
    #print "Predicted: ", predictions[0]



    # print "Predicted value of client's home: {0:.2f} -> real value {0:.2f}".format(sale_price, y_test[0])

    try:
        visualise.plot_curves(X_train, y_train, X_test, y_test)
        visualise.model_complexity(X_train, y_train, X_test, y_test)
    except Exception,e:
        print str(e)
        print "Something went wrong when displaying graphs"

main()