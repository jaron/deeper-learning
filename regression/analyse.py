# coding=utf-8

import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.metrics import r2_score
import statsmodels.formula.api as smf
import visualise
import regressors


def info(details, prices):
    # Number of properties in the dataset
    total_houses = len(details)

    # Number of features in the dataset
    total_features = details.shape[1]

    # Minimum property value in the dataset
    minimum_price = np.min(prices)

    # Maximum property value in the dataset
    maximum_price = np.max(prices)

    # Mean property value of the dataset
    mean_price = np.mean(prices)

    # Median property value of the dataset
    median_price = np.median(prices)

    # Standard deviation of property prices
    std_dev = np.std(prices)

    # Show the calculated statistics
    print "\n*** Dataset stats ***"
    print "Total num of properties    :\t", total_houses
    print "Total num of features      :\t", total_features
    print "Minimum property price     :\t£", minimum_price
    print "Maximum property price     :\t£", maximum_price
    print "Mean property price        :\t£ {0:.2f}".format(mean_price)
    print "Median property price      :\t£ {0:.2f}".format(median_price)
    print "Std Dev of property price  :\t£ {0:.2f}".format(std_dev)


# Shows some sample data so we can inspect it
def show_data(X_all, y_all):
    print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))
    print "\nFeatures..."
    print X_all.head()
    print "\nTop 5 Prices..."
    print y_all.head()

    model = smf.OLS(y_all, X_all)
    res = model.fit()
    # small p-values indicate values of significance
    print res.summary()


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


# Generate predictions using the supplied regressor
def show_predictions(best, X_train, X_test, y_train, y_test):

    print "\n-----", type(best).__name__, " Results -----"
    print "Best parameters:", best.get_params()
    predictions = best.predict(X_test)
    test_predictions = best.predict(X_train)

    # Calculate performance on the training set
    training_r2 = r2_score(y_train, test_predictions)

    # Calculate performance on the testing set
    test_r2 = r2_score(y_test, predictions)

    print type(best).__name__, "Training R2 score = ", training_r2, "\n"

    # display some sample predictions and the corresponding real values
    for i in range(min(10, len(predictions))):
        diff =  float(predictions[i] - y_test.iloc[i]) / float(y_test.iloc[i]) * 100
        print "Is: ", y_test.iloc[i], "\t-[predicted]-> {0:.0f} \t({1:.1f} %)".format(predictions[i], diff)

    print "\n>>", type(best).__name__, "Test Set R2 score = ", test_r2

    print "\n- - - - - - - - - - - - -\n"


def main():
    # change the line below to load an alternative data set (see the ../data directory)
    data_file = "../data/E14-properties.csv"
    print "Loading: ", data_file
    property_data = pd.read_csv(data_file)

    # if you want to exclude the categorical property type field, uncomment this
    # property_data = property_data.drop('Type', 1)

    print "Dataset has been loaded with {0} rows".format(len(property_data))

    feature_cols = list(property_data.columns[1:])  # all columns but first are features
    target_col = property_data.columns[0]           # first column is property price
    print "Feature column(s):-\n{}".format(feature_cols)
    print "Target column: {}".format(target_col)

    X_all = property_data[feature_cols]  # feature values
    y_all = property_data[target_col]    # corresponding prices

    # now update any categorical fields so all are continuous
    X_all = preprocess_features(X_all)

    show_data(X_all, y_all)
    info(X_all,  y_all)

    # Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_all, y_all, test_size=0.3)

    print "\nTraining set: {} samples".format(X_train.shape[0])
    print "Test set: {} samples".format(X_test.shape[0])

    # create and evaluate several different regressors
    best_dt = regressors.fit_model_decision_tree(X_train, y_train)
    show_predictions(best_dt, X_train, X_test, y_train, y_test)

    best = regressors.fit_model_svm(X_train, y_train)
    show_predictions(best, X_train, X_test, y_train, y_test)

    best = regressors.fit_model_knn(X_train, y_train)
    show_predictions(best, X_train, X_test, y_train, y_test)

    # try Adaboost with the best decision tree
    boosted = regressors.fit_model_adaboost(best_dt, X_train, y_train)
    show_predictions(boosted, X_train, X_test, y_train, y_test)


    # visualisations don't seem to run from the console, but do work in Spyder
    #try:
        # visualise.plot_learning_curve(best, "Learning Curves", X_train, y_train, n_jobs=4).show()
        # visualise.plot_curves(X_train, y_train, X_test, y_test)
        # visualise.model_complexity(X_train, y_train, X_test, y_test)
    #except Exception,e:
    #    print str(e)
    #    print "Something went wrong when displaying graphs"

main()