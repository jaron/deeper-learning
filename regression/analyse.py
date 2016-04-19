# coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl


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
    print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))

    print "\nFeatures..."
    print X_all  # use .head() to print the first 5 rows
    print "\nTop 5 Prices..."
    print y_all.head()

    info(X_all,  y_all)



main()