import numpy as np
import pandas as pd
from collections import Counter
import classifiers

from sklearn import cross_validation
from sklearn.metrics import f1_score


def show_info(data):
    n_features = len(data.columns)
    print "Total number of properties: {}".format(len(data))
    counts = Counter(data['Type'])
    print "Types of properties: ", counts
    print "Number of features: {}".format(n_features)


def extract_features(data):
    # Extract feature (X) and target (y) columns

    # use the Type column as the label
    target_col = data.columns[4]  # this is the target/label
    y_all = data[target_col]      # corresponding labels

    # now remove the type column, everything that remains will be features
    data = data.drop('Type', 1)
    feature_cols = list(data.columns)
    X_all = data[feature_cols]

    print "Feature column(s): \t{}".format(feature_cols)
    print "Target column: {}".format(target_col)
    print "\nFeature values:-", X_all.head()  # print the first 5 rows
    return X_all, y_all


# Generate predictions using the supplied regressor
def show_predictions(clf, X_train, X_test, y_train, y_test):

    print "\n-----", type(clf).__name__, " Results -----"
    print "Best parameters:", clf.get_params()
    predictions = clf.predict(X_test)
    test_predictions = clf.predict(X_train)

    # Calculate performance on the training set
    training_score = f1_score(y_train, test_predictions, average="micro")

    # Calculate performance on the testing set
    test_score = f1_score(y_test, predictions, average="micro")

    print type(clf).__name__, "Training f1 score = ", training_score, "\n"

    # display some sample predictions and the corresponding real values
    for i in range(min(10, len(predictions))):
        is_correct = y_test.iloc[i] == predictions[i]
        print "Is: {0:10}\t-[predicted]-> {1:10} \t({2})".format(y_test.iloc[i], predictions[i], is_correct)

    print "\n>>", type(clf).__name__, "Test Set f1 score = ", test_score

    print "\n- - - - - - - - - - - - -\n"


def main():
    # Read student data
    property_data = pd.read_csv("../data/IP5_IP12-properties.csv")
    print "Dataset has been loaded with {0} rows".format(len(property_data))

    # remove entries where the type is unknown, as they'll be difficult to classifiy
    property_data = property_data[property_data.Type != "Unknown"]

    show_info(property_data)

    X_all, y_all = extract_features(property_data)
    print "Processed feature columns ({}): \t{}".format(len(X_all.columns), list(X_all.columns))

    # now shuffle and split data into training and test sets
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_all, y_all, test_size=0.3, random_state=42, stratify = y_all)

    print "Training set: {} samples".format(X_train.shape[0])
    print "Test set: {} samples".format(X_test.shape[0])

    # try Decision Tree classifier
    best = classifiers.fit_model_decision_tree(X_train, y_train)
    show_predictions(best, X_train, X_test, y_train, y_test)

    # try SVM classifier
    best = classifiers.fit_model_svm(X_train, y_train)
    show_predictions(best, X_train, X_test, y_train, y_test)

    # try KNN classifier
    best = classifiers.fit_model_knn(X_train, y_train)
    show_predictions(best, X_train, X_test, y_train, y_test)

main()