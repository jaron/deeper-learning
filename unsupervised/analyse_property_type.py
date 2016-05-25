# coding=utf-8

"""
This is an example of using unsupervised learning to analyse and explore a data set by looking at
property features (number of bedrooms, bathrooms and garages)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import cross_validation
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import renders as rs
import seaborn as sns
from IPython.display import display # Allows the use of display() for DataFrames
import matplotlib.pyplot as plt


def main():

	# 1 load the data set
	property_data = load_data()

	# and create a samples subset DataFrame
	# 6 is a 5-bed detached property
	# 255 is a 3-bed terraced property
	# 309 is a 2-bed flat
	indices = [6, 155, 309]
	samples = pd.DataFrame(property_data.loc[indices], columns = property_data.keys()).reset_index(drop = True)

	# 2 inspect data
	describe(property_data, samples, indices)

	# 3 determine the importance of a feature
	feature_importance(property_data)

	# 4 feature scaling
	log_data, log_samples = scale_features(property_data, samples)

	# 5 remove outliers
	good_data, outliers = remove_outliers(log_data)

	# 6 do principal component analysis
	pca, reduced_data, pca_samples = do_pca(good_data, log_samples)

	# 7 do unsupervised
	centers = do_clustering(reduced_data, pca_samples)

	# 8 make predictions
	do_data_recovery(pca, centers, property_data)

	# recreate the scatter plot, this time colour by the known types
	reveal_type(reduced_data, outliers, pca_samples)



def load_data():
	# change the line below to load an alternative data set (see the ../data directory)
	data_file = "../data/IP5_IP12-properties.csv"
	print "Loading: ", data_file
	property_data = pd.read_csv(data_file)

	# drop the type field and the location fields
	property_data.drop(['Type', 'Latitude', 'Longitude', 'Remoteness'], axis = 1, inplace = True)
	print "Loaded dataset: {} properties with {} features each".format(*property_data.shape)

	return property_data


'''
First, let's explore the data set by looking at some statistics for the whole distribution, and then a heatmap
that shows how the features of a few samples relate to the others in the dataset.
'''
def describe(property_data, samples, indices):

	print "\nDataset Statistics:"
	display(property_data.describe())

	print "\nChosen samples of dataset:"
	display(samples)

	# look at percentile ranks
	pcts = 100. * property_data.rank(axis=0, pct=True).iloc[indices].round(decimals=4)
	print "\nPercentile ranks:\n", pcts

	# visualize percentiles with heatmap
	# sns.heatmap(pcts.reset_index(drop=True), annot=True, vmin=1, vmax=99, cmap='YlGnBu')
	# plt.show()


'''
When exploring data it's important consider which of the features is actually relevant when it comes to making
predictions. We can make this determination by training a supervised regression learner on a subset of the data
with one feature removed, and then score how well that model can predict the removed feature.
'''
def feature_importance(property_data):

	exclude_feature = 'Bedrooms'
	# Make a copy of the DataFrame, using the 'drop' function to drop the given feature
	new_data = property_data.drop([exclude_feature], axis = 1, inplace = False)

	# Split the data into training and testing sets using the given feature as the target
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(new_data, property_data[exclude_feature], test_size=0.3, random_state=99)

	# Create a decision tree regressor and fit it to the training set
	regressor = DecisionTreeRegressor().fit(X_train, y_train)

	# Report the score of the prediction using the testing set
	predictions = regressor.predict(X_test)
	score = r2_score(y_test, predictions)

	print "\nR^2 coefficient of determination after excluding '", exclude_feature, "' = ", score
	print "An R^2 of 1 indicates a perfect fit and that the model can make predictions without the missing feature."
	print "A negative R^2 indicates the model fails to fit the data."


'''
If the data is not normally distributed (skewed), where the mean and median vary significantly it can be a good idea
to apply some non-linear scaling — such as applying the natural logarithm.
'''
def scale_features(property_data, samples):

	# Scale the data using the natural logarithm
	log_data = property_data
	log_data['Price'] = np.log(property_data['Price'])

	# Scale the sample data using the natural logarithm
	log_samples = samples
	log_samples['Price'] = np.log(samples['Price'])
	print "\nSamples after scaling:"
	display(log_samples)

	# Produce a scatter matrix for each pair of newly-transformed features
	pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
	plt.show()
	return log_data, log_samples


'''
Detecting outliers in the data is extremely important in the data preprocessing step of any analysis. The presence of
outliers can often skew results which take into consideration these data points. There are many "rules of thumb" for
what constitutes an outlier in a dataset. Here, we will use Tukey's Method for identfying outliers
(http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey- method/):
An outlier step is calculated as 1.5 times the interquartile range (IQR). A data point with a feature that is beyond
an outlier step outside of the IQR for that feature is considered abnormal, and should be removed.
'''
def remove_outliers(log_data):

	counter = Counter()
	print "\nOutliers..."
	# For each feature find the data points with extreme high or low values
	for feature in log_data.keys():

		if feature == 'Garages' or feature == 'Bathrooms' or feature == 'Bedrooms':
			continue
		# Calculate Q1 (25th percentile of the data) for the given feature
		Q1 = np.percentile(log_data[feature], 25)

		# Calculate Q3 (75th percentile of the data) for the given feature
		Q3 = np.percentile(log_data[feature], 75)

		if Q1 == Q3:
			continue
		print feature, ":", Q1, " <->", Q3

		# Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
		step = (Q3 - Q1) * 1.5

		# Display the outliers
		print "Data points considered outliers for the feature '{}':".format(feature)
		outlier_rows = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
		display(outlier_rows.index.values)

		for o in outlier_rows.index:
			counter[o] += 1

	print "\n", counter
	outliers = counter.keys()

	# Remove the outliers, if any were identified
	good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)

	return good_data, outliers


'''
Once the data has been scaled to a more normal distribution and has had any necessary outliers removed, we can apply
principal component analysis (PCA) to draw conclusions about the underlying structure of the property data.
PCA calculates the dimensions which best maximize variance, so will find which compound combinations of features that
best describe properties.
'''
def do_pca(good_data, log_samples):

	# Apply PCA to the good data with the same number of dimensions as features
	pca = PCA(n_components=4)
	pca.fit(good_data)

	# Apply a PCA transformation to the sample log-data
	pca_samples = pca.transform(log_samples)

	# Generate PCA results plot
	pca_results = rs.pca_results(good_data, pca)

	# Display sample log-data after having a PCA transformation applied
	print "Principal Components..."
	display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))

	print "Explained variance ratios for all principal components: ", pca.explained_variance_ratio_

	# Fit PCA to the good data using only two dimensions
	pca = PCA(n_components=2)
	pca.fit(good_data)

	# Apply a PCA transformation the good data
	reduced_data = pca.transform(good_data)

	# Apply a PCA transformation to the sample log-data
	pca_samples = pca.transform(log_samples)

	# Create a DataFrame for the reduced data
	reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

	# Display sample log-data after applying PCA transformation in two dimensions
	display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))

	print "Explained variance ratios for top 2 principal components: ", pca.explained_variance_ratio_

	return pca, reduced_data, pca_samples


'''
Once we've determined the optimal number of clusters, we can visualise them.
'''
def do_clustering(reduced_data, pca_samples):
	range_n_clusters = [2, 3, 4, 5]
	best_score = 0
	best_cluster_size = 0

	for num_clusters in range_n_clusters:
		# Apply your unsupervised algorithm of choice to the reduced data
		clusterer = KMeans(n_clusters=num_clusters)
		clusterer.fit(reduced_data)

		# Predict the cluster for each data point
		preds = clusterer.predict(reduced_data)

		# Calculate the mean silhouette coefficient for the number of clusters chosen
		score = silhouette_score(reduced_data, preds)
		if score > best_score:
			best_score= score
			best_cluster_size = num_clusters
		print "Silhouette score for" , num_clusters, "clusters =", score

	print "Best cluster size = ", best_cluster_size

	# re-run the unsupervised with a specific number of clusters
	clusterer = KMeans(n_clusters=best_cluster_size)
	clusterer.fit(reduced_data)
	preds = clusterer.predict(reduced_data)
	centers = clusterer.cluster_centers_
	sample_preds = clusterer.predict(pca_samples)

	# Display the results of the unsupervised from implementation
	rs.cluster_results(reduced_data, preds, centers, pca_samples)

	# Display the predictions
	for i, pred in enumerate(sample_preds):
		print "Sample point", i, "predicted to be in Cluster", pred

	return centers


'''
Each cluster present in the visualization above has a central point. These centres (or means) are the averages of all
the data points predicted in the respective clusters (in this case, the average property of that segment). Since the
price is currently scaled by a logarithm, we can recover it by applying an inverse transformation.
'''
def do_data_recovery(pca, centers, property_data):
	# Data Recovery
	# Inverse transform the centers
	log_centers = pca.inverse_transform(centers)

	# Display the true centers
	segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
	true_centers = pd.DataFrame(np.round(log_centers), columns = property_data.keys())
	true_centers.index = segments
	true_centers['Price'] = np.round(np.exp(true_centers['Price']), 0)

	print "\nValues of each feature at centroids:"
	display(true_centers)


'''
Visualizes the PCA-reduced cluster data in two dimensions using the full dataset
labeled by "Type" and cues added for the chosen samples
'''
def reveal_type(reduced_data, outliers, pca_samples):

	try:
		full_data = pd.read_csv("../data/IP5_IP12-properties.csv")
	except:
		print "Dataset could not be loaded. Is the file missing?"
		return False

	# Create the Channel DataFrame
	channel = pd.DataFrame(full_data['Type'], columns = ['Type'])
	channel = channel.drop(channel.index[outliers]).reset_index(drop = True)
	labeled = pd.concat([reduced_data, channel], axis = 1)

	# Generate the cluster plot
	fig, ax = plt.subplots(figsize = (14,8))

	# assign colour based on distance
	grouped = labeled.groupby('Type')
	colours = { 'Bungalow' : 'yellow', 'Detached' : 'green', 'Semi' : 'pink', 'Terraced' : 'red', 'Flat' : 'cyan'}

	for i, channel in grouped:
		if (i == 'Unknown'): continue
		dotcolor = colours[i]
		channel.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', color = dotcolor, label = i, s=30)

	# Plot transformed sample points
	for i, sample in enumerate(pca_samples):
		ax.scatter(x = sample[0], y = sample[1], \
			   s = 200, linewidth = 3, color = 'black', marker = 'o', facecolors = 'none');
		ax.scatter(x = sample[0]+0.25, y = sample[1]+0.3, marker='$%d$'%(i), alpha = 1, s=125);

	# Set plot title
	ax.set_title("True property types revealed\nDimension 1 is associated with cheaper properties, Dimension 2 is associated with larger properties.");
	plt.show()


main()

