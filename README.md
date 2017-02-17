# deeper-learning
_Little experiments in artificial intelligence_

####Getting Started

* this project uses [scikit-learn](http://scikit-learn.org/stable/install.html), you can either install scikit or the more comprehensive [Anaconda distribution](https://docs.continuum.io/anaconda/install)
* then just clone the project to your local file system 
* to see the Seaborn visualisations, enter ```conda install seaborn```


__Regression__

* in a terminal, navigate to where you've stored this project and ```cd regression```
* ```python analyse.py``` will run 3 regressors on a dataset of around 1700 London properties, each will use regression to predict the property price from the supplied features.
* there is one categorical feature ('Type') which is converted into several binary [dummy variables](https://en.wikipedia.org/wiki/Dummy_variable_%28statistics%29)
* the 'Remoteness' feature is straight-line distance between property and the nearest major rail/tube station
* ```python explore.py``` creates visualisations of raw data, if there's a rendering error, run inside Spyder


__Classification__

* attempts to predict the Type of a property from the remaining features, (Price, Rooms, Remoteness)
* as London listings are dominated by flats, uses a dataset of rural properties to make it interesting
* to run these, ```cd classification``` then ```python classify.py```


__Unsupervised__

* this example uses unsupervised learning techniques, like Principal Component Analysis and K-means clustering to analyse the property data without relying on hand-coded annotations
* there are two examples in the ```unsupervised``` directory
* run ```python analyse_locations.py``` to analyse a set of properties by their size and location, as the location is the dominant factor in determining property price, you'll notice the principal components are based around the Remoteness attribute, and this influences the clusters created
* run ```python analyse_property_type.py``` to omit the geographical features and consider property features alone, this results in components and clusters associated with the number of rooms in the properties


__Numpy-NN__

* this implements a simple neural network in just numpy to demonstrate how to solve a regression problem, in this case predicting bike hire numbers. 
* also included is simple sentiment analysis using a numpy-based network, demonstrating how to distinguish between signal and noise, and visualise what's going on under the hood 
* these are Jupyter notebooks, so they can run be and experimented with in a browser


__TFLearn__

* examples using TFLearn to simplify building models for sentiment analysis and MNIST classification

More to come...
