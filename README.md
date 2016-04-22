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
* the 'Remoteness' feature is the straight-line distance between the property and the nearest major rail/tube station
* ```python explore.py``` creates visualisations of the raw data, if you get a rendering error, run this inside Spyder


__Classification__

* the classifiers attempt to predict the Type of a property from the remaining features, (Price, Rooms, Remoteness)
* as London listings are dominated by flats, this uses a dataset of more rural properties to make it more interesting
* to run these, ```cd classification``` then ```python classify.py```


More to come...
