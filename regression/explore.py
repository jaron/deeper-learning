import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# if you get the message "UserWarning: tight_layout : falling back to Agg renderer" when running this
# (especially on OS X) try running this inside Spyder

# a few useful blog posts to get you started on Seaborn:
# * http://blog.insightdatalabs.com/advanced-functionality-in-seaborn/
# * http://twiecki.github.io/blog/2014/11/18/python-for-data-science/



def main():
    # change the line below to load an alternative data set (see the ../data directory)
    data_file = "../data/E14-properties.csv"
    print "Loading: ", data_file
    property_data = pd.read_csv(data_file)

    # remove entries where the type is unknown, to simplify things
    property_data = property_data[property_data.Type != "Unknown"]

    print "Dataset has been loaded with {0} rows".format(len(property_data))

    # first, let's see the relationship between bedrooms and price, with a poly line of best fit
    g = sns.JointGrid(x="Bedrooms", y="Price", data=property_data)  
    g.plot_joint(sns.regplot, order=2)  
    g.plot_marginals(sns.distplot)

    # now let's use a Pairplot to see the interactions between variables
    g = sns.pairplot(property_data[["Price", "Bedrooms", "Remoteness", "Type"]], hue="Type", diag_kind="hist")
    for ax in g.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=45)

    # PairGrid gis like pairplot, but lets us control the individual plot types separately
    g = sns.PairGrid(property_data[["Price", "Bedrooms", "Remoteness", "Type"]], hue="Type")  
    g.map_upper(sns.regplot)    # Upper panels show Regression Lines
    g.map_lower(sns.residplot)  # Lower Panels show Residuals
    g.map_diag(plt.hist)  
    for ax in g.axes.flat:  
        plt.setp(ax.get_xticklabels(), rotation=45)
    g.add_legend()  
    g.set(alpha=0.5)  


main()