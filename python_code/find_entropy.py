from sklearn import tree, metrics, ensemble
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import pylab
import joblib
import os
from collections import Counter
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from pprint import pprint


if __name__ == '__main__':
    forest = joblib.load('./forests/oversample_60_20_20_split_rcp85_2100.joblib')
    for i in range(0, len(forest.estimators_)):
        estimator = forest.estimators_[i]
        tree_feature = estimator.tree_.feature
        print("tree", i,)
        for j in range(estimator.tree_.node_count):
            print("\tnode", j, "entropy =", estimator.tree_.impurity[j])
