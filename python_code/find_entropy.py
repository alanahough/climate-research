"""
Prints from every split in every tree of the specified forest
"""


import joblib


if __name__ == '__main__':
    forest = joblib.load('./forests/oversample_60_20_20_split_rcp85_2100.joblib')
    for i in range(0, len(forest.estimators_)):
        estimator = forest.estimators_[i]
        tree_feature = estimator.tree_.feature
        print("tree", i,)
        for j in range(estimator.tree_.node_count):
            print("\tnode", j, "entropy =", estimator.tree_.impurity[j])
