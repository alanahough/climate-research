from sklearn import tree, metrics, ensemble
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def encode(df):
    # change categorical features to 0 and 1 values (bc sklean decision tree only takes numeical values)
    ones = {"sun": "sunny", "wind": "windy", "humidity": "humid", "season": "summer", "time": "AM"}
    row_list = []
    for i in range(len(df)):
        row = []
        for col in ones:
            if df.loc[i, col] == ones[col]:
                row.append(1)
            else:
                row.append(0)
        row.append(df.loc[i, "temp"])
        row.append(df.loc[i, "travel"])
        if df.loc[i, "tennis"] == "tennis":
            row.append(1)
        else:
            row.append(0)
        row_list.append(row)
    new_df = pd.DataFrame(row_list, columns=df.columns.values)
    return new_df


def find_forest_splits(forest, feature_names, feature):
    tot_split=[]
    for i in range (0, len(forest.estimators_)):
        estimator= forest.estimators_[i]
        tree_feature = estimator.tree_.feature
        feature_new = []
        for node in tree_feature:
            feature_new.append(feature_names[node])
        split_list=[]
        for i in range (0, len(feature_new)):
            if feature_new[i] == feature and estimator.tree_.threshold[i] != -2:    #used -2 when leaf node
                split_list.append(estimator.tree_.threshold[i])
        tot_split.append(split_list)
    return tot_split


def old_tennis_test():
    #test to compare an output of mine to sklearn output and try random forest

    tennis1000 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/tennis/tennis1000.csv")
    new_tennis1000 = encode(tennis1000)
    feature_names=["sun", "wind", "humidity", "season", "time", "temp", "travel"]
    class_names=["no tennis", "tennis"]

    # set up subsets
    tennis1000_train = new_tennis1000.loc[:449, ]  # 1st 500 data points
    tennis1000_validation = new_tennis1000.loc[450:, ]  # 2nd 500 data points
    x_train = tennis1000_train.loc[:, :"travel"]
    y_train = tennis1000_train.loc[:, "tennis"]
    x_validate = tennis1000_validation.loc[:, :"travel"]
    y_validate = tennis1000_validation.loc[:, "tennis"]

    # making the tree
    tennis = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=4)
    tennis = tennis.fit(x_train, y_train)

    #printing
    text=tree.export_text(tennis, feature_names=feature_names)
    print(text)

    # graphic of the tree
    plt.figure(figsize=(13, 8))
    tree.plot_tree(tennis, feature_names=feature_names, class_names=class_names, filled=True)
    plt.show()

    # validation
    y_predicted = tennis.predict(x_validate)
    accuracy = metrics.accuracy_score(y_validate, y_predicted)
    print("Decision Tree Accuracy =", accuracy)
    #accuracy of mine w/ same options = 0.822
    #accuracy of sklearn = 0.909

    #random forest creation
    tennis_forest = ensemble.RandomForestClassifier(n_estimators=100, criterion="entropy", max_depth=4)
    tennis_forest = tennis_forest.fit(x_train, y_train)

    #graphic of random forest
    #extremely tiny and can't read
    #fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (10,10))
    #for i in range(0, 4):
    #    if i < 2:
    #        n=0
    #        l=i
    #    else:
    #        l= i % 2
    #        n=1
    #    tree.plot_tree(tennis_forest.estimators_[i],
    #                   feature_names=feature_names,
    #                   class_names=class_names,
    #                   filled=True,
    #                   ax=axes[n, l])
    #    axes[n, l].set_title('Estimator: ' + str(i), fontsize=11)
    #one at a time
    for i in range (0, 4):
        plt.figure(figsize=(13, 8))
        tree.plot_tree(tennis_forest.estimators_[i], feature_names=feature_names, class_names=class_names, filled=True)
        #plt.show()

    #print random forest
    for i in range (0, len(tennis_forest.estimators_)):
        print("\nEstimator", i, ":")
        text=tree.export_text(tennis_forest.estimators_[i], feature_names=feature_names)
        print(text)

    #find forest splits
    split_list=find_forest_splits(tennis_forest, feature_names, "temp")
    for i in range (0, len(split_list)):
        print("\nEstimator", i, "temp splits:")
        for j in range (0, len(split_list[i])):
            print("\t", split_list[i][j])
    splitdf=pd.DataFrame(split_list)
    split_array=splitdf.to_numpy()
    print("\nMean of temp splits:", np.nanmean(split_array))
    quantiles=np.nanquantile(split_array, [0, .25, .5, .75, 1])
    print("Minimum of temp splits:", quantiles[0])
    print("Q1 of temp splits:", quantiles[1])
    print("Median of temp splits:", quantiles[2])
    print("Q3 of temp splits:", quantiles[3])
    print("Maximum of temp splits:", quantiles[4])


    #random forest validation
    y_predicted = tennis_forest.predict(x_validate)
    accuracy = metrics.accuracy_score(y_validate, y_predicted)
    print("\nRandom Forest Accuracy =", accuracy)
    #accuracy of my tree w/ same options = 0.822
    #accuracy of sklearn tree = 0.909
    #accuracy of sklearn random forest = 0.915


def hitters_test():
    hitters = pd.read_csv("C:/Users/hough/Documents/research/data/hitters.csv")
    hitters= hitters.loc[:, "AtBat":]
    hitters = hitters.dropna()
    hitters = hitters.reset_index(drop=True)
    hitters= pd.get_dummies(hitters, columns=["League", "Division"])
    x_train = hitters.loc[:149, :"Errors"]
    x_validate= hitters.loc[150:, :"Errors"]
    y_train= hitters.loc[:149, "Salary"]
    y_validate= hitters.loc[150:, "Salary"]

    hitters_tree= tree.DecisionTreeRegressor(splitter= "best", max_depth= 2)
    hitters_tree= hitters_tree.fit(x_train, y_train)

    plt.figure(figsize=(12, 7))
    tree.plot_tree(hitters_tree, feature_names=["AtBat", "Hits", "HmRun", "Runs", "RBI", "Walks", "Years", "CAtBat",
                                                "CHits", "CHmRun", "CRuns", "CRBI", "CWalks", "League", "Division",
                                                "PutOuts", "Assists", "Errors"], filled=True)
    plt.show()

    y_predicted = hitters_tree.predict(x_validate)
    #accuracy = metrics.accuracy_score(y_validate, y_predicted)
    #print("Decision Tree Accuracy =", accuracy)


def new_tennis_test():
    tennis1000 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/tennis/new_tennis1000.csv")
    new_tennis1000 = encode(tennis1000)
    feature_names = ["sun", "wind", "humidity", "season", "time", "temp", "travel"]
    class_names = ["no tennis", "tennis"]

    # set up subsets
    x = new_tennis1000[feature_names]
    y = new_tennis1000.tennis
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # making the tree
    tennis = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=4)
    tennis = tennis.fit(x_train, y_train)

    # printing tree
    text = tree.export_text(tennis, feature_names=feature_names)
    print(text)

    # graphic of the tree
    plt.figure(figsize=(13, 8))
    tree.plot_tree(tennis, feature_names=feature_names, class_names=class_names, filled=True)
    plt.show()

    # validation
    y_predicted = tennis.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_predicted)
    print("Decision Tree Accuracy =", accuracy)

    # random forest creation
    tennis_forest = ensemble.RandomForestClassifier(n_estimators=5, criterion="entropy", max_depth=3)
    tennis_forest = tennis_forest.fit(x_train, y_train)

    # print random forest
    for i in range(0, len(tennis_forest.estimators_)):
        print("\nEstimator", i, ":")
        text = tree.export_text(tennis_forest.estimators_[i], feature_names=feature_names)
        print(text)

    # find forest splits
    split_list = find_forest_splits(tennis_forest, feature_names, "temp")
    for i in range(0, len(split_list)):
        print("\nEstimator", i, "temp splits:")
        for j in range(0, len(split_list[i])):
            print("\t", split_list[i][j])
    splitdf = pd.DataFrame(split_list)
    split_array = splitdf.to_numpy()
    print("\nMean of temp splits:", np.nanmean(split_array))
    quantiles = np.nanquantile(split_array, [0, .25, .5, .75, 1])
    print("Minimum of temp splits:", quantiles[0])
    print("Q1 of temp splits:", quantiles[1])
    print("Median of temp splits:", quantiles[2])
    print("Q3 of temp splits:", quantiles[3])
    print("Maximum of temp splits:", quantiles[4])

    # random forest validation
    y_predicted = tennis_forest.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_predicted)
    print("\nRandom Forest Accuracy =", accuracy)

    #fun
    dict_list=[]
    for i in range(len(tennis_forest.estimators_)):
        estimator = tennis_forest.estimators_[i]
        tree_feature = estimator.tree_.feature
        feature_new = []
        for node in tree_feature:
            if node == -2:
                feature_new.append('leaf')
            else:
                feature_new.append(feature_names[node])
        dict = {"p": feature_new[0]}
        i = 1
        split = "l"
        for i in range(1, len(feature_new)):
            dict[split] = feature_new[i]
            if i == len(feature_new) - 1:
                break
            if feature_new[i] != 'leaf':
                split += "l"
            elif feature_new[i] == 'leaf' and feature_new[i - 1] == 'leaf':
                split = split[:-2] + 'r'
            else:
                split = split[:-1] + "r"

            while split in dict:
                split = split[:-2] + 'r'
        dict_list.append(dict)
    print(dict_list)

    nodes=["p", "l", "ll", "lll", "llr", "lr", "lrl", "lrr", "r", "rl", "rll", "rlr", "rr", "rrl", "rrr"]   # for trees w max depth = 3
    row_list=[]
    feature_names.append("leaf")
    for node in nodes:
        feature_list = []
        print(node)
        for i in range (len(dict_list)):
            if node in dict_list[i]:
                feature_list.append(dict_list[i][node])
        print("\t", feature_list)
        feature_sums = {}
        for name in feature_names:
            feature_sums[name] = 0
        for f in feature_list:
            feature_sums[f] += 1
        print("\t", feature_sums)
        tot=len(feature_list)
        feature_fractions=[]
        for f in feature_sums.keys():
            feature_fractions.append(feature_sums[f] / tot)
        print("\t", feature_fractions)
        row_list.append(feature_fractions)

    df=pd.DataFrame(row_list, index=nodes, columns=feature_names)
    print(df)


def main():
    #old_tennis_test()
    #hitters_test()
    new_tennis_test()


if __name__ == '__main__':
    main()