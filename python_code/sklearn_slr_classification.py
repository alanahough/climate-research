from sklearn import tree, metrics, ensemble
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import math
import pylab

MAX_DEPTH = 5
MAX_FEATURES = 20
MAX_SAMPLES = 2000

def find_forest_splits(forest, feature_names, feature, firstsplit=False):
    """
    Determines the split values from all the trees in the forest for the splits of a specific feature.
    :param forest: a forest that has already been fit with training data
    :param feature_names: list of all the feature names from the dataframe used to fit the forest
    :param feature: string of specific feature to find the splits for
    :param firstsplit: boolean value--True returns values from only the first occurrence of the feature's split in
                        each tree--False returns values from all of the splits of the specified feature from each tree
    :return: a list of lists, each inner list contains the desired feature's split values for a tree-- the length of
             the outer list is equal to the number of trees in the forest
    """
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
                if firstsplit == True:
                    break
        tot_split.append(split_list)
    return tot_split


def split_stats(split_list, split_feature):
    """
    Calculates the minimum, 1st quartile, median, mean, 3rd quartile, and maximum of the splits in the split_list.
    :param split_list: a list of lists, each inner list contains the desired feature's split values for a tree
    :param split_feature: string of the feature whose split values are in split_list
    :return: splitdf, quantiles_list -- splitdf is a pandas dataframe created from split_list.  Each row of the
                                        dataframe is a different tree in the forest.
                                     -- quantiles_list is a list of the calculated statistics of split_list.  The
                                        order of the list is [min, Q1, median, Q3, max, mean]
    """
    #for i in range(0, len(split_list)):
    #    print("\nEstimator", i, split_feature, "splits:")
    #    for j in range(0, len(split_list[i])):
    #        print("\t", split_list[i][j])
    splitdf = pd.DataFrame(split_list)
    split_array = splitdf.to_numpy()
    print("\nMean of", split_feature, "splits:", np.nanmean(split_array))
    quantiles = np.nanquantile(split_array, [0, .25, .5, .75, 1])
    print("Minimum of", split_feature, "splits:", quantiles[0])
    print("Q1 of", split_feature, "splits:", quantiles[1])
    print("Median of", split_feature, "splits:", quantiles[2])
    print("Q3 of", split_feature, "splits:", quantiles[3])
    print("Maximum of", split_feature, "splits:", quantiles[4])
    quantiles_list=[val for val in quantiles]
    quantiles_list.append(np.nanmean(split_array))
    return splitdf, quantiles_list


def splits_table(forest, feature_names):
    """
    Creates a dataframe that shows what fraction of the time in this forest that each feature is used at each
    split point.
    :param forest: a forest that has already been fit with training data
    :param feature_names: list of all the feature names from the dataframe used to fit the forest
    :return: dataframe that shows what fraction of the time in this forest that each feature is used at each
             split point -- the rows of the dataframe are the split names (p, l, lr, etc) and the columns are the
                            feature names
    """
    dict_list=[]
    for i in range(len(forest.estimators_)):
        estimator = forest.estimators_[i]
        tree_feature = estimator.tree_.feature
        feature_new = []
        for node in tree_feature:
            if node == -2:
                feature_new.append('leaf')
            else:
                feature_new.append(feature_names[node])
        dict = {"p": feature_new[0]}
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

    nodes = ["p", "l", "ll", "lll", "llll", "lllr", "llr", "llrl", "llrr", "lr", "lrl", "lrll", "lrlr", "lrr", "lrrl",
             "lrrr", "r", "rl", "rll", "rlll", "rllr", "rlr", "rlrl", "rlrr", "rr", "rrl", "rrll", "rrlr", "rrr",
             "rrrl", "rrrr"]    #for trees w/ max depth= 4
    row_list=[]
    feature_names.append("leaf")

    for node in nodes:
        feature_list = []
        for i in range (len(dict_list)):
            if node in dict_list[i]:
                feature_list.append(dict_list[i][node])
        feature_sums = {}
        for name in feature_names:
            feature_sums[name] = 0
        for f in feature_list:
            feature_sums[f] += 1
        tot=len(feature_list)
        feature_fractions=[]
        for f in feature_sums.keys():
            feature_fractions.append(feature_sums[f] / tot)
        row_list.append(feature_fractions)

    df=pd.DataFrame(row_list, index=nodes, columns=feature_names)
    return df


def slr_forest(feature_list, x_train, x_test, x_valid, y_valid, y_train, y_test, max_feat="auto", max_d=None,
               min_samp_leaf=1, max_samp=None, print_forest= False):
    # random forest creation
    forest = ensemble.RandomForestClassifier(n_estimators=1000, criterion="entropy", max_features=max_feat,
                                             max_depth=max_d, min_samples_leaf=min_samp_leaf, max_samples=max_samp)
    forest = forest.fit(x_train, y_train)

    # print random forest
    if print_forest == True:
        for i in range(0, len(forest.estimators_)):
            print("\nEstimator", i, ":")
            text = tree.export_text(forest.estimators_[i], feature_names=feature_list)
            print(text)

    # random forest validation
    y_predicted = forest.predict(x_valid)
    v_accuracy = metrics.accuracy_score(y_valid, y_predicted)
    print("\nValidation Accuracy =", v_accuracy)

    # random forest training
    y_predicted = forest.predict(x_train)
    t_accuracy = metrics.accuracy_score(y_train, y_predicted)
    print("Training Accuracy =", t_accuracy)

    return forest, v_accuracy, t_accuracy


def perform_splits(forest, feature_list, split_feature):
    """
    Find splits of a specific feature and calculate the statistics
    :param forest: a forest that has already been fit with training data
    :param feature_list: list of all the feature names from the dataframe used to fit the forest
    :param split_feature: string of specific feature to find the splits for
    :return: split_df, all, first_only--splitdf is a pandas dataframe created from split_list.  Each row of the
                                        dataframe is a different tree in the forest.
                                      -- all is a list of the calculated statistics for all the split values of
                                         split_feature.  The order of the list is [min, Q1, median, Q3, max, mean]
                                      -- first_only is a list of the calculated statistics for only the first split
                                         values of split_feature in each tree.  The order of the list is [min, Q1,
                                         median, Q3, max, mean]
    """
    # find forest splits
    split_list = find_forest_splits(forest, feature_list, split_feature)
    empty = True
    for list in split_list:
        if len(list) != 0:
            empty = False
    if empty == True:
        return None, None, None
    else:
        split_df, all = split_stats(split_list, split_feature)
        first_split_list = find_forest_splits(forest, feature_list, split_feature, firstsplit=True)
        first_only = split_stats(first_split_list, split_feature)[1]
        return split_df, all, first_only


def tree_splits(response):
    df = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/RData_parameters_sample.csv")
    slr_rcp26 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/slr_rcp26.csv")
    slr_rcp85 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/slr_rcp85.csv")
    Tgav_rcp26 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/Tgav_rcp26.csv")
    Tgav_rcp85 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/Tgav_rcp85.csv")

    years = ["2025", "2050", "2075", "2100"]
    name = ["RCP2.6", "RCP8.5"]

    slr_threshold = slr_rcp26.quantile(q=.9)
    print("SLR RCP2.6 Threshold:\n", slr_threshold)
    row_list=[]
    for i in range (slr_rcp26.shape[0]):
        row=[]
        for j in range (4):
            if slr_rcp26.iloc[i, j] >= slr_threshold.iloc[j]:
                row.append("high")
            else:
                row.append("low")
        row_list.append(row)
    slr_rcp26_classify= pd.DataFrame(row_list, columns= years)

    slr_threshold = slr_rcp85.quantile(q=.9)
    print("SLR RCP8.5 Threshold:\n", slr_threshold)
    row_list = []
    for i in range(slr_rcp85.shape[0]):
        row = []
        for j in range(4):
            if slr_rcp85.iloc[i, j] >= slr_threshold.iloc[j]:
                row.append("high")
            else:
                row.append("low")
        row_list.append(row)
    slr_rcp85_classify = pd.DataFrame(row_list, columns=years)

    Tgav_threshold = Tgav_rcp26.quantile(q=.9)
    print("Tgav RCP2.6 Threshold:\n", Tgav_threshold)
    row_list = []
    for i in range(Tgav_rcp26.shape[0]):
        row = []
        for j in range(4):
            if Tgav_rcp26.iloc[i, j] >= Tgav_threshold.iloc[j]:
                row.append("high")
            else:
                row.append("low")
        row_list.append(row)
    Tgav_rcp26_classify = pd.DataFrame(row_list, columns=years)

    Tgav_threshold = Tgav_rcp85.quantile(q=.9)
    print("Tgav RCP8.5 Threshold:\n", Tgav_threshold)
    row_list = []
    for i in range(Tgav_rcp85.shape[0]):
        row = []
        for j in range(4):
            if Tgav_rcp85.iloc[i, j] >= Tgav_threshold.iloc[j]:
                row.append("high")
            else:
                row.append("low")
        row_list.append(row)
    Tgav_rcp85_classify = pd.DataFrame(row_list, columns=years)

    df_slr_rcp26 = df.join(slr_rcp26_classify, how="outer")
    df_slr_rcp85 = df.join(slr_rcp85_classify, how="outer")
    df_Tgav_rcp26 = df.join(Tgav_rcp26_classify, how="outer")
    df_Tgav_rcp85 = df.join(Tgav_rcp85_classify, how="outer")

    years = ["2025", "2050", "2075", "2100"]
    name = ["RCP2.6", "RCP8.5"]
    if response == "SLR":
        dflist = [df_slr_rcp26, df_slr_rcp85]
        path = [[r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\classification_forest\rcp26_2025_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\classification_forest\rcp26_2050_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\classification_forest\rcp26_2075_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\classification_forest\rcp26_2100_splits_d4.csv'],
                [r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\classification_forest\rcp85_2025_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\classification_forest\rcp85_2050_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\classification_forest\rcp85_2075_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\classification_forest\rcp85_2100_splits_d4.csv']]
        table_path = [[r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\classification_forest\rcp26_2025_split_table_d4.csv',
                       r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\classification_forest\rcp26_2050_split_table_d4.csv',
                       r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\classification_forest\rcp26_2075_split_table_d4.csv',
                       r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\classification_forest\rcp26_2100_split_table_d4.csv'],
                      [r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\classification_forest\rcp85_2025_split_table_d4.csv',
                       r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\classification_forest\rcp85_2050_split_table_d4.csv',
                       r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\classification_forest\rcp85_2075_split_table_d4.csv',
                       r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\classification_forest\rcp85_2100_split_table_d4.csv']]
    elif response == "Tgav":
        dflist = [df_Tgav_rcp26, df_Tgav_rcp85]
        path = [[r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\classification_forest\rcp26_2025_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\classification_forest\rcp26_2050_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\classification_forest\rcp26_2075_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\classification_forest\rcp26_2100_splits_d4.csv'],
                [r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\classification_forest\rcp85_2025_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\classification_forest\rcp85_2050_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\classification_forest\rcp85_2075_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\classification_forest\rcp85_2100_splits_d4.csv']]
        table_path = [[r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\classification_forest\rcp26_2025_split_table_d4.csv',
                       r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\classification_forest\rcp26_2050_split_table_d4.csv',
                       r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\classification_forest\rcp26_2075_split_table_d4.csv',
                       r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\classification_forest\rcp26_2100_split_table_d4.csv'],
                      [r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\classification_forest\rcp85_2025_split_table_d4.csv',
                       r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\classification_forest\rcp85_2050_split_table_d4.csv',
                       r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\classification_forest\rcp85_2075_split_table_d4.csv',
                       r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\classification_forest\rcp85_2100_split_table_d4.csv']]
    first_quartile_data = []
    all_quartile_data = []

    for i in range(len(dflist)):
        responsedf = dflist[i]
        responsedf = responsedf.dropna()
        importances_info = []
        fig, axs = plt.subplots(1, 4)
        for j in range(len(years)):
            features = df.columns.tolist()
            yr = years[j]
            # set up subsets
            x = responsedf[features]
            y = responsedf[yr]
            x_train, x_rest, y_train, y_rest = train_test_split(x, y,
                                                                test_size=0.4)  # train= 60%, validation + test= 40%
            # split up rest of 40% into validation & test
            x_validation, x_test, y_validation, y_test = train_test_split(x_rest, y_rest,
                                                                          test_size=.5)  # validation= 20%, test= 20%

            all_label = name[i] + yr + " all splits"
            first_label = name[i] + yr + " first split"
            all = (all_label,)
            first = (first_label,)
            forest, validation_acc, training_acc = slr_forest(features, x_train, x_test, x_validation, y_validation,
                                                              y_train, y_test)
            split_df, all_quartiles, first_quartiles=perform_splits(forest, features,"S.temperature")
            if isinstance(split_df, pd.DataFrame):
                pass
            else:
                continue

            split_df.to_csv(path[i][j], index=False)
            for n in range(len(all_quartiles)):
                all += (all_quartiles[n],)
                first += (first_quartiles[n],)
            first_quartile_data.append(first)
            all_quartile_data.append(all)

            table_df = splits_table(forest, features)
            table_df.to_csv(table_path[i][j], index=True)

            # importances plot
            importances = forest.feature_importances_
            std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                         axis=0)
            indices = np.argsort(importances)[::-1]
            importances_list = []
            std_list = []
            features_list = []
            for idx in indices:
                if importances[idx] > .025:
                    importances_list.append(importances[idx])
                    std_list.append(std[idx])
                    features_list.append(features[idx])
            importances_info.append([importances_list, std_list, features_list])
            axs[j].bar(range(len(importances_list)), importances_list, color="tab:blue",
                       yerr=std_list, align="center")
            title = yr
            axs[j].set_title(title)
            axs[j].set_xticks(range(len(importances_list)))
            axs[j].set_xticklabels(features_list, rotation=90)
            axs[j].set_ylim(top=1.0)
            axs[j].set_ylim(bottom= 0.0)
        main_title = response + " " + name[i] + " Feature Importances"
        fig.suptitle(main_title, fontsize=15)
        fig.text(0.52, 0.04, 'Features', ha='center', fontsize=12)
        fig.text(0.04, 0.5, 'Relative Importance', va='center', rotation='vertical', fontsize=12)
        plt.show()

    df_first_quartile = pd.DataFrame(first_quartile_data, columns=["Name", "0%", "25%", "50%", "75%", "100%", "Mean"])
    df_all_quartile = pd.DataFrame(all_quartile_data, columns=["Name", "0%", "25%", "50%", "75%", "100%", "Mean"])
    return df_first_quartile, df_all_quartile


def slr_output():
    slr_first_depth4_quartile, slr_all_depth4_quartile = tree_splits("SLR")
    slr_first_depth4_quartile.to_csv(r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\classification_forest\first_split_stats.csv',
                                     index=False)
    slr_all_depth4_quartile.to_csv(r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\classification_forest\all_split_stats.csv',
                                   index=False)


def Tgav_output():
    Tgav_first_depth4_quartile, Tgav_all_depth4_quartile = tree_splits("Tgav")
    Tgav_first_depth4_quartile.to_csv(
        r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\classification_forest\first_split_stats.csv',
        index=False)
    Tgav_all_depth4_quartile.to_csv(r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\classification_forest\all_split_stats.csv',
                                    index=False)


def max_features(max_d=None, min_samp_leaf=1, max_samp=None, print_threshold=True):
    df = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/RData_parameters_sample.csv")
    slr_rcp85 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/slr_rcp85.csv")
    years = ["2025", "2050", "2075", "2100"]
    slr_threshold = slr_rcp85.quantile(q=.9)
    if print_threshold is True:
        print("SLR RCP8.5 Threshold:\n", slr_threshold)
    row_list = []
    for i in range(slr_rcp85.shape[0]):
        row = []
        for j in range(4):
            if slr_rcp85.iloc[i, j] >= slr_threshold.iloc[j]:
                row.append("high")
            else:
                row.append("low")
        row_list.append(row)
    slr_rcp85_classify = pd.DataFrame(row_list, columns=years)
    df_slr_rcp85 = df.join(slr_rcp85_classify, how="outer")
    df_slr_rcp85 = df_slr_rcp85.dropna()
    features = df.columns.tolist()
    yr="2100"

    # set up subsets
    x = df_slr_rcp85[features]
    y = df_slr_rcp85[yr]
    x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.4)  # train= 60%, validation + test= 40%
    # split up rest of 40% into validation & test
    x_validation, x_test, y_validation, y_test = train_test_split(x_rest, y_rest,
                                                                  test_size=.5)  # validation= 20%, test= 20%

    max_list = ["sqrt", 10, 15, 20, 25, 30, 35]
    v_accuracy_list=[]
    t_accuracy_list=[]
    for max in max_list:
        print("\nmax_features =", max)
        forest, v_accuracy, t_accuracy= slr_forest(features, x_train, x_test, x_validation, y_validation, y_train,
                                                   y_test, max_feat=max, max_d=max_d, min_samp_leaf=min_samp_leaf,
                                                   max_samp=max_samp)
        v_accuracy_list.append(v_accuracy)
        t_accuracy_list.append(t_accuracy)
    max_list[0] = math.sqrt(38)
    plt.scatter(max_list, v_accuracy_list, label="Validation Accuracy")
    plt.scatter(max_list, t_accuracy_list, label="Training Accuracy")
    plt.legend()
    plt.xlabel("max_features")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs max_features for SLR RCP8.5 2100")
    plt.show()


def max_depth(max_feat="auto", min_samp_leaf=1, max_samp=None, print_threshold=True):
    df = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/RData_parameters_sample.csv")
    slr_rcp85 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/slr_rcp85.csv")
    years = ["2025", "2050", "2075", "2100"]
    slr_threshold = slr_rcp85.quantile(q=.9)
    if print_threshold is True:
        print("SLR RCP8.5 Threshold:\n", slr_threshold)
    row_list = []
    for i in range(slr_rcp85.shape[0]):
        row = []
        for j in range(4):
            if slr_rcp85.iloc[i, j] >= slr_threshold.iloc[j]:
                row.append("high")
            else:
                row.append("low")
        row_list.append(row)
    slr_rcp85_classify = pd.DataFrame(row_list, columns=years)
    df_slr_rcp85 = df.join(slr_rcp85_classify, how="outer")
    df_slr_rcp85 = df_slr_rcp85.dropna()
    features = df.columns.tolist()
    yr = "2100"

    # set up subsets
    x = df_slr_rcp85[features]
    y = df_slr_rcp85[yr]
    x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.4)  # train= 60%, validation + test= 40%
    # split up rest of 40% into validation & test
    x_validation, x_test, y_validation, y_test = train_test_split(x_rest, y_rest,
                                                                  test_size=.5)  # validation= 20%, test= 20%

    max_depth = [2, 3, 4, 5, 6, 7]
    v_accuracy_list=[]
    t_accuracy_list=[]
    for max in max_depth:
        print("\nmax_depth =", max)
        forest, v_accuracy, t_accuracy= slr_forest(features, x_train, x_test, x_validation, y_validation, y_train,
                                                   y_test, max_d=max, max_feat=max_feat, min_samp_leaf=min_samp_leaf,
                                                   max_samp=max_samp)
        v_accuracy_list.append(v_accuracy)
        t_accuracy_list.append(t_accuracy)
    plt.scatter(max_depth, v_accuracy_list, label="Validation Accuracy")
    plt.scatter(max_depth, t_accuracy_list, label="Training Accuracy")
    plt.legend()
    plt.xlabel("max_depth")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs max_depth for SLR RCP8.5 2100 (max_features = 25)")
    plt.show()


def min_samples_leaf(max_feat="auto", max_d=None, max_samp=None, print_threshold=True):
    df = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/RData_parameters_sample.csv")
    slr_rcp85 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/slr_rcp85.csv")
    years = ["2025", "2050", "2075", "2100"]
    slr_threshold = slr_rcp85.quantile(q=.9)
    if print_threshold is True:
        print("SLR RCP8.5 Threshold:\n", slr_threshold)
    row_list = []
    for i in range(slr_rcp85.shape[0]):
        row = []
        for j in range(4):
            if slr_rcp85.iloc[i, j] >= slr_threshold.iloc[j]:
                row.append("high")
            else:
                row.append("low")
        row_list.append(row)
    slr_rcp85_classify = pd.DataFrame(row_list, columns=years)
    df_slr_rcp85 = df.join(slr_rcp85_classify, how="outer")
    df_slr_rcp85 = df_slr_rcp85.dropna()
    features = df.columns.tolist()
    yr = "2100"

    # set up subsets
    x = df_slr_rcp85[features]
    y = df_slr_rcp85[yr]
    x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.4)  # train= 60%, validation + test= 40%
    # split up rest of 40% into validation & test
    x_validation, x_test, y_validation, y_test = train_test_split(x_rest, y_rest,
                                                                  test_size=.5)  # validation= 20%, test= 20%

    min_leaf = [1, 2, 4, 8, 12, 16, 20, 24, 28]
    v_accuracy_list=[]
    t_accuracy_list=[]
    for min in min_leaf:
        print("\nmin_samples_leaf =", min)
        forest, v_accuracy, t_accuracy= slr_forest(features, x_train, x_test, x_validation, y_validation, y_train,
                                                   y_test, min_samp_leaf=min, max_feat=max_feat, max_d=max_d,
                                                   max_samp=max_samp)
        v_accuracy_list.append(v_accuracy)
        t_accuracy_list.append(t_accuracy)
    plt.scatter(min_leaf, v_accuracy_list, label="Validation Accuracy")
    plt.scatter(min_leaf, t_accuracy_list, label="Training Accuracy")
    plt.legend()
    plt.xticks(min_leaf)
    plt.xlabel("min_samples_leaf")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs min_samples_leaf for SLR RCP8.5 2100 \n(max_depth = 4, max_features = 25)")
    plt.show()


def max_samples(max_feat="auto", max_d=None, min_samp_leaf=1, print_threshold=True):
    df = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/RData_parameters_sample.csv")
    slr_rcp85 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/slr_rcp85.csv")
    years = ["2025", "2050", "2075", "2100"]
    slr_threshold = slr_rcp85.quantile(q=.9)
    if print_threshold is True:
        print("SLR RCP8.5 Threshold:\n", slr_threshold)
    row_list = []
    for i in range(slr_rcp85.shape[0]):
        row = []
        for j in range(4):
            if slr_rcp85.iloc[i, j] >= slr_threshold.iloc[j]:
                row.append("high")
            else:
                row.append("low")
        row_list.append(row)
    slr_rcp85_classify = pd.DataFrame(row_list, columns=years)
    df_slr_rcp85 = df.join(slr_rcp85_classify, how="outer")
    df_slr_rcp85 = df_slr_rcp85.dropna()
    features = df.columns.tolist()
    yr = "2100"

    # set up subsets
    x = df_slr_rcp85[features]
    y = df_slr_rcp85[yr]
    x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.4)  # train= 60%, validation + test= 40%
    # split up rest of 40% into validation & test
    x_validation, x_test, y_validation, y_test = train_test_split(x_rest, y_rest,
                                                                  test_size=.5)  # validation= 20%, test= 20%
    max_samples = [1000, 2000, 3000, 4000, 5000, 6000]
    v_accuracy_list=[]
    t_accuracy_list=[]
    for max in max_samples:
        print("\nmax_samples =", max)
        forest, v_accuracy, t_accuracy= slr_forest(features, x_train, x_test, x_validation, y_validation, y_train,
                                                   y_test, max_samp=max, max_feat=max_feat, max_d=max_d,
                                                   min_samp_leaf=min_samp_leaf)
        v_accuracy_list.append(v_accuracy)
        t_accuracy_list.append(t_accuracy)
    plt.scatter(max_samples, v_accuracy_list, label="Validation Accuracy")
    plt.scatter(max_samples, t_accuracy_list, label="Training Accuracy")
    plt.legend()
    plt.xticks(max_samples)
    plt.xlabel("max_samples")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs max_samples for SLR RCP8.5 2100\n(max_depth = 4, max_features = 25)")
    plt.show()


def parameter_loop():
    stop = 1
    counter = 1
    max_depth_val = None
    max_samples_val = None
    while stop != 0:
        print("Iteration", counter)
        max_features(max_d=max_depth_val, max_samp=max_samples_val, print_threshold=False)
        max_features_val = int(input("Max features: "))
        max_depth(max_feat=max_features_val, max_samp=max_samples_val, print_threshold=False)
        max_depth_val = int(input("Max depth: "))
        max_samples(max_feat=max_features_val, max_d=max_depth_val, print_threshold=False)
        max_samples_val = int(input("Max samples: "))
        print("Iteration", counter, "Summary")
        print("\tMax features =", max_features_val)
        print("\tMax depth = ", max_depth_val)
        print("\tMax samples =", max_samples_val)
        stop = int(input("Enter 0 to stop"))
        counter += 1


def feature_color_dict(params_df):
    features = params_df.columns.tolist()
    features.append("Other")
    color_map = pylab.get_cmap('gist_rainbow')
    color_dict = {}
    for i in range(len(features)):
        color = color_map(i/len(features))
        color_dict[features[i]] = color
    return color_dict


def slr_stacked_importances_plot(importance_threshold):
    df = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/RData_parameters_sample.csv")
    slr_rcp26 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/slr_rcp26.csv")
    slr_rcp85 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/slr_rcp85.csv")
    slr_rcp26_5step = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/slr_rcp26_5yrstep.csv")
    slr_rcp85_5step = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/slr_rcp85_5yrstep.csv")

    #years = ["2025", "2050", "2075", "2100"]
    years = ["2020", "2025", "2030", "2035", "2040", "2045", "2050", "2055", "2060", "2065", "2070", "2075", "2080",
             "2085", "2090", "2095", "2100", "2105", "2110", "2115", "2120", "2125", "2130", "2135", "2140", "2145",
             "2150"]
    name = ["RCP2.6", "RCP8.5"]

    slr_threshold = slr_rcp26_5step.quantile(q=.9)
    print("SLR RCP2.6 Threshold:\n", slr_threshold)
    row_list = []
    for i in range(slr_rcp26_5step.shape[0]):
        row = []
        for j in range(len(years)):
            if slr_rcp26_5step.iloc[i, j] >= slr_threshold.iloc[j]:
                row.append("high")
            else:
                row.append("low")
        row_list.append(row)
    slr_rcp26_classify = pd.DataFrame(row_list, columns=years)

    slr_threshold = slr_rcp85_5step.quantile(q=.9)
    print("SLR RCP8.5 Threshold:\n", slr_threshold)
    row_list = []
    for i in range(slr_rcp85_5step.shape[0]):
        row = []
        for j in range(len(years)):
            if slr_rcp85_5step.iloc[i, j] >= slr_threshold.iloc[j]:
                row.append("high")
            else:
                row.append("low")
        row_list.append(row)
    slr_rcp85_classify = pd.DataFrame(row_list, columns=years)

    df_slr_rcp26 = df.join(slr_rcp26_classify, how="outer")
    df_slr_rcp85 = df.join(slr_rcp85_classify, how="outer")
    dflist=[df_slr_rcp26, df_slr_rcp85]

    #set color for each feature
    color_dict = feature_color_dict(df)

    for i in range(len(dflist)):
        responsedf = dflist[i]
        responsedf = responsedf.dropna()
        importances_info = {}
        for j in range(len(years)):
            features = df.columns.tolist()
            yr = years[j]
            # set up subsets
            x = responsedf[features]
            y = responsedf[yr]
            x_train, x_rest, y_train, y_rest = train_test_split(x, y,
                                                                test_size=0.4)  # train= 60%, validation + test= 40%
            # split up rest of 40% into validation & test
            x_validation, x_test, y_validation, y_test = train_test_split(x_rest, y_rest,
                                                                          test_size=.5)  # validation= 20%, test= 20%
            # make forest -- max depth = 4, max features = 25
            forest, validation_acc, training_acc = slr_forest(features, x_train, x_test, x_validation, y_validation,
                                                              y_train, y_test, max_d=MAX_DEPTH, max_feat=MAX_FEATURES,
                                                              max_samples= MAX_SAMPLES)
            # stacked importances dictionary
            importances = forest.feature_importances_
            indices = np.argsort(importances)[::-1]

            sum = 0
            if j == 0:
                #set up importances_info dictionary with first pass through
                for idx in indices:
                    feature = features[idx]
                    importance = importances[idx]
                    if importance > importance_threshold:
                        importances_info[feature] = [importance]
                        sum += importance
                importances_info['Other'] = [1 - sum]
            else:
                #add to importances_info dictionary that's already set up
                for idx in indices:
                    feature = features[idx]
                    importance = importances[idx]
                    if importance > importance_threshold:
                        if feature in importances_info:
                            importances_info[feature].append(importance)
                        else:
                            list = [0 for n in range(0, j)]
                            list.append(importance)
                            importances_info[feature] = list
                        sum += importance
                importances_info['Other'].append(1 - sum)
                for f in importances_info:
                    if len(importances_info[f]) < (j + 1):
                        importances_info[f].append(0)

        # stacked importances plot
        x = np.arange(len(years))
        bottom = None
        for feature in importances_info:
            color = color_dict[feature]
            if feature == "Other":
                pass
            else:
                if bottom is None:
                    plt.bar(x, importances_info[feature], label=feature, c=color)
                    bottom = np.array(importances_info[feature])
                else:
                    plt.bar(x, importances_info[feature], bottom=bottom, label=feature, c=color)
                    bottom += np.array(importances_info[feature])
        plt.bar(x, importances_info["Other"], bottom=bottom, label="Other (< 5%)")
        plt.ylabel("Relative Importances")
        yrs_10=[]
        for yr in years:
            if int(yr) % 10 == 0:
                yrs_10.append(yr)
            else:
                yrs_10.append("")
        plt.xticks(x, yrs_10)
        plt.xlabel('Year')
        title = "SLR " + name[i] + " Feature Importances"
        plt.title(title, fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.show()


def Stemp_histograms(df_2025, df_2050, df_2075, df_2100, rcp, first_only=False):
    if first_only is True:
        list_2025 = df_2025["0"].values.tolist()
        list_2050 = df_2050["0"].values.tolist()
        list_2075 = df_2075["0"].values.tolist()
        list_2100 = df_2100["0"].values.tolist()
    else:
        list_2025 = df_2025.stack().tolist()
        list_2050 = df_2050.stack().tolist()
        list_2075 = df_2075.stack().tolist()
        list_2100 = df_2100.stack().tolist()
    df_dict={"2025": list_2025, "2050": list_2050, "2075": list_2075, "2100": list_2100}

    fig, axs = plt.subplots(1, 4)
    i = 0
    bin_seq=np.arange(0, 10, step=.5)
    for yr in df_dict:
        axs[i].hist(df_dict[yr], bins=bin_seq, edgecolor='white')
        axs[i].set_title(yr)
        if first_only is True:
            axs[i].set_ylim(top=450)
        else:
            axs[i].set_ylim(top=3250)
        axs[i].set_ylim(bottom=0)
        axs[i].set_xlim(right=9.5)
        axs[i].set_xlim(left=1.5)
        i += 1
    main_title = "SLR " + rcp + " Histogram of S.temperature Split Values"
    if first_only is True:
        main_title += " (First Split Only)"
    fig.suptitle(main_title)
    fig.text(0.52, 0.04, 'S.temperature Split', ha='center')
    fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')
    plt.show()


if __name__ == '__main__':
    # slr_output()
    # Tgav_output()
    # slr_stacked_importances_plot(.05)
    rcp26_2025 = pd.read_csv(
        "C:/Users/hough/Documents/research/data/new_csv/SLR_splits/classification_forest/rcp26_2025_splits_d4.csv")
    rcp26_2050 = pd.read_csv(
        "C:/Users/hough/Documents/research/data/new_csv/SLR_splits/classification_forest/rcp26_2050_splits_d4.csv")
    rcp26_2075 = pd.read_csv(
        "C:/Users/hough/Documents/research/data/new_csv/SLR_splits/classification_forest/rcp26_2075_splits_d4.csv")
    rcp26_2100 = pd.read_csv(
        "C:/Users/hough/Documents/research/data/new_csv/SLR_splits/classification_forest/rcp26_2100_splits_d4.csv")
    rcp85_2025 = pd.read_csv(
        "C:/Users/hough/Documents/research/data/new_csv/SLR_splits/classification_forest/rcp85_2025_splits_d4.csv")
    rcp85_2050 = pd.read_csv(
        "C:/Users/hough/Documents/research/data/new_csv/SLR_splits/classification_forest/rcp85_2050_splits_d4.csv")
    rcp85_2075 = pd.read_csv(
        "C:/Users/hough/Documents/research/data/new_csv/SLR_splits/classification_forest/rcp85_2075_splits_d4.csv")
    rcp85_2100 = pd.read_csv(
        "C:/Users/hough/Documents/research/data/new_csv/SLR_splits/classification_forest/rcp85_2100_splits_d4.csv")
    # Stemp_histograms(rcp26_2025, rcp26_2050, rcp26_2075, rcp26_2100, "RCP 2.6", first_only=True)

    df = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/RData_parameters_sample.csv")
    slr_rcp85 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/slr_rcp85.csv")
    years = ["2025", "2050", "2075", "2100"]
    slr_threshold = slr_rcp85.quantile(q=.9)
    print("SLR RCP8.5 Threshold:\n", slr_threshold)
    row_list = []
    for i in range(slr_rcp85.shape[0]):
        row = []
        for j in range(4):
            if slr_rcp85.iloc[i, j] >= slr_threshold.iloc[j]:
                row.append("high")
            else:
                row.append("low")
        row_list.append(row)
    slr_rcp85_classify = pd.DataFrame(row_list, columns=years)
    df_slr_rcp85 = df.join(slr_rcp85_classify, how="outer")
    df_slr_rcp85 = df_slr_rcp85.dropna()
    features = df.columns.tolist()
    yr = "2100"

    # set up subsets
    x = df_slr_rcp85[features]
    y = df_slr_rcp85[yr]
    x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.4)  # train= 60%, validation + test= 40%
    # split up rest of 40% into validation & test
    x_validation, x_test, y_validation, y_test = train_test_split(x_rest, y_rest,
                                                                  test_size=.5)  # validation= 20%, test= 20%
    for i in range(4):
        forest, v_accuracy, t_accuracy = slr_forest(features, x_train, x_test, x_validation, y_validation, y_train,
                                                    y_test, max_samp=MAX_SAMPLES, max_feat=MAX_FEATURES,
                                                    max_d=MAX_DEPTH)