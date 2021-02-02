from sklearn import tree, metrics, ensemble
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import math
import pylab
import joblib
import os


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
    features = feature_names.copy()
    dict_list=[]
    for i in range(len(forest.estimators_)):
        estimator = forest.estimators_[i]
        tree_feature = estimator.tree_.feature
        feature_new = []
        for node in tree_feature:
            if node == -2:
                feature_new.append('leaf')
            else:
                feature_new.append(features[node])
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

    # for trees w/ max depth = 5
    nodes = ["p", "l", "ll", "lll", "llll", "lllll", "llllr", "lllr", "lllrl", "lllrr", "llr", "llrl", "llrll", "llrlr",
             "llrr", "llrrl", "llrrr", "lr", "lrl", "lrll", "lrlll", "lrllr", "lrlr", "lrlrl", "lrlrr", "lrr", "lrrl",
             "lrrll", "lrrlr", "lrrr", "lrrrl", "lrrrr", "r", "rl", "rll", "rlll", "rllll", "rlllr", "rllr", "rllrl",
             "rllrr", "rlr", "rlrl", "rlrll", "rlrlr", "rlrr", "rlrrl", "rlrrr", "rr", "rrl", "rrll", "rrlll", "rrllr",
             "rrlr", "rrlrl", "rrlrr", "rrr", "rrrl", "rrrll", "rrrlr", "rrrr", "rrrrl", "rrrrr"]

    row_list=[]
    features.append("leaf")

    for node in nodes:
        feature_list = []
        for i in range (len(dict_list)):
            if node in dict_list[i]:
                feature_list.append(dict_list[i][node])
        feature_sums = {}
        for name in features:
            feature_sums[name] = 0
        for f in feature_list:
            feature_sums[f] += 1
        tot=len(feature_list)
        feature_fractions=[]
        for f in feature_sums.keys():
            feature_fractions.append(feature_sums[f] / tot)
        row_list.append(feature_fractions)

    df = pd.DataFrame(row_list, index=nodes, columns=features)
    return df


def slr_forest(feature_list, df, year, max_feat="auto", max_d=None, min_samp_leaf=1, max_samp=None,
               print_forest= False):
    # set up subsets
    x = df[feature_list]
    y = df[year]
    x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.4)  # train= 60%, validation + test= 40%
    # split up rest of 40% into validation & test
    x_validation, x_test, y_validation, y_test = train_test_split(x_rest, y_rest,
                                                                  test_size=.5)  # validation= 20%, test= 20%

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
    y_predicted = forest.predict(x_validation)
    v_accuracy = metrics.accuracy_score(y_validation, y_predicted)
    print("\nValidation Accuracy =", v_accuracy)

    # random forest training
    y_predicted = forest.predict(x_train)
    t_accuracy = metrics.accuracy_score(y_train, y_predicted)
    print("Training Accuracy =", t_accuracy)

    return forest, v_accuracy, t_accuracy


def classify_data(df, print_threshold=False):
    years = df.columns.tolist()
    threshold = df.quantile(q=.9)
    if print_threshold is True:
        print("Threshold:\n", threshold)
    row_list = []
    for i in range(df.shape[0]):
        row = []
        for j in range(len(years)):
            if df.iloc[i, j] >= threshold.iloc[j]:
                row.append("high")
            else:
                row.append("low")
        row_list.append(row)
    df_classify = pd.DataFrame(row_list, columns=years)
    return df_classify


def make_tree_and_export(parameter_sample_df, slr_df, yrs_to_output, rcp_str, forest_path, accuracy_path):
    slr_classify = classify_data(slr_df)
    df_slr_classify = parameter_sample_df.join(slr_classify, how="outer")
    df_slr_classify = df_slr_classify.dropna()
    features = parameter_sample_df.columns.tolist()
    for yr in yrs_to_output:
        forest, v_accuracy, t_accuracy = slr_forest(features, df_slr_classify, yr, max_feat=MAX_FEATURES,
                                                    max_d=MAX_DEPTH, max_samp=MAX_SAMPLES)
        forest_file_path = forest_path + rcp_str + "_" + yr + ".joblib"
        joblib.dump(forest, forest_file_path, compress=3)
        print(rcp_str, yr)
        print("\t", f"Compressed Random Forest: {np.round(os.path.getsize(forest_file_path) / 1024 / 1024, 2)} MB")
        accuracy_df = pd.DataFrame({"Validation Accuracy": [v_accuracy], "Training Accuracy": [t_accuracy]})
        accuracy_file_path = accuracy_path + rcp_str + "_" + yr + "_accuracy.csv"
        accuracy_df.to_csv(accuracy_file_path, index=False)


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
    if empty:
        return None, None, None
    else:
        split_df, all = split_stats(split_list, split_feature)
        first_split_list = find_forest_splits(forest, feature_list, split_feature, firstsplit=True)
        first_only = split_stats(first_split_list, split_feature)[1]
        return split_df, all, first_only


def tree_splits(param_sample_df, response, rcp, forests_list, year_list, folder_path):
    fig, axs = plt.subplots(1, len(year_list))
    features = param_sample_df.columns.tolist()
    first_quartile_data = []
    all_quartile_data = []
    rcp_no_space = rcp.replace(" ", "")
    rcp_no_space_no_period = rcp_no_space.replace(".", "")

    for j in range(len(year_list)):
        yr = str(year_list[j])
        all_label = response + " " + yr + " all splits"
        first_label = response + " " + yr + " first split"
        all = (all_label,)
        first = (first_label,)
        forest = forests_list[j]
        split_df, all_quartiles, first_quartiles=perform_splits(forest, features,"S.temperature")
        if isinstance(split_df, pd.DataFrame):
            pass
        else:
            continue

        split_file_path = folder_path + rcp_no_space_no_period + "_" + yr + "_splits.csv"
        split_df.to_csv(split_file_path, index=False)
        for n in range(len(all_quartiles)):
            all += (all_quartiles[n],)
            first += (first_quartiles[n],)
        first_quartile_data.append(first)
        all_quartile_data.append(all)

        table_df = splits_table(forest, features)
        split_table_file_path = folder_path + rcp_no_space_no_period + "_" + yr + "_split_table.csv"
        table_df.to_csv(split_table_file_path, index=True)

        # importances plot
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        importances_list = []
        std_list = []
        features_list = []
        for idx in indices:
            if importances[idx] > .025:
                importances_list.append(importances[idx])
                std_list.append(std[idx])
                features_list.append(features[idx])
        axs[j].bar(range(len(importances_list)), importances_list, color="tab:blue",
                       yerr=std_list, align="center")
        title = yr
        axs[j].set_title(title)
        axs[j].set_xticks(range(len(importances_list)))
        axs[j].set_xticklabels(features_list, rotation=90)
        axs[j].set_ylim(top=1.0)
        axs[j].set_ylim(bottom= 0.0)
    main_title = response + " " + rcp + " Feature Importances"
    fig.suptitle(main_title, fontsize=15)
    fig.text(0.52, 0.04, 'Features', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Relative Importance', va='center', rotation='vertical', fontsize=12)
    plt.show()

    df_first_quartile = pd.DataFrame(first_quartile_data, columns=["Name", "0%", "25%", "50%", "75%", "100%", "Mean"])
    first_file_path = folder_path + rcp_no_space_no_period + "_first_splits.csv"
    df_first_quartile.to_csv(first_file_path, index=False)

    df_all_quartile = pd.DataFrame(all_quartile_data, columns=["Name", "0%", "25%", "50%", "75%", "100%", "Mean"])
    all_file_path = folder_path + rcp_no_space_no_period + "_all_splits.csv"
    df_all_quartile.to_csv(all_file_path, index=False)


def max_features(param_samples_df, slr_df, max_d=None, min_samp_leaf=1, max_samp=None, print_threshold=True):
    slr_rcp85_classify = classify_data(slr_df, print_threshold=print_threshold)
    df_slr_rcp85 = param_samples_df.join(slr_rcp85_classify, how="outer")
    df_slr_rcp85 = df_slr_rcp85.dropna()
    features = param_samples_df.columns.tolist()
    yr="2100"

    max_list = ["sqrt", 10, 15, 20, 25, 30, 35]
    v_accuracy_list=[]
    t_accuracy_list=[]
    for max in max_list:
        print("\nmax_features =", max)
        forest, v_accuracy, t_accuracy= slr_forest(features, df_slr_rcp85, yr, max_feat=max, max_d=max_d,
                                                   min_samp_leaf=min_samp_leaf, max_samp=max_samp)
        v_accuracy_list.append(v_accuracy)
        t_accuracy_list.append(t_accuracy)
    max_list[0] = math.sqrt(38)
    plt.scatter(max_list, v_accuracy_list, label="Validation Accuracy")
    plt.scatter(max_list, t_accuracy_list, label="Training Accuracy")
    plt.legend()
    plt.xlabel("max_features")
    plt.ylabel("accuracy")
    title = "Accuracy vs max_features for SLR RCP8.5 2100 \n(max_depth = " + str(max_d) + ", min_samples_leaf = " + \
            str(min_samp_leaf) + ", max_samples = " + str(max_samp) + ")"
    plt.title(title)
    plt.show()


def max_depth(param_samples_df, slr_df, max_feat="auto", min_samp_leaf=1, max_samp=None, print_threshold=True):
    slr_rcp85_classify = classify_data(slr_df, print_threshold=print_threshold)
    df_slr_rcp85 = param_samples_df.join(slr_rcp85_classify, how="outer")
    df_slr_rcp85 = df_slr_rcp85.dropna()
    features = param_samples_df.columns.tolist()
    yr = "2100"

    max_depth = [2, 3, 4, 5, 6, 7]
    v_accuracy_list=[]
    t_accuracy_list=[]
    for max in max_depth:
        print("\nmax_depth =", max)
        forest, v_accuracy, t_accuracy = slr_forest(features, df_slr_rcp85, yr, max_feat=max_feat, max_d=max,
                                                    min_samp_leaf=min_samp_leaf, max_samp=max_samp)
        v_accuracy_list.append(v_accuracy)
        t_accuracy_list.append(t_accuracy)
    plt.scatter(max_depth, v_accuracy_list, label="Validation Accuracy")
    plt.scatter(max_depth, t_accuracy_list, label="Training Accuracy")
    plt.legend()
    plt.xlabel("max_depth")
    plt.ylabel("accuracy")
    title = "Accuracy vs max_depth for SLR RCP8.5 2100 \n(min_samples_leaf = " + str(min_samp_leaf) + \
            ", max_features = " + str(max_feat) + ", max_samples = " + str(max_samp) + ")"
    plt.title(title)
    plt.show()


def min_samples_leaf(param_samples_df, slr_df, max_feat="auto", max_d=None, max_samp=None, print_threshold=True):
    slr_rcp85_classify = classify_data(slr_df, print_threshold=print_threshold)
    df_slr_rcp85 = param_samples_df.join(slr_rcp85_classify, how="outer")
    df_slr_rcp85 = df_slr_rcp85.dropna()
    features = param_samples_df.columns.tolist()
    yr = "2100"

    min_leaf = [1, 2, 4, 8, 12, 16, 20, 24, 28]
    v_accuracy_list=[]
    t_accuracy_list=[]
    for min in min_leaf:
        print("\nmin_samples_leaf =", min)
        forest, v_accuracy, t_accuracy = slr_forest(features, df_slr_rcp85, yr, max_feat=max_feat, max_d=max_d,
                                                    min_samp_leaf=min, max_samp=max_samp)
        v_accuracy_list.append(v_accuracy)
        t_accuracy_list.append(t_accuracy)
    plt.scatter(min_leaf, v_accuracy_list, label="Validation Accuracy")
    plt.scatter(min_leaf, t_accuracy_list, label="Training Accuracy")
    plt.legend()
    plt.xticks(min_leaf)
    plt.xlabel("min_samples_leaf")
    plt.ylabel("accuracy")
    title = "Accuracy vs min_samples_leaf for SLR RCP8.5 2100 \n(max_depth = " + str(max_d) + ", max_features = " + \
            str(max_feat) + ", max_samples = " + str(max_samp) + ")"
    plt.title(title)
    plt.show()


def max_samples(param_samples_df, slr_df, max_feat="auto", max_d=None, min_samp_leaf=1, print_threshold=True):
    slr_rcp85_classify = classify_data(slr_df, print_threshold=print_threshold)
    df_slr_rcp85 = param_samples_df.join(slr_rcp85_classify, how="outer")
    df_slr_rcp85 = df_slr_rcp85.dropna()
    features = param_samples_df.columns.tolist()
    yr = "2100"

    max_samples = [1000, 2000, 3000, 4000, 5000, 6000]
    v_accuracy_list=[]
    t_accuracy_list=[]
    for max in max_samples:
        print("\nmax_samples =", max)
        forest, v_accuracy, t_accuracy = slr_forest(features, df_slr_rcp85, yr, max_feat=max_feat, max_d=max_d,
                                                    min_samp_leaf=min_samp_leaf, max_samp=max)
        v_accuracy_list.append(v_accuracy)
        t_accuracy_list.append(t_accuracy)
    plt.scatter(max_samples, v_accuracy_list, label="Validation Accuracy")
    plt.scatter(max_samples, t_accuracy_list, label="Training Accuracy")
    plt.legend()
    plt.xticks(max_samples)
    plt.xlabel("max_samples")
    plt.ylabel("accuracy")
    title = "Accuracy vs max_samples for SLR RCP8.5 2100 \n(max_depth = " + str(max_d) + ", min_samples_leaf = " + \
            str(min_samp_leaf) + ", max_features = " + str(max_feat) + ")"
    plt.title(title)
    plt.show()


def parameter_loop_max_features():
    df = pd.read_csv("C:/Users/hough/Documents/research/climate-research/data/new_csv/RData_parameters_sample.csv")
    slr_rcp85 = pd.read_csv("C:/Users/hough/Documents/research/climate-research/data/new_csv/slr_rcp85.csv")
    stop = 1
    counter = 1
    max_depth_val = None
    max_samples_val = None
    while stop != 0:
        print("Iteration", counter)
        max_features(df, slr_rcp85, max_d=max_depth_val, max_samp=max_samples_val, print_threshold=False)
        max_features_val = int(input("Max features: "))
        max_depth(df, slr_rcp85, max_feat=max_features_val, max_samp=max_samples_val, print_threshold=False)
        max_depth_val = int(input("Max depth: "))
        max_samples(df, slr_rcp85, max_feat=max_features_val, max_d=max_depth_val, print_threshold=False)
        max_samples_val = int(input("Max samples: "))
        print("Iteration", counter, "Summary")
        print("\tMax features =", max_features_val)
        print("\tMax depth = ", max_depth_val)
        print("\tMax samples =", max_samples_val)
        stop = int(input("Enter 0 to stop"))
        counter += 1


def parameter_loop_min_samples_leaf():
    df = pd.read_csv("C:/Users/hough/Documents/research/climate-research/data/new_csv/RData_parameters_sample.csv")
    slr_rcp85 = pd.read_csv("C:/Users/hough/Documents/research/climate-research/data/new_csv/slr_rcp85.csv")
    stop = 1
    counter = 1
    min_samples_leaf_val = 1
    max_samples_val = None
    while stop != 0:
        print("Iteration", counter)
        max_features(df, slr_rcp85, min_samp_leaf=min_samples_leaf_val, max_samp=max_samples_val, print_threshold=False)
        max_features_val = int(input("Max features: "))
        min_samples_leaf(df, slr_rcp85, max_feat=max_features_val, max_samp=max_samples_val, print_threshold=False)
        min_samples_leaf_val = int(input("Min samples leaf: "))
        max_samples(df, slr_rcp85, max_feat=max_features_val, min_samp_leaf=min_samples_leaf_val, print_threshold=False)
        max_samples_val = int(input("Max samples: "))
        print("Iteration", counter, "Summary")
        print("\tMax features =", max_features_val)
        print("\tMin samples leaf = ", min_samples_leaf_val)
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


def slr_stacked_importances_plot(param_sample_df, slr_rcp26_5step, slr_rcp85_5step, importance_threshold):

    years = slr_rcp85_5step.columns.tolist()
    features = param_sample_df.columns.tolist()
    name = ["RCP2.6", "RCP8.5"]

    slr_rcp26_classify = classify_data(slr_rcp26_5step)
    slr_rcp85_classify = classify_data(slr_rcp85_5step)
    df_slr_rcp26 = param_sample_df.join(slr_rcp26_classify, how="outer")
    df_slr_rcp85 = param_sample_df.join(slr_rcp85_classify, how="outer")
    dflist=[df_slr_rcp26, df_slr_rcp85]

    #set color for each feature
    color_dict = feature_color_dict(param_sample_df)

    for i in range(len(dflist)):
        responsedf = dflist[i]
        responsedf = responsedf.dropna()
        importances_info = {}
        for j in range(len(years)):
            yr = years[j]

            # make forest
            forest, validation_acc, training_acc = slr_forest(features, responsedf, yr, max_d=MAX_DEPTH,
                                                              max_feat=MAX_FEATURES, max_samples= MAX_SAMPLES)
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

    df = pd.read_csv("C:/Users/hough/Documents/research/climate-research/data/new_csv/RData_parameters_sample.csv")
    slr_rcp85 = pd.read_csv("C:/Users/hough/Documents/research/climate-research/data/new_csv/slr_rcp85.csv")
    #make_tree_and_export(df, slr_rcp85, ["2025", "2100"], "rcp85", "./forests/", "./forests/forest_accuracy/")

    #min_samples_leaf(df, slr_rcp85, max_feat=MAX_FEATURES, max_d=MAX_DEPTH, max_samp=MAX_SAMPLES)

    #max_depth(df, slr_rcp85, min_samp_leaf=4, max_feat=20, max_samp=2000)

    forest_2025 = joblib.load(
        "C:/Users/hough/Documents/research/climate-research/python_code/forests/rcp85_2025.joblib")
    forest_2100 = joblib.load(
        "C:/Users/hough/Documents/research/climate-research/python_code/forests/rcp85_2100.joblib")
    path = "C:/Users/hough/Documents/research/climate-research/data/new_csv/SLR_splits/classification_forest/"
    tree_splits(df, "SLR", "RCP 8.5", [forest_2025, forest_2100], [2025, 2100], path)