from sklearn import tree, metrics, ensemble
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from hypopt import GridSearch
import pylab
import joblib
import os
from collections import Counter
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from pprint import pprint


# hyperparameters (via gridsearch)
MAX_DEPTH = 15
MAX_FEATURES = 24
N_ESTIMATORS = 100
MIN_IMPURITY_DECREASE = 0.001

PARAMETER_DICT = {'S.temperature': "ECS",
                  'diff.temperature': r"$\kappa_{DOECLIM}$",
                  'alpha.temperature': r"$\alpha_{DOECLIM}$",
                  'beta0_gsic.slr_brick': r"$\beta_0$",
                  'V0_gsic.slr_brick': r"$V_{0, GSIC}$",
                  'n_gsic.slr_brick': r"$n$",
                  'Gs0_gsic.slr_brick': r"$G_{s, 0}$",
                  'a_tee.slr_brick': r"$\alpha_{TE}$",
                  'a_simple.slr_brick': r"$a_{SIMPLE}$",
                  'b_simple.slr_brick': r"$b_{SIMPLE}$",
                  'alpha_simple.slr_brick': r"$\alpha_{SIMPLE}$",
                  'beta_simple.slr_brick': r"$\beta_{SIMPLE}$",
                  'V0_simple.slr_brick': r"$V_{0, SIMPLE}$",
                  'offset.Tgav_obs': r"$T_0$",
                  'sigma.Tgav_obs': r"$\sigma_T$",
                  'rho.Tgav_obs': r"$\rho_T$",
                  'sigma.slr_gsic_obs': r"$\sigma_{GSIC}$",
                  'rho.slr_gsic_obs': r"$\rho_{GSIC}$",
                  'sigma.slr_gis_obs': r"$\sigma_{GIS}$",
                  'rho.slr_gis_obs': r"$\rho_{GIS}$",
                  'a_anto.slr_brick': r"$a_{ANTO}$",
                  'b_anto.slr_brick': r"$b_{ANTO}$",
                  'gamma_dais.slr_brick': r"$\gamma$",
                  'alpha_dais.slr_brick': r"$\alpha_{DAIS}$",
                  'mu_dais.slr_brick': r"$\mu$",
                  'nu_dais.slr_brick': r"$\nu$",
                  'P0_dais.slr_brick': r"$P_0$",
                  'kappa_dais.slr_brick': r"$\kappa_{DAIS}$",
                  'f0_dais.slr_brick': r"$f_0$",
                  'h0_dais.slr_brick': r"$h_0$",
                  'c_dais.slr_brick': r"$C$",
                  'b0_dais.slr_brick': r"$b_0$",
                  'slope_dais.slr_brick': r"$slope$",
                  'lambda_dais.slr_brick': r"$\lambda$",
                  'Tcrit_dais.slr_brick': r"$T_{crit}$",
                  'offset.ocheat_obs': r"$H_0$",
                  'sigma.ocheat_obs': r"$\sigma_H$",
                  'rho.ocheat_obs': r"$\rho_H$"}

MODEL_DICT = {"Climate": ['S.temperature', 'diff.temperature', 'alpha.temperature', 'offset.Tgav_obs',
                          'sigma.Tgav_obs', 'rho.Tgav_obs', 'offset.ocheat_obs', 'sigma.ocheat_obs',
                          'rho.ocheat_obs'],
              "\nGlaciers &\nIce Caps": ['beta0_gsic.slr_brick', 'V0_gsic.slr_brick', 'n_gsic.slr_brick',
                                      'Gs0_gsic.slr_brick', 'sigma.slr_gsic_obs', 'rho.slr_gsic_obs'],
              "\nThermal\nExpansion": ['a_tee.slr_brick'],
              "\nGreenland\nIce Sheet": ['sigma.slr_gis_obs', 'rho.slr_gis_obs', 'a_simple.slr_brick',
                                      'b_simple.slr_brick', 'alpha_simple.slr_brick', 'beta_simple.slr_brick',
                                      'V0_simple.slr_brick'],
              "\nAntarctic\nIce Sheet": ['a_anto.slr_brick', 'b_anto.slr_brick', 'gamma_dais.slr_brick',
                                      'alpha_dais.slr_brick', 'mu_dais.slr_brick', 'nu_dais.slr_brick',
                                      'P0_dais.slr_brick', 'kappa_dais.slr_brick', 'f0_dais.slr_brick',
                                      'h0_dais.slr_brick', 'c_dais.slr_brick', 'b0_dais.slr_brick',
                                      'slope_dais.slr_brick', 'lambda_dais.slr_brick', 'Tcrit_dais.slr_brick']}


def get_previous_splits(forest, feature_names, rcp, year, folder_path):
    """
    Finds the splits that occurred leading up to the S.temperature split with the highest split value.  The previous
    splits for each estimator in the forest are collected and exported into a CSV.  The rows indices are the estimators
    and the columns are the depth in the estimator where the split occurred.
    :param forest: a forest that has already been fit with training data
    :param feature_names: list of all the feature names from the parameter sample dataframe used to fit the forest
    :param rcp: RCP name as a string (ex: "RCP 8.5")
    :param year: the year of the data used to make the forest
    :param folder_path: path to the folder where the CSV file will be saved
    :return:
    """
    features = feature_names.copy()
    all_prev_splits_list = []
    all_prev_split_features = []
    for i in range(len(forest.estimators_)):
        estimator = forest.estimators_[i]
        tree_feature = estimator.tree_.feature
        feature_new = []
        for node in tree_feature:
            if node == -2:
                feature_new.append('leaf')
            else:
                feature_new.append(features[node])
        dict = {"p": (feature_new[0], estimator.tree_.threshold[0])}
        split = "l"
        for i in range(1, len(feature_new)):
            dict[split] = (feature_new[i], estimator.tree_.threshold[i])
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

        max_Stemp = (None, 0)   # (node id, Stemp split value)
        for node_id in dict:
            if dict[node_id][0] == "S.temperature" and dict[node_id][1] > max_Stemp[1]:
                max_Stemp = (node_id, dict[node_id][1])

        prev_split = max_Stemp[0][:-1]
        prev_splits_list = []
        while len(prev_split) > 0:
            prev_splits_list.append(dict[prev_split])
            all_prev_split_features.append(dict[prev_split][0])
            prev_split = prev_split[:-1]
        prev_splits_list.append(dict["p"])
        prev_splits_list.reverse()
        all_prev_splits_list.append(prev_splits_list)

    index = pd.MultiIndex.from_tuples(all_prev_splits_list)
    prev_split_df = index.to_frame(index=False)
    prev_split_df.index.name = "Estimator"
    prev_split_df.columns = ["Depth %s" % i for i in range(len(prev_split_df.columns))]

    rcp_no_space = rcp.replace(" ", "")
    rcp_no_space_no_period = rcp_no_space.replace(".", "")
    file_path = folder_path + rcp_no_space_no_period + "_" + str(year) + "_splits_before_Stemp_max_split.csv"
    prev_split_df.to_csv(file_path)

    prev_split_feature_summary = Counter(all_prev_split_features)
    tot_splits = sum(prev_split_feature_summary.values())
    for feature in prev_split_feature_summary:
        prev_split_feature_summary[feature] /= tot_splits

    prev_feature_summary_df = pd.DataFrame(prev_split_feature_summary, index=[0])
    file_path = folder_path + rcp_no_space_no_period + "_" + str(year) + "_splits_before_Stemp_max_split_summary.csv"
    prev_feature_summary_df.to_csv(file_path, index=False)



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

    # node list
    nodes = ["p"]
    split = "l"
    last_split = "r" * MAX_DEPTH
    while True:
        nodes.append(split)
        if split == last_split:
            break
        if len(split) < MAX_DEPTH:
            split += "l"
        elif len(split) == MAX_DEPTH and len(nodes[-2]) == MAX_DEPTH:
            split = split[:-2] + "r"
        else:
            split = split[:-1] + "r"

        while split in nodes:
            split = split[:-2] + "r"

    row_list = []
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


def slr_forest(x_train, y_train, x_validation, y_validation, x_test, y_test, year, rcp, feature_list,
               max_feat="auto", max_d=None, n_estimators=100, min_impurity_decrease=0, print_forest=False):
    """
    Creates a forest using the desired parameters and fits the forest with 60% of the data in df
    :param feature_list: list of the input column names as strings
    :param df: dataframe that contains both the input data and the output data, NOT already split into training,
    validation, and testing subsets
    :param year: string of the year to use as the output data
    :param max_feat: integer number of features to consider when determining the best split -- default = "auto" which
    takes the square root of the total number of features
    :param max_d: the maximum depth of the tree as an integer -- default = None
    :param min_samp_leaf: the minimum integer number of samples required to be in a leaf node -- default = 1
    :param n_estimators: the number of trees in the forest as an integer -- default = 100
    :param min_samples_split: the minimum integer number of samples required in a node to be able to split --
    default = 2
    :param print_forest: boolean that controls whether the trees in the forest should be printed out in text form
    :return: forest, v_accuracy, t_accuracy -- forest is the forest that was created and fit with the data
                                            -- v_accuracy is the validation accuracy as a decimal value
                                            -- t_accuracy is the training accuracy as a decimal value
    """
    # random forest creation
    forest = ensemble.RandomForestClassifier(n_estimators=n_estimators, criterion="entropy", max_features=max_feat,
                                             max_depth=max_d, min_impurity_decrease=min_impurity_decrease)
    forest = forest.fit(x_train, y_train)

    # print random forest
    if print_forest:
        for i in range(0, len(forest.estimators_)):
            print("\nEstimator", i, ":")
            text = tree.export_text(forest.estimators_[i], feature_names=feature_list)
            print(text)

    performance_dict = {}

    # random forest training
    y_predicted = forest.predict(x_train)
    performance_dict["train_accuracy"] = metrics.accuracy_score(y_train, y_predicted)
    print("Training Accuracy =", performance_dict["train_accuracy"])
    performance_dict["train_tn"], performance_dict["train_fp"], performance_dict["train_fn"], \
    performance_dict["train_tp"] = metrics.confusion_matrix(y_train, y_predicted).ravel()
    print("Training true positive =", performance_dict["train_tp"],
          "\tTraining true negative =", performance_dict["train_tn"],
          "\tTraining false positive =", performance_dict["train_fp"],
          "\tTraining false negative =", performance_dict["train_fn"])

    # random forest validation
    y_predicted = forest.predict(x_validation)
    performance_dict["val_accuracy"] = metrics.accuracy_score(y_validation, y_predicted)
    print("Validation Accuracy =", performance_dict["val_accuracy"])
    performance_dict["val_tn"], performance_dict["val_fp"], performance_dict["val_fn"], \
    performance_dict["val_tp"] = metrics.confusion_matrix(y_validation, y_predicted).ravel()
    print("Validation true positive =", performance_dict["val_tp"],
          "\tValidation true negative =", performance_dict["val_tn"],
          "\tValidation false positive =", performance_dict["val_fp"],
          "\tValidation false negative =", performance_dict["val_fn"])

    # random forest test
    y_predicted = forest.predict(x_test)
    performance_dict["test_accuracy"] = metrics.accuracy_score(y_test, y_predicted)
    print("Test Accuracy =", performance_dict["test_accuracy"])
    performance_dict["test_tn"], performance_dict["test_fp"], performance_dict["test_fn"], \
    performance_dict["test_tp"] = metrics.confusion_matrix(y_test, y_predicted).ravel()
    print("Test true positive =", performance_dict["test_tp"],
          "\tTest true negative =", performance_dict["test_tn"],
          "\tTest false positive =", performance_dict["test_fp"],
          "\tTest false negative =", performance_dict["test_fn"])

    for measure in performance_dict:
        performance_dict[measure] = [performance_dict[measure]]

    return forest, performance_dict


def make_forest_and_export(yrs_to_output, rcp_str, forest_path, performance_path):
    """
    Creates forests for the given years, saves each forest as a file, and saves the validation and training accuracy
    of each forest in a CSV file
    :param parameter_sample_df: dataframe of the input feature values
    :param slr_df: dataframe of the output year values
    :param yrs_to_output: list of the years as strings to create and export forests for
    :param rcp_str: RCP name as a string with no spaces (ex: "rcp85")
    :param forest_path: path of the folder to save the forests into (ex: "./forests/")
    :param performance_path: path of the folder to save the performance measurements CSV files into
    (ex: "./forests/forest_accuracy/")
    :return: None
    """
    for yr in yrs_to_output:
        data_path = "../data/new_csv/preprocessed_data/" + rcp_str + "_" + yr
        x_train = pd.read_csv(data_path + "_Xtrain.csv")
        y_train = pd.read_csv(data_path + "_ytrain.csv", header=None)
        x_validation = pd.read_csv(data_path + "_Xvalidation.csv")
        y_validation = pd.read_csv(data_path + "_yvalidation.csv", header=None)
        x_test = pd.read_csv(data_path + "_Xtest.csv")
        y_test = pd.read_csv(data_path + "_ytest.csv", header=None)

        features = x_train.columns.tolist()

        print("\n", rcp_str, " ", yr, sep="")
        forest, performance_dict = slr_forest(x_train, y_train, x_validation, y_validation, x_test, y_test,
                                              year=yr, rcp=rcp_str, feature_list=features, max_feat=MAX_FEATURES,
                                              max_d=MAX_DEPTH, min_impurity_decrease=MIN_IMPURITY_DECREASE,
                                              n_estimators=N_ESTIMATORS)

        forest_file_path = forest_path + rcp_str + "_" + yr + ".joblib"
        joblib.dump(forest, forest_file_path, compress=3)
        print(f"Compressed Random Forest: {np.round(os.path.getsize(forest_file_path) / 1024 / 1024, 2)} MB")
        performance_df = pd.DataFrame(performance_dict)
        performance_file_path = performance_path + rcp_str + "_" + yr + "_performance_measures.csv"
        performance_df.to_csv(performance_file_path, index=False)


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


def tree_splits(param_sample_df, response, rcp, forests_list, year_list, folder_path, show_plot=False):
    """
    Runs the perform_splits() function and runs the splits_table() function for each forest, and creates a plot of the
    feature importances of each forest in the same pop-up.  The S.temperature split values are saved into a separate
    CSV for each forest, the split breakdown from the splits_table() function are saved into a separate CSV for each
    forest, the statistics of all the S.temperature splits for each forest/year are saved into one CSV, and the
    statistics of the first split value of the S.temeprature splits for each forest/year are saved into one CSV.
    :param param_sample_df: dataframe of the input feature values
    :param response: "SLR" or "Tgav"
    :param rcp: RCP name as a string (ex: "RCP 8.5")
    :param forests_list: a list of already fit forests for this function to be perfomred on
    :param year_list: list of the years (as integers) that correspond to years of the forests in forest_list
    :param folder_path: path to the folder where the CSV files will be saved
    :return: None
    """
    fig, axs = plt.subplots(1, len(year_list), squeeze=False)
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

       # table_df = splits_table(forest, features)
       # split_table_file_path = folder_path + rcp_no_space_no_period + "_" + yr + "_split_table.csv"
       # table_df.to_csv(split_table_file_path, index=True)

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
        axs[0, j].bar(range(len(importances_list)), importances_list, color="tab:blue",
                       yerr=std_list, align="center")
        title = yr
        axs[0, j].set_title(title)
        axs[0, j].set_xticks(range(len(importances_list)))
        axs[0, j].set_xticklabels(features_list, rotation=90)
        axs[0, j].set_ylim(top=1.0)
        axs[0, j].set_ylim(bottom= 0.0)
    main_title = response + " " + rcp + " Feature Importances"
    fig.suptitle(main_title, fontsize=15)
    fig.text(0.52, 0.04, 'Features', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Relative Importance', va='center', rotation='vertical', fontsize=12)
    if show_plot is True:
        plt.show()

    df_first_quartile = pd.DataFrame(first_quartile_data, columns=["Name", "0%", "25%", "50%", "75%", "100%", "Mean"])
    first_file_path = folder_path + rcp_no_space_no_period + "_first_splits.csv"
    df_first_quartile.to_csv(first_file_path, index=False)

    df_all_quartile = pd.DataFrame(all_quartile_data, columns=["Name", "0%", "25%", "50%", "75%", "100%", "Mean"])
    all_file_path = folder_path + rcp_no_space_no_period + "_all_splits.csv"
    df_all_quartile.to_csv(all_file_path, index=False)


def feature_color_dict(features_list):
    """
    Assigns each feature in the features_list a color and stores it in a dictionary
    :param features_list: list of feature names to associate colors with
    :return: color_dict -- dictionary where the keys are the features and the values are the colors
    """
    color_map = pylab.get_cmap('terrain')
    color_dict = {}
    for i in range(0, len(features_list)):
        feature = features_list[i]
        if feature == 'a_tee.slr_brick':
            color = 'black'
        elif feature == 'beta0_gsic.slr_brick':
            color = 'lightblue'
        else:
            color = color_map(i / len(features_list))
        color_dict[feature] = color
    return color_dict


def slr_stacked_importances_plot(param_sample_df, rcp26_forest_list, rcp85_forest_list, years,
                                 importance_threshold=.05):
    """
    Create a stacked histogram of the feature importances over time for each RCP
    :param param_sample_df: dataframe of the input feature values
    :param rcp26_forest_list: a list of already fit forests created from RCP 2.6 data
    :param rcp85_forest_list: a list of already fit forests created from RCP 8.5 data
    :param years: list of the years (as strings) that correspond to years of the forests in rcp26_forest_list and
    rcp85_forest_list
    :param importance_threshold: decimal value where every feature whose feature importance is under this threshold
    will be added into the "Other" category on the plot
    :return: None
    """
    features = param_sample_df.columns.tolist()
    name = ["RCP2.6", "RCP8.5"]
    forest_masterlist=[rcp26_forest_list, rcp85_forest_list]
    importances_info_list = []

    for i in range(len(forest_masterlist)):
        forest_list = forest_masterlist[i]
        importances_info = {}
        for j in range(len(forest_list)):
            forest = forest_list[j]

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
                            lst = [0 for n in range(0, j)]
                            lst.append(importance)
                            importances_info[feature] = lst
                        sum += importance
                importances_info['Other'].append(1 - sum)
                for f in importances_info:
                    if len(importances_info[f]) < (j + 1):
                        importances_info[f].append(0)
        importances_info_list.append(importances_info)

    # set color for each feature
    features_on_plot_ordered = []
    model_components_to_plot = {}
    idx = 0
    for model_component in MODEL_DICT:
        for importances_info in importances_info_list:
            for feature in importances_info:
                if feature == "Other":
                    pass
                elif feature not in features_on_plot_ordered and feature in MODEL_DICT[model_component]:
                    if model_component not in model_components_to_plot.values():
                        model_components_to_plot[idx] = model_component
                    features_on_plot_ordered.append(feature)
                    idx += 1
    color_dict = feature_color_dict(features_on_plot_ordered)

    # set up alternating hatching
    len_plot_features_even = len(features_on_plot_ordered) % 2 == 0     # length doesn't include "other" category
    hatch_dict = {}
    idx = 0
    for feature in features_on_plot_ordered:
        if len_plot_features_even:
            if idx % 2 == 0:
                hatch_dict[feature] = ".."
            else:
                hatch_dict[feature] = ""
        else:
            if idx % 2 == 1:
                hatch_dict[feature] = ".."
            else:
                hatch_dict[feature] = ""
        idx += 1

    # plotting
    fig, axs = plt.subplots(2, 1)
    handles = []
    labels = []
    order_plotted = []
    for i in range(len(importances_info_list)):
        importances_info = importances_info_list[i]
        # stacked importances plot
        x = np.arange(len(years))
        bottom = None
        for feature in features_on_plot_ordered:
            if feature == "Other":
                pass
            elif feature in importances_info:
                if feature not in order_plotted:
                    order_plotted.append(feature)
                color = color_dict[feature]
                if bottom is None:
                    axs[i].bar(x, importances_info[feature], label=PARAMETER_DICT[feature], color=color,
                               hatch=hatch_dict[feature])
                    bottom = np.array(importances_info[feature])
                else:
                    axs[i].bar(x, importances_info[feature], bottom=bottom, label=PARAMETER_DICT[feature], color=color,
                               hatch=hatch_dict[feature])
                    bottom += np.array(importances_info[feature])
        percent_label = "Other (< " + str(round(importance_threshold*100, 1)) + " %)"
        axs[i].bar(x, importances_info["Other"], bottom=bottom, label=percent_label, color='white',
                   hatch='//')
        axs[i].set_ylabel("Relative Importances", fontsize=14)
        ticks_loc = axs[i].get_yticks().tolist()
        axs[i].yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        axs[i].set_yticklabels(["{:.1f}".format(x) for x in ticks_loc], fontsize=12)
        yrs_10=[]
        for yr in years:
            if int(yr) % 10 == 0:
                yrs_10.append(yr)
            else:
                yrs_10.append("")
        axs[i].set_xticks(x)
        half_default_bar_width = .4
        axs[i].set_xlim(left=x[0] - half_default_bar_width, right=x[-1] + half_default_bar_width)
        axs[i].set_xticklabels(yrs_10, fontsize=12)
        axs[i].set_xlabel('Year', fontsize=14)
        axs[i].set_ylim(bottom=0, top=1)
        if i == 0:
            title_label = "(a)"
        else:
            title_label = "(b)"
        title = title_label + " " + name[i] + " Feature Importances"
        axs[i].set_title(title, fontsize=18)
        legend_details = axs[i].get_legend_handles_labels()
        handles += legend_details[0]
        labels += legend_details[1]

    order_plotted.append(percent_label)
    features_other_ordered = features_on_plot_ordered.copy()
    features_other_ordered.append(percent_label)

    by_label = dict(zip(labels, handles))
    other = by_label.pop(percent_label)
    by_label[percent_label] = other

    reorder = []
    for val in features_other_ordered:
        idx = order_plotted.index(val)
        reorder.append(idx)
    label_values = list(by_label.values())
    label_values = [label_values[idx] for idx in reorder]
    label_keys = list(by_label.keys())
    label_keys = [label_keys[idx] for idx in reorder]
    adder = 0
    for idx in model_components_to_plot:
        label_values.insert(idx + adder, mpatches.Patch(color='none'))
        label_keys.insert(idx + adder,  model_components_to_plot[idx])
        adder += 1

    label_values.insert(-1, mpatches.Patch(color='none'))
    label_keys.insert(-1, "")

    plt.figlegend(handles=label_values, labels=label_keys, bbox_to_anchor=(.99, .97), fontsize=14)
    plt.subplots_adjust(left=.105, right=.815, top=.96, bottom=.065, hspace=.258)
    plt.show()


def Stemp_histograms(year_list, rcp, first_only=False):
    """
    Opens the saved CSV files of the S.temperature splits for each year in the year_list and creates histograms of
    the S.temperature splits in each tree for each year
    :param year_list: list of the years (string or int) for the dataframes in split_df_list
    :param rcp: RCP name as a string (ex: "RCP 8.5")
    :param first_only: boolean that controls whether to only plot the values of the first S.temperture split in
    the trees
    :return: None
    """
    rcp_no_space = rcp.replace(" ", "")
    rcp_no_space_no_period = rcp_no_space.replace(".", "")
    split_df_list = []
    for yr in year_list:
        file_path = "../data/new_csv/SLR_splits/classification_forest/" + rcp_no_space_no_period + "_" + str(yr) \
                    + "_splits.csv"
        df = pd.read_csv(file_path)
        split_df_list.append(df)

    split_list = []
    if first_only is True:
        for df in split_df_list:
            split_list.append(df["0"].dropna().values.tolist())
    else:
        for df in split_df_list:
            split_list.append(df.stack().tolist())

    fig, axs = plt.subplots(1, len(year_list), squeeze=False)
    i = 0
    bin_seq=np.arange(0, 10, step=.5)
    for i in range(len(split_list)):
        data = split_list[i]
        yr = year_list[i]
        axs[0, i].hist(data, bins=bin_seq, edgecolor='white')
        axs[0, i].set_title(yr)
        if first_only is True:
            axs[0, i].set_ylim(top=500)
        else:
            axs[0, i].set_ylim(top=1000)
        axs[0, i].set_ylim(bottom=0)
        axs[0, i].set_xlim(right=10)
        axs[0, i].set_xlim(left=1)
        i += 1
    main_title = "SLR " + rcp + " Histogram of S.temperature Split Values"
    if first_only is True:
        main_title += " (First Split Only)"
    fig.suptitle(main_title)
    fig.text(0.52, 0.04, 'S.temperature Split', ha='center')
    fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')
    plt.show()


def Stemp_max_split_histogram(year_list, rcp):
    """
    Opens the saved CSV files of the S.temperature splits for each year in the year_list and creates histograms of
    the highest S.temperature split in each tree for each year
    :param year_list: list of the years (string or int) for the dataframes in split_df_list
    :param rcp: RCP name as a string (ex: "RCP 8.5")
    :return: None
    """
    rcp_no_space = rcp.replace(" ", "")
    rcp_no_space_no_period = rcp_no_space.replace(".", "")
    df_dict={}
    for yr in year_list:
        file_path = "../data/new_csv/SLR_splits/classification_forest/" + rcp_no_space_no_period + "_" + str(yr) \
                    + "_splits.csv"
        df = pd.read_csv(file_path)
        max_list = df.max(axis=1).tolist()
        df_dict[str(yr)] = max_list

    fig, axs = plt.subplots(1, len(year_list), squeeze=False)
    i = 0
    bin_seq = np.arange(0, 10, step=.5)
    for yr in df_dict:
        axs[0, i].hist(df_dict[yr], bins=bin_seq, edgecolor='white')
        axs[0, i].set_title(yr)
        axs[0, i].set_ylim(bottom=0)
        axs[0, i].set_ylim(top=200)
        axs[0, i].set_xlim(right=10)
        axs[0, i].set_xlim(left=1)
        i += 1
    main_title = "SLR " + rcp + " Histogram of Highest S.temperature Split Values of each Tree"
    fig.suptitle(main_title)
    fig.text(0.52, 0.04, 'S.temperature Split', ha='center')
    fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')
    plt.show()


def all_Stemp_max_split_histograms(year_list, ECS_splits_folder_path="../data/new_csv/SLR_splits/classification_forest/",):
    """
    Opens the saved CSV files of the S.temperature splits for each year in the year_list for both RCPs and creates
    density histograms of the highest S.temperature split in each tree for each year.  The left plots are RCP 2.6 and
    the right plots are RCP 8.5
    :param year_list: list of the years (string or int) for the dataframes in split_df_list
    :return: None
    """
    rcp26_split_dict = {}
    rcp85_split_dict = {}
    for yr in year_list:
        rcp_26_file_path = ECS_splits_folder_path + "RCP26_" + str(yr) + "_splits.csv"
        rcp26_df = pd.read_csv(rcp_26_file_path)
        rcp26_max_list = rcp26_df.max(axis=1).dropna().tolist()
        rcp26_split_dict[str(yr)] = rcp26_max_list

        rcp_85_file_path = ECS_splits_folder_path + "RCP85_" + str(yr) + "_splits.csv"
        rcp85_df = pd.read_csv(rcp_85_file_path)
        rcp85_max_list = rcp85_df.max(axis=1).dropna().tolist()
        rcp85_split_dict[str(yr)] = rcp85_max_list
    all_split_lists = [rcp26_split_dict, rcp85_split_dict]

    fig, axs = plt.subplots(len(year_list), 2, squeeze=False)
    i = 0
    label_counter = 0
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", '(h)', '(i)', '(j)', '(k)', '(l)']
    bin_seq = np.arange(0, 10, step=.5)
    for yr in year_list:
        for j in range(2):
            axs[i, j].hist(all_split_lists[j][str(yr)], bins=bin_seq, edgecolor='white', density=True)
            axs[i, j].set_ylim(bottom=0)
            axs[i, j].set_ylim(top=1)
            axs[i, j].set_xlim(right=10)
            axs[i, j].set_xlim(left=3)
            axs[i, j].yaxis.set_visible(False)
            title = " " + labels[label_counter] + " " + str(yr)
            axs[i, j].set_title(title, loc='left', y=.75)
            label_counter += 1
        i += 1
    main_title = "SLR Histograms of Maximum ECS Split Values ($^\circ$C) of each Tree"
    fig.suptitle(main_title)
    fig.text(.39, .94, "RCP2.6", fontsize='large', ha='center')
    fig.text(.61, .94, "RCP8.5", fontsize='large', ha='center')
    fig.text(0.39, 0.015, 'Maximum ECS Split ($^\circ$C)', ha='center')
    fig.text(0.61, 0.015, 'Maximum ECS Split ($^\circ$C)', ha='center')
    fig.text(0.28, 0.5, 'Density', va='center', rotation='vertical')
    plt.subplots_adjust(left=.3, right=.7, wspace=.2, hspace=.3, top=.935, bottom=.055)
    plt.show()


def Stemp_boxplots(year_list, rcp, first_only=False, show_outliers=True):
    """
    Opens the saved CSV files of the S.temperature splits for each year in the year_list and creates a plot of
    boxplots of the S.temperature splits
    :param year_list: list of the years (string or int) for the dataframes in split_df_list
    :param rcp: RCP name as a string (ex: "RCP 8.5")
    :param first_only: boolean that controls whether to only plot the values of the first S.temperture split in
    the trees
    :param show_outliers: boolean that controls whether to show outliers on the plot
    :return: None
    """
    rcp_no_space = rcp.replace(" ", "")
    rcp_no_space_no_period = rcp_no_space.replace(".", "")
    split_df_list = []
    for yr in year_list:
        file_path = "../data/new_csv/SLR_splits/classification_forest/" + rcp_no_space_no_period + "_" + str(yr) \
                    + "_splits.csv"
        df = pd.read_csv(file_path)
        split_df_list.append(df)

    split_list = []
    if first_only is True:
        split_str = "(First Split Only)"
        for df in split_df_list:
            split_list.append(df["0"].dropna().values.tolist())
    else:
        split_str = ""
        for df in split_df_list:
            split_list.append(df.stack().tolist())
    fig, ax = plt.subplots()
    ax.boxplot(split_list, showfliers=show_outliers, patch_artist=True, medianprops=dict(color="black"),
               flierprops=dict(markeredgecolor='silver'), labels=[str(yr) for yr in year_list])
    title = "SLR " + rcp + " Boxplots of S.temperature Split Values " + split_str
    plt.ylabel("S.temperature Split Value")
    plt.xlabel("Year")
    plt.title(title, fontsize=15)
    plt.grid(b=True, axis='y', color='gray')
    plt.show()


def Stemp_max_split_boxplots(year_list, rcp, show_outliers=True):
    """
    Opens the saved CSV files of the S.temperature splits for each year in the year_list and creates boxplots of
    the highest S.temperature split in each tree for each year
    :param year_list: list of the years (string or int) for the dataframes in split_df_list
    :param rcp: RCP name as a string (ex: "RCP 8.5")
    :param show_outliers: boolean that controls whether to show outliers on the plot
    :return: None
    """
    rcp_no_space = rcp.replace(" ", "")
    rcp_no_space_no_period = rcp_no_space.replace(".", "")
    split_lists=[]
    for yr in year_list:
        file_path = "../data/new_csv/SLR_splits/classification_forest/" + rcp_no_space_no_period + "_" + str(yr) \
                    + "_splits.csv"
        df = pd.read_csv(file_path)
        max_list = df.max(axis=1).dropna().tolist()
        split_lists.append(max_list)

    fig, ax = plt.subplots()
    ax.boxplot(split_lists, showfliers=show_outliers, patch_artist=True, medianprops=dict(color="black"),
               flierprops=dict(markeredgecolor='silver'), labels=[str(yr) for yr in year_list])
    title = "SLR " + rcp + " Boxplots of Highest S.temperature Split Values of each Tree"
    plt.ylabel("S.temperature Split Value")
    plt.xlabel("Year")
    plt.title(title, fontsize=15)
    plt.grid(b=True, axis='y', color='gray')
    plt.show()


def all_Stemp_max_split_boxplots(year_list, show_outliers=True,
                                 ECS_splits_folder_path="../data/new_csv/SLR_splits/classification_forest/",
                                 print_medians=False, print_IQR=False, print_in_latex_table_format=False):
    """
    Opens the saved CSV files of the S.temperature splits for each year in the year_list for both RCPs and creates
    boxplots of the highest S.temperature split in each tree for each year.  The top panel of the plot is RCP 2.6 and
    the bottom panel is RCP 8.5
    :param year_list: list of the years (string or int) for the dataframes in split_df_list
    :param show_outliers: boolean that controls whether to show outliers on the plot
    :return: None
    """
    rcp26_split_lists = []
    rcp85_split_lists = []
    for yr in year_list:
        rcp_26_file_path = ECS_splits_folder_path + "RCP26_" + str(yr) + "_splits.csv"
        rcp26_df = pd.read_csv(rcp_26_file_path)
        rcp26_max_list = rcp26_df.max(axis=1).dropna().tolist()
        rcp26_split_lists.append(rcp26_max_list)

        rcp_85_file_path = ECS_splits_folder_path + "RCP85_" + str(yr) + "_splits.csv"
        rcp85_df = pd.read_csv(rcp_85_file_path)
        rcp85_max_list = rcp85_df.max(axis=1).dropna().tolist()
        rcp85_split_lists.append(rcp85_max_list)

        if print_in_latex_table_format and print_medians and print_IQR:
            print(yr, "%5.4f" % np.quantile(rcp26_max_list, .25), "%5.4f" % np.median(rcp26_max_list),
                  "%5.4f" % np.quantile(rcp26_max_list, .75),
                  "%5.4f" % (np.quantile(rcp26_max_list, .75) - np.quantile(rcp26_max_list, .25)),
                  "%5.4f" % np.quantile(rcp85_max_list, .25), "%5.4f" % np.median(rcp85_max_list),
                  "%5.4f" % np.quantile(rcp85_max_list, .75),
                  "%5.4f" % (np.quantile(rcp85_max_list, .75) - np.quantile(rcp85_max_list, .25)),
                  sep=" & ")
            print("\\\\")
        else:
            if print_medians:
                print("RCP2.6", yr, "\tmedian = %5.4f" % np.median(rcp26_max_list))
                print("RCP8.5", yr, "\tmedian = %5.4f" % np.median(rcp85_max_list))
            if print_IQR:
                print("RCP2.6", yr, "\tQ1 = %5.4f" % np.quantile(rcp26_max_list, .25),
                      "\tQ3 = %5.4f" % np.quantile(rcp26_max_list, .75),
                      "\tIQR = %5.4f" % (np.quantile(rcp26_max_list, .75) - np.quantile(rcp26_max_list, .25)))
                print("RCP8.5", yr, "\tQ1 = %5.4f" % np.quantile(rcp85_max_list, .25),
                      "\tQ3 = %5.4f" % np.quantile(rcp85_max_list, .75),
                      "\tIQR = %5.4f" % (np.quantile(rcp85_max_list, .75) - np.quantile(rcp85_max_list, .25)))


    all_split_lists = [rcp26_split_lists, rcp85_split_lists]

    fig, axs = plt.subplots(2, 1)
    for i in range(2):
        axs[i].boxplot(all_split_lists[i], showfliers=show_outliers, patch_artist=True, medianprops=dict(color="black"),
                   flierprops=dict(markeredgecolor='silver'))
        if i == 0:
            axs[i].xaxis.set_visible(False)
        elif i == 1:
            axs[i].set_xlabel("Year", fontsize=14)
            axs[i].set_xticklabels([str(yr) for yr in year_list], fontsize=12)
        axs[i].set_ylabel("Maximum ECS Split ($^\circ$C)", fontsize=14)
        axs[i].grid(b=True, axis='y', color='gray')
        axs[i].set_ylim(top=10)
        axs[i].set_ylim(bottom=1.5)
    axs[0].text(.7, 2.1, "(a) RCP2.6", fontsize=18)
    axs[1].text(.7, 2.1, "(b) RCP8.5", fontsize=18)
    plt.subplots_adjust(left=.25, right=.75, hspace=.02, top=.97, bottom=.055)
    plt.show()


def gridsearch(year, rcp):
    """
    Perform a gridsearch of the parameters used to create the forests, saves the best parameters to a CSV, and saves
    the cross validation information/result to another CSV.
    :param param_samples_df: dataframe of the input feature values
    :param slr_df: dataframe of the output year values
    :param year: string of the year for the data to use when making the forests in the gridsearch
    :param rcp: RCP name as a string (ex: "RCP 8.5")
    :param folder_path: path to the folder where the CSV files will be saved
    :return: None
    """
    rcp_no_space = rcp.replace(" ", "")
    rcp = rcp_no_space.replace(".", "")

    data_path = "../data/new_csv/preprocessed_data/" + rcp + "_" + str(yr)
    x_train = pd.read_csv(data_path + "_Xtrain.csv")
    y_train = pd.read_csv(data_path + "_ytrain.csv", header=None)
    x_validation = pd.read_csv(data_path + "_Xvalidation.csv")
    y_validation = pd.read_csv(data_path + "_yvalidation.csv", header=None)
    x_test = pd.read_csv(data_path + "_Xtest.csv")
    y_test = pd.read_csv(data_path + "_ytest.csv", header=None)


    revision_2_values_param_grid = {
        'n_estimators': [25, 50, 100, 150, 200, 250],
        'max_depth': [10, 15, 20, 25, 30],
        'max_features': [3, 6, 12, 24, 30, 38],
        'min_impurity_decrease': [0.001]
    }

    grid_search = GridSearch(model=ensemble.RandomForestClassifier(criterion="entropy"), param_grid=revision_2_values_param_grid)
    grid_search.fit(x_train, y_train, x_validation, y_validation)
    
    print("BEST PARAMETERS:")
    print(grid_search.best_params)
    
    print("Mean cross-validated score of the best_estimator: ", grid_search.best_score)
    best_params_df = pd.DataFrame(grid_search.best_params, index=[0])
    # ****CHANGE FILE NAME WHEN CHANGE PARAM_GRID****
    best_params_df.to_csv("./gridsearchcv_results/revisions_2_"+rcp+"-"+year+".csv",
                          index=False)
    score_dict = {}
    for nn in grid_search.param_grid.keys():
        score_dict[nn] = [grid_search.param_scores[k][0][nn] for k in range(len(grid_search.param_scores))]
    score_dict["score"] = [grid_search.param_scores[k][1] for k in range(len(grid_search.param_scores))]
    score_df = pd.DataFrame(score_dict)
    # ****CHANGE FILE NAME WHEN CHANGE PARAM_GRID****
    score_df.to_csv("./gridsearchcv_results/revisions_2_cv_results_"+rcp+"-"+year+".csv", index=False)

    # Compare with default model without hyperopt
    outofbox = ensemble.RandomForestClassifier(criterion="entropy",random_state=0)
    outofbox.fit(x_train, y_train)
    print('Out-of-box hyperparameters:', outofbox.score(x_test, y_test))
    print('Optimized harameters:', grid_search.score(x_test, y_test))


def load_forests(year_list, rcp_str):
    """
    Loads forests from a saved forest file into a list.
    :param year_list: list of years (string or int) to load forests for
    :param rcp_str: RCP name as a string with no spaces (ex: "rcp85")
    :return: a list of the loaded forests
    """
    forests = []
    for yr in year_list:
        path = "./forests/" + rcp_str + "_" + str(yr) + ".joblib"
        forests.append(joblib.load(path))
    return forests


if __name__ == '__main__':
    # saving forests
    df = pd.read_csv("../data/new_csv/RData_parameters_sample.csv")
    slr_rcp26_5step = pd.read_csv("../data/new_csv/slr_rcp26_5yrstep.csv")
    slr_rcp85_5step = pd.read_csv("../data/new_csv/slr_rcp85_5yrstep.csv")
    yrs_rcp26 = slr_rcp26_5step.columns.tolist()
    yrs_rcp85 = slr_rcp85_5step.columns.tolist()

    make_forest_and_export(yrs_rcp26, "rcp26", "./forests/revision_2_",
                           "./forests/forest_accuracy/revision_2_")
    make_forest_and_export(yrs_rcp85, "rcp85", "./forests/revision_2_",
                           "./forests/forest_accuracy/revision_2_")

    #make_forest_and_export(df, slr_rcp26_5step, yrs_rcp26, "rcp26", "./forests/", "./forests/forest_accuracy/")
    #make_forest_and_export(df, slr_rcp85_5step, yrs_rcp85, "rcp85", "./forests/", "./forests/forest_accuracy/")
    # make w/ 80th percentile
    #make_forest_and_export(df, slr_rcp26_5step, yrs_rcp26, "rcp26", "./forests/new_hyperparams_80th_percentile",
    #                       "./forests/forest_performance/new_hyperparams_80th_percentile",
    #                       classification_percentile=.8)
    #make_forest_and_export(df, slr_rcp85_5step, yrs_rcp85, "rcp85", "./forests/new_hyperparams_80th_percentile",
    #                       "./forests/forest_performance/new_hyperparams_80th_percentile",
    #                       classification_percentile=.8)

    # new hyperparams
    #make_forest_and_export(df, slr_rcp26_5step, yrs_rcp26, "rcp26", "./forests/new_hyperparams_10fold_",
    #                       "./forests/forest_performance/new_hyperparams_10fold_", save_train_test=True)
    #make_forest_and_export(df, slr_rcp85_5step, yrs_rcp85, "rcp85", "./forests/new_hyperparams_10fold_",
    #                       "./forests/forest_performance/new_hyperparams_10fold_", save_train_test=True)

    # making S.temp split csv's
    list_10_yrs = []
    for yr in range(2020, 2151, 10):
        list_10_yrs.append(yr)
    #rcp26_forest_list_10yrs = load_forests(list_10_yrs, "rcp26")
    #rcp85_forest_list_10yrs = load_forests(list_10_yrs, "rcp85")
    #rcp26_forest_list_10yrs = load_forests(list_10_yrs, "new_hyperparams_80th_percentilercp26") # path for 80th percentile forests
    #rcp85_forest_list_10yrs = load_forests(list_10_yrs, "new_hyperparams_80th_percentilercp85") # path for 80th percentile forests
    #path = "../data/new_csv/SLR_splits/classification_forest/"
    #path_80th_percent = "../data/new_csv/SLR_splits/classification_forest/new_hyperparams/new_hyperparams_80th_percentile"
    #rcp26_forest_list = load_forests(yrs_rcp26,
    #                                       "new_hyperparams_rcp26")  # path for new hyperparameters (post-review) forests
    #rcp85_forest_list = load_forests(yrs_rcp85,
    #                                       "new_hyperparams_rcp85")  # path for new hyperparameters (post-review) forests
    #new_hyperparams_path = "../data/new_csv/SLR_splits/classification_forest/new_hyperparams/new_hyperparams"  # path for new hyperparameters (post-review) forests
    #rcp26_forest_list = load_forests(yrs_rcp26, "new_hyperparams_10fold_rcp26")
    #rcp85_forest_list = load_forests(yrs_rcp85, "new_hyperparams_10fold_rcp85")
    #rcp26_forest_list = load_forests(yrs_rcp26,
    #                                       "new_hyperparams_80th_percentilercp26")  # path for 80th percentile forests
    #rcp85_forest_list = load_forests(yrs_rcp85,
    #                                       "new_hyperparams_80th_percentilercp85")  # path for 80th percentile forests
    #tree_splits(df, "SLR", "RCP 2.6", rcp26_forest_list, yrs_rcp26, path_80th_percent)
    #tree_splits(df, "SLR", "RCP 8.5", rcp85_forest_list, yrs_rcp85, path_80th_percent)

    # plots
    #rcp26_forest_list = load_forests(yrs_rcp26, "rcp26")
    #rcp85_forest_list = load_forests(yrs_rcp85, "rcp85")
    #slr_stacked_importances_plot(df, rcp26_forest_list, rcp85_forest_list, yrs_rcp26, importance_threshold=.04)
    #all_Stemp_max_split_boxplots(list_10_yrs, print_IQR=True, print_medians=True, print_in_latex_table_format=True)
    #all_Stemp_max_split_histograms([2025, 2050, 2075, 2100, 2125, 2150])

    #slr_stacked_importances_plot(df, rcp26_forest_list, rcp85_forest_list, yrs_rcp26, importance_threshold=.04)
    #all_Stemp_max_split_boxplots(list_10_yrs, ECS_splits_folder_path=new_hyperparams_path, print_IQR=True,
    #                             print_medians=True, print_in_latex_table_format=True)
    #all_Stemp_max_split_histograms([2025, 2050, 2075, 2100, 2125, 2150], ECS_splits_folder_path=new_hyperparams_path)

    # 80th percentile boxplot:
    #all_Stemp_max_split_boxplots(list_10_yrs, ECS_splits_folder_path=path_80th_percent)

    #forest_rcp85_2020 = rcp85_forest_list_10yrs[0]
    #features = df.columns.tolist()
    #get_previous_splits(forest_rcp85_2020, features, "RCP 8.5", 2020, path)


    # perform grid search
    #gridsearch("2100", "RCP 8.5")

