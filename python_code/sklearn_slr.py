from sklearn import tree, metrics, ensemble
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import math

def find_forest_splits(forest, feature_names, feature, firstsplit=False):
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


def slr_tree():
    df = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/RData_parameters_sample.csv")
    slr_rcp26 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/slr_rcp26.csv")
    slr_rcp85 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/slr_rcp85.csv")
    Tgav_rcp26 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/Tgav_rcp26.csv")
    Tgav_rcp85 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/Tgav_rcp85.csv")

    df_slr_rcp26 = df.join(slr_rcp26, how="outer")
    df_slr_rcp85 = df.join(slr_rcp85, how="outer")
    df_Tgav_rcp26 = df.join(Tgav_rcp26, how="outer")
    df_Tgav_rcp85 = df.join(Tgav_rcp85, how="outer")

    #set up subsets
    features= df.columns.tolist()
    x= df_slr_rcp85[features]
    y= df_slr_rcp85["2075"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    #making the tree
    slr_rcp85_tree = tree.DecisionTreeRegressor(splitter="best", max_depth=3)
    slr_rcp85_tree = slr_rcp85_tree.fit(x_train, y_train)

    # printing tree
    text = tree.export_text(slr_rcp85_tree, feature_names=features)
    print(text)

    #graphic of the tree
    plt.figure(figsize=(15, 10))
    tree.plot_tree(slr_rcp85_tree, feature_names=features, filled=True)
    plt.show()

    # validation
    y_predicted = slr_rcp85_tree.predict(x_test)
    mse = metrics.mean_squared_error(y_test, y_predicted)
    print("Decision Tree Mean Square Error =", mse)


def slr_forest(feature_list, x_train, x_test, x_valid, y_valid, y_train, y_test, split_feature, print_forest= False):
    maxfeatures=[math.sqrt(38), 10, 15, 20, 25, 30, 35]
    for max in maxfeatures:
        # random forest creation
        forest = ensemble.RandomForestRegressor(n_estimators=1000, max_depth=4, max_features=max)
        forest = forest.fit(x_train, y_train)

        # print random forest
        if print_forest == True:
            for i in range(0, len(forest.estimators_)):
                print("\nEstimator", i, ":")
                text = tree.export_text(forest.estimators_[i], feature_names=feature_list)
                print(text)

        # find forest splits
        split_list = find_forest_splits(forest, feature_list, split_feature)
        split_df, all=split_stats(split_list, split_feature)
        first_split_list= find_forest_splits(forest, feature_list, split_feature, firstsplit=True)
        first_only=split_stats(first_split_list, split_feature)[1]

        # random forest validation
        y_predicted = forest.predict(x_valid)
        mse = metrics.mean_squared_error(y_valid, y_predicted)
        print("\nValidation Mean Square Error =", mse)

        # random forest test
        y_predicted = forest.predict(x_test)
        mse = metrics.mean_squared_error(y_test, y_predicted)
        print("\nTest Mean Square Error =", mse)

    return split_df, all, first_only, forest


def tree_splits(response):
    df = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/RData_parameters_sample.csv")
    slr_rcp26 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/slr_rcp26.csv")
    slr_rcp85 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/slr_rcp85.csv")
    Tgav_rcp26 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/Tgav_rcp26.csv")
    Tgav_rcp85 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/Tgav_rcp85.csv")

    df_slr_rcp26 = df.join(slr_rcp26, how="outer")
    df_slr_rcp85 = df.join(slr_rcp85, how="outer")
    df_Tgav_rcp26 = df.join(Tgav_rcp26, how="outer")
    df_Tgav_rcp85 = df.join(Tgav_rcp85, how="outer")

    #features = df.columns.tolist()
    years=["2025", "2050", "2075", "2100"]
    name=["RCP2.6", "RCP8.5"]
    if response == "SLR":
        dflist = [df_slr_rcp26, df_slr_rcp85]
        path = [[r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\rcp26_2025_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\rcp26_2050_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\rcp26_2075_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\rcp26_2100_splits_d4.csv'],
                [r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\rcp85_2025_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\rcp85_2050_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\rcp85_2075_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\rcp85_2100_splits_d4.csv']]
        table_path= [[r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\rcp26_2025_split_table_d4.csv',
                      r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\rcp26_2050_split_table_d4.csv',
                      r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\rcp26_2075_split_table_d4.csv',
                      r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\rcp26_2100_split_table_d4.csv'],
                     [r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\rcp85_2025_split_table_d4.csv',
                      r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\rcp85_2050_split_table_d4.csv',
                      r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\rcp85_2075_split_table_d4.csv',
                      r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\rcp85_2100_split_table_d4.csv']]
    elif response == "Tgav":
        dflist = [df_Tgav_rcp26, df_Tgav_rcp85]
        path = [[r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\rcp26_2025_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\rcp26_2050_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\rcp26_2075_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\rcp26_2100_splits_d4.csv'],
                [r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\rcp85_2025_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\rcp85_2050_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\rcp85_2075_splits_d4.csv',
                 r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\rcp85_2100_splits_d4.csv']]
        table_path = [[r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\rcp26_2025_split_table_d4.csv',
                       r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\rcp26_2050_split_table_d4.csv',
                       r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\rcp26_2075_split_table_d4.csv',
                       r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\rcp26_2100_split_table_d4.csv'],
                      [r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\rcp85_2025_split_table_d4.csv',
                       r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\rcp85_2050_split_table_d4.csv',
                       r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\rcp85_2075_split_table_d4.csv',
                       r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\rcp85_2100_split_table_d4.csv']]
    first_quartile_data=[]
    all_quartile_data=[]

    for i in range(len(dflist)):
        if name[i] == "RCP2.6":
            continue
        responsedf = dflist[i]
        responsedf = responsedf.dropna()
        importances_info = []
        fig, axs = plt.subplots(1, 4)
        for j in range(len(years)):
            features = df.columns.tolist()
            yr = years[j]
            if yr != "2100":
                continue
            # set up subsets
            x = responsedf[features]
            y = responsedf[yr]
            x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.4)    #train= 60%, validation + test= 40%
            #split up rest of 40% into validation & test
            x_validation, x_test, y_validation, y_test= train_test_split(x_rest, y_rest, test_size= .5) #validation= 20%, test= 20%

            all_label = name[i] + yr + " all splits"
            first_label = name[i] + yr + " first split"
            all = (all_label,)
            first = (first_label,)
            split_df, all_quartiles, first_quartiles, forest = slr_forest(features, x_train, x_test, x_validation,
                                                                          y_validation, y_train, y_test, "S.temperature")
            #split_df.to_csv(path[i][j], index=False)
            for n in range(len(all_quartiles)):
                all += (all_quartiles[n],)
                first += (first_quartiles[n],)
            first_quartile_data.append(first)
            all_quartile_data.append(all)

            table_df = splits_table(forest, features)
            #table_df.to_csv(table_path[i][j], index=True)

            # importances plot
            importances = forest.feature_importances_
            std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                         axis=0)
            indices = np.argsort(importances)[::-1]
            importances_list = []
            std_list = []
            features_list = []
            for idx in indices:
                if importances[idx] > .01:
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
            axs[j].set_ylim(bottom=0.0)
        main_title = response + " " + name[i] + " Feature Importances"
        fig.suptitle(main_title, fontsize=15)
        fig.text(0.52, 0.04, 'Features', ha='center', fontsize=12)
        fig.text(0.04, 0.5, 'Relative Importance', va='center', rotation='vertical', fontsize=12)
        plt.show()

    df_first_quartile= pd.DataFrame(first_quartile_data, columns=["Name", "0%", "25%", "50%", "75%", "100%", "Mean"])
    df_all_quartile = pd.DataFrame(all_quartile_data, columns=["Name", "0%", "25%", "50%", "75%", "100%", "Mean"])
    return df_first_quartile, df_all_quartile


def slr_output():
    slr_first_depth4_quartile, slr_all_depth4_quartile = tree_splits("SLR")
    slr_first_depth4_quartile.to_csv(r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\first_split_stats.csv',
                                     index=False)
    slr_all_depth4_quartile.to_csv(r'C:\Users\hough\Documents\research\data\new_csv\SLR_splits\all_split_stats.csv',
                                   index=False)


def Tgav_output():
    Tgav_first_depth4_quartile, Tgav_all_depth4_quartile = tree_splits("Tgav")
    Tgav_first_depth4_quartile.to_csv(
        r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\first_split_stats.csv',
        index=False)
    Tgav_all_depth4_quartile.to_csv(r'C:\Users\hough\Documents\research\data\new_csv\Tgav_splits\all_split_stats.csv',
                                    index=False)


def main():
    #slr_output()
    #Tgav_output()


if __name__ == '__main__':
    main()