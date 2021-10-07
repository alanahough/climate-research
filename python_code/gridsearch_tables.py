import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter


def top_10_cv_tables(rcp26_2050, rcp26_2075, rcp26_2100, rcp85_2050, rcp85_2075, rcp85_2100, top_amount,
                     file_descriptor):
    rcp26_2050_top20 = rcp26_2050[rcp26_2050["rank_test_score"] <= top_amount].sort_values(by=['rank_test_score'])
    rcp26_2050_top20 = rcp26_2050_top20[["param_max_depth", "param_max_features", "param_min_samples_leaf",
                                         "param_min_samples_split", "param_n_estimators", "mean_test_score",
                                         "rank_test_score"]]
    rcp26_2050_top20.to_csv("../data/new_csv/hyperparameter_tuning/rcp26_2050_top" + str(top_amount) + "_" +
                            file_descriptor + ".csv", index=False)

    rcp26_2075_top20 = rcp26_2075[rcp26_2075["rank_test_score"] <= top_amount].sort_values(by=['rank_test_score'])
    rcp26_2075_top20 = rcp26_2075_top20[["param_max_depth", "param_max_features", "param_min_samples_leaf",
                                         "param_min_samples_split", "param_n_estimators", "mean_test_score",
                                         "rank_test_score"]]
    rcp26_2075_top20.to_csv("../data/new_csv/hyperparameter_tuning/rcp26_2075_top" + str(top_amount) + "_" +
                            file_descriptor + ".csv", index=False)

    rcp26_2100_top20 = rcp26_2100[rcp26_2100["rank_test_score"] <= top_amount].sort_values(by=['rank_test_score'])
    rcp26_2100_top20 = rcp26_2100_top20[["param_max_depth", "param_max_features", "param_min_samples_leaf",
                                         "param_min_samples_split", "param_n_estimators", "mean_test_score",
                                         "rank_test_score"]]
    rcp26_2100_top20.to_csv("../data/new_csv/hyperparameter_tuning/rcp26_2100_top" + str(top_amount) + "_" +
                            file_descriptor + ".csv", index=False)

    rcp85_2050_top20 = rcp85_2050[rcp85_2050["rank_test_score"] <= top_amount].sort_values(by=['rank_test_score'])
    rcp85_2050_top20 = rcp85_2050_top20[["param_max_depth", "param_max_features", "param_min_samples_leaf",
                                         "param_min_samples_split", "param_n_estimators", "mean_test_score",
                                         "rank_test_score"]]
    rcp85_2050_top20.to_csv("../data/new_csv/hyperparameter_tuning/rcp85_2050_top" + str(top_amount) + "_" +
                            file_descriptor + ".csv", index=False)

    rcp85_2075_top20 = rcp85_2075[rcp85_2075["rank_test_score"] <= top_amount].sort_values(by=['rank_test_score'])
    rcp85_2075_top20 = rcp85_2075_top20[["param_max_depth", "param_max_features", "param_min_samples_leaf",
                                         "param_min_samples_split", "param_n_estimators", "mean_test_score",
                                         "rank_test_score"]]
    rcp85_2075_top20.to_csv("../data/new_csv/hyperparameter_tuning/rcp85_2075_top" + str(top_amount) + "_" +
                            file_descriptor + ".csv", index=False)

    rcp85_2100_top20 = rcp85_2100[rcp85_2100["rank_test_score"] <= top_amount].sort_values(by=['rank_test_score'])
    rcp85_2100_top20 = rcp85_2100_top20[["param_max_depth", "param_max_features", "param_min_samples_leaf",
                                         "param_min_samples_split", "param_n_estimators", "mean_test_score",
                                         "rank_test_score"]]
    rcp85_2100_top20.to_csv("../data/new_csv/hyperparameter_tuning/rcp85_2100_top" + str(top_amount) + "_" +
                            file_descriptor + ".csv", index=False)


def plot_max_depth():
    rcp26_2050_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp26_2050_top20.csv")
    rcp26_2075_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp26_2075_top20.csv")
    rcp26_2100_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp26_2100_top20.csv")
    rcp85_2050_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp85_2050_top20.csv")
    rcp85_2075_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp85_2075_top20.csv")
    rcp85_2100_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp85_2100_top20.csv")

    plt.scatter(rcp26_2050_top20[["param_max_depth"]], rcp26_2050_top20[["mean_test_score"]], label="RCP 2.6\n2050",
                facecolors='none', edgecolors='blue')
    plt.scatter(rcp26_2075_top20[["param_max_depth"]], rcp26_2075_top20[["mean_test_score"]], label="RCP 2.6\n2075",
                facecolors='none', edgecolors='red')
    plt.scatter(rcp26_2100_top20[["param_max_depth"]], rcp26_2100_top20[["mean_test_score"]], label="RCP 2.6\n2100",
                facecolors='none', edgecolors='orange')
    plt.scatter(rcp85_2050_top20[["param_max_depth"]], rcp85_2050_top20[["mean_test_score"]], label="RCP 8.5\n2050",
                color='blue', marker='x')
    plt.scatter(rcp85_2075_top20[["param_max_depth"]], rcp85_2075_top20[["mean_test_score"]], label="RCP 8.5\n2075",
                color='red', marker='x')
    plt.scatter(rcp85_2100_top20[["param_max_depth"]], rcp85_2100_top20[["mean_test_score"]], label="RCP 8.5\n2050",
                color='orange', marker='x')
    plt.xlabel("max_depth")
    plt.ylabel("Mean cross-validation accuracy")
    plt.title("Mean cross-validation accuracy vs max_depth")
    plt.legend(bbox_to_anchor=(.995, 1.02))
    plt.subplots_adjust(right=.81)
    plt.show()


def plot_min_samples_leaf():
    rcp26_2050_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp26_2050_top20.csv")
    rcp26_2075_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp26_2075_top20.csv")
    rcp26_2100_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp26_2100_top20.csv")
    rcp85_2050_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp85_2050_top20.csv")
    rcp85_2075_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp85_2075_top20.csv")
    rcp85_2100_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp85_2100_top20.csv")

    plt.scatter(rcp26_2050_top20[["param_min_samples_leaf"]], rcp26_2050_top20[["mean_test_score"]],
                label="RCP 2.6\n2050", facecolors='none', edgecolors='blue')
    plt.scatter(rcp26_2075_top20[["param_min_samples_leaf"]], rcp26_2075_top20[["mean_test_score"]],
                label="RCP 2.6\n2075", facecolors='none', edgecolors='red')
    plt.scatter(rcp26_2100_top20[["param_min_samples_leaf"]], rcp26_2100_top20[["mean_test_score"]],
                label="RCP 2.6\n2100", facecolors='none', edgecolors='orange')
    plt.scatter(rcp85_2050_top20[["param_min_samples_leaf"]], rcp85_2050_top20[["mean_test_score"]],
                label="RCP 8.5\n2050", color='blue', marker='x')
    plt.scatter(rcp85_2075_top20[["param_min_samples_leaf"]], rcp85_2075_top20[["mean_test_score"]],
                label="RCP 8.5\n2075", color='red', marker='x')
    plt.scatter(rcp85_2100_top20[["param_min_samples_leaf"]], rcp85_2100_top20[["mean_test_score"]],
                label="RCP 8.5\n2050", color='orange', marker='x')
    plt.xlabel("min_samples_leaf")
    plt.ylabel("Mean cross-validation accuracy")
    plt.title("Mean cross-validation accuracy vs min_samples_leaf")
    plt.legend(bbox_to_anchor=(1.265, 1.02))
    plt.subplots_adjust(right=.81)
    plt.show()


def plot_min_samples_split():
    rcp26_2050_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp26_2050_top20.csv")
    rcp26_2075_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp26_2075_top20.csv")
    rcp26_2100_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp26_2100_top20.csv")
    rcp85_2050_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp85_2050_top20.csv")
    rcp85_2075_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp85_2075_top20.csv")
    rcp85_2100_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp85_2100_top20.csv")

    plt.scatter(rcp26_2050_top20[["param_min_samples_split"]], rcp26_2050_top20[["mean_test_score"]],
                label="RCP 2.6\n2050", facecolors='none', edgecolors='blue')
    plt.scatter(rcp26_2075_top20[["param_min_samples_split"]], rcp26_2075_top20[["mean_test_score"]],
                label="RCP 2.6\n2075", facecolors='none', edgecolors='red')
    plt.scatter(rcp26_2100_top20[["param_min_samples_split"]], rcp26_2100_top20[["mean_test_score"]],
                label="RCP 2.6\n2100", facecolors='none', edgecolors='orange')
    plt.scatter(rcp85_2050_top20[["param_min_samples_split"]], rcp85_2050_top20[["mean_test_score"]],
                label="RCP 8.5\n2050", color='blue', marker='x')
    plt.scatter(rcp85_2075_top20[["param_min_samples_split"]], rcp85_2075_top20[["mean_test_score"]],
                label="RCP 8.5\n2075", color='red', marker='x')
    plt.scatter(rcp85_2100_top20[["param_min_samples_split"]], rcp85_2100_top20[["mean_test_score"]],
                label="RCP 8.5\n2050", color='orange', marker='x')
    plt.xlabel("min_samples_split")
    plt.ylabel("Mean cross-validation accuracy")
    plt.title("Mean cross-validation accuracy vs min_samples_split")
    plt.legend(bbox_to_anchor=(.995, 1.02))
    plt.subplots_adjust(right=.81)
    plt.show()


def plot_n_estimators():
    rcp26_2050_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp26_2050_top20.csv")
    rcp26_2075_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp26_2075_top20.csv")
    rcp26_2100_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp26_2100_top20.csv")
    rcp85_2050_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp85_2050_top20.csv")
    rcp85_2075_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp85_2075_top20.csv")
    rcp85_2100_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp85_2100_top20.csv")

    plt.scatter(rcp26_2050_top20[["param_n_estimators"]], rcp26_2050_top20[["mean_test_score"]],
                label="RCP 2.6\n2050", facecolors='none', edgecolors='blue')
    plt.scatter(rcp26_2075_top20[["param_n_estimators"]], rcp26_2075_top20[["mean_test_score"]],
                label="RCP 2.6\n2075", facecolors='none', edgecolors='red')
    plt.scatter(rcp26_2100_top20[["param_n_estimators"]], rcp26_2100_top20[["mean_test_score"]],
                label="RCP 2.6\n2100", facecolors='none', edgecolors='orange')
    plt.scatter(rcp85_2050_top20[["param_n_estimators"]], rcp85_2050_top20[["mean_test_score"]],
                label="RCP 8.5\n2050", color='blue', marker='x')
    plt.scatter(rcp85_2075_top20[["param_n_estimators"]], rcp85_2075_top20[["mean_test_score"]],
                label="RCP 8.5\n2075", color='red', marker='x')
    plt.scatter(rcp85_2100_top20[["param_n_estimators"]], rcp85_2100_top20[["mean_test_score"]],
                label="RCP 8.5\n2050", color='orange', marker='x')
    plt.xlabel("n_estimators")
    plt.ylabel("Mean cross-validation accuracy")
    plt.title("Mean cross-validation accuracy vs n_estimators")
    plt.legend(bbox_to_anchor=(.995, 1.02))
    plt.subplots_adjust(right=.81)
    plt.show()


def histogram_max_depth():
    rcp26_2050_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp26_2050_top20.csv")
    rcp26_2075_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp26_2075_top20.csv")
    rcp26_2100_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp26_2100_top20.csv")
    rcp85_2050_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp85_2050_top20.csv")
    rcp85_2075_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp85_2075_top20.csv")
    rcp85_2100_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp85_2100_top20.csv")

    max_depth_list = [x[0] for x in rcp26_2050_top20[["param_max_depth"]].values.tolist()]
    max_depth_list += [x[0] for x in rcp26_2075_top20[["param_max_depth"]].values.tolist()]
    max_depth_list += [x[0] for x in rcp26_2100_top20[["param_max_depth"]].values.tolist()]
    max_depth_list += [x[0] for x in rcp85_2050_top20[["param_max_depth"]].values.tolist()]
    max_depth_list += [x[0] for x in rcp85_2075_top20[["param_max_depth"]].values.tolist()]
    max_depth_list += [x[0] for x in rcp85_2100_top20[["param_max_depth"]].values.tolist()]
    max_depth_counter = Counter(max_depth_list)
    print(max_depth_counter)
    plt.bar(list(max_depth_counter.keys()), list(max_depth_counter.values()))
    plt.xlabel("max_depth")
    plt.ylabel("frequency")
    plt.title("Histogram of max_depth")
    plt.show()


def histogram_min_samples_split():
    rcp26_2050_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp26_2050_top20.csv")
    rcp26_2075_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp26_2075_top20.csv")
    rcp26_2100_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp26_2100_top20.csv")
    rcp85_2050_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp85_2050_top20.csv")
    rcp85_2075_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp85_2075_top20.csv")
    rcp85_2100_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp85_2100_top20.csv")

    min_samples_split_list = [x[0] for x in rcp26_2050_top20[["param_min_samples_split"]].values.tolist()]
    min_samples_split_list += [x[0] for x in rcp26_2075_top20[["param_min_samples_split"]].values.tolist()]
    min_samples_split_list += [x[0] for x in rcp26_2100_top20[["param_min_samples_split"]].values.tolist()]
    min_samples_split_list += [x[0] for x in rcp85_2050_top20[["param_min_samples_split"]].values.tolist()]
    min_samples_split_list += [x[0] for x in rcp85_2075_top20[["param_min_samples_split"]].values.tolist()]
    min_samples_split_list += [x[0] for x in rcp85_2100_top20[["param_min_samples_split"]].values.tolist()]
    min_samples_split_counter = Counter(min_samples_split_list)
    print(min_samples_split_counter)
    plt.bar(list(min_samples_split_counter.keys()), list(min_samples_split_counter.values()))
    plt.xlabel("min_samples_split")
    plt.ylabel("frequency")
    plt.title("Histogram of min_samples_split")
    plt.show()


def histogram_n_estimators():
    rcp26_2050_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp26_2050_top20.csv")
    rcp26_2075_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp26_2075_top20.csv")
    rcp26_2100_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp26_2100_top20.csv")
    rcp85_2050_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp85_2050_top20.csv")
    rcp85_2075_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp85_2075_top20.csv")
    rcp85_2100_top20 = pd.read_csv("../data/new_csv/hyperparameter_tuning/rcp85_2100_top20.csv")

    n_estimators_list = [x[0] for x in rcp26_2050_top20[["param_n_estimators"]].values.tolist()]
    n_estimators_list += [x[0] for x in rcp26_2075_top20[["param_n_estimators"]].values.tolist()]
    n_estimators_list += [x[0] for x in rcp26_2100_top20[["param_n_estimators"]].values.tolist()]
    n_estimators_list += [x[0] for x in rcp85_2050_top20[["param_n_estimators"]].values.tolist()]
    n_estimators_list += [x[0] for x in rcp85_2075_top20[["param_n_estimators"]].values.tolist()]
    n_estimators_list += [x[0] for x in rcp85_2100_top20[["param_n_estimators"]].values.tolist()]
    n_estimators_counter = Counter(n_estimators_list)
    print(n_estimators_counter)
    plt.bar(list(n_estimators_counter.keys()), list(n_estimators_counter.values()), width=50)
    plt.xlabel("n_estimators")
    plt.ylabel("frequency")
    plt.title("Histogram of n_estimators")
    plt.show()


if __name__ == '__main__':
    rcp26_2050 = pd.read_csv("gridsearchcv_results/lower_values_grid_cv_results_RCP 2.6-2050.csv")
    rcp26_2075 = pd.read_csv("gridsearchcv_results/lower_values_grid_cv_results_RCP 2.6-2075.csv")
    rcp26_2100 = pd.read_csv("gridsearchcv_results/lower_values_grid_cv_results_RCP 2.6-2100.csv")
    rcp85_2050 = pd.read_csv("gridsearchcv_results/lower_values_grid_cv_results_RCP 8.5-2050.csv")
    rcp85_2075 = pd.read_csv("gridsearchcv_results/lower_values_grid_cv_results_RCP 8.5-2075.csv")
    rcp85_2100 = pd.read_csv("gridsearchcv_results/lower_values_grid_cv_results_RCP 8.5-2100.csv")

    #top_10_cv_tables(rcp26_2050, rcp26_2075, rcp26_2100, rcp85_2050, rcp85_2075, rcp85_2100, 10, "lower_values_grid")

    df = rcp26_2050[rcp26_2050['params'] == "{'max_depth': 14, 'max_features': 15, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 500}"]
    df = df[["param_max_depth", "param_max_features", "param_min_samples_leaf",
                      "param_min_samples_split", "param_n_estimators", "mean_test_score",
                      "rank_test_score"]]
    pd.set_option('display.max_columns', None)
    print(df)