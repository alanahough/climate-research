import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    rcp26_2050 = pd.read_csv("gridsearchcv_results/cvResults_param_grid_tw_rcp26-2050.csv")
    rcp26_2050_top20 = rcp26_2050[rcp26_2050["rank_test_score"] <= 20].sort_values(by=['rank_test_score'])
    rcp26_2050_top20 = rcp26_2050_top20[["param_max_depth",	"param_max_features", "param_min_samples_leaf",
                                         "param_min_samples_split",	"param_n_estimators", "mean_test_score",
                                         "rank_test_score"]]
    rcp26_2050_top20.to_csv("../data/new_csv/hyperparameter_tuning/rcp26_2050_top20.csv", index=False)

    rcp26_2075 = pd.read_csv("gridsearchcv_results/cvResults_param_grid_tw_rcp26-2075.csv")
    rcp26_2075_top20 = rcp26_2075[rcp26_2075["rank_test_score"] <= 20].sort_values(by=['rank_test_score'])
    rcp26_2075_top20 = rcp26_2075_top20[["param_max_depth",	"param_max_features", "param_min_samples_leaf",
                                         "param_min_samples_split",	"param_n_estimators", "mean_test_score",
                                         "rank_test_score"]]
    rcp26_2075_top20.to_csv("../data/new_csv/hyperparameter_tuning/rcp26_2075_top20.csv", index=False)

    rcp26_2100 = pd.read_csv("gridsearchcv_results/cvResults_param_grid_tw_rcp26-2100.csv")
    rcp26_2100_top20 = rcp26_2100[rcp26_2100["rank_test_score"] <= 20].sort_values(by=['rank_test_score'])
    rcp26_2100_top20 = rcp26_2100_top20[["param_max_depth",	"param_max_features", "param_min_samples_leaf",
                                         "param_min_samples_split",	"param_n_estimators", "mean_test_score",
                                         "rank_test_score"]]
    rcp26_2100_top20.to_csv("../data/new_csv/hyperparameter_tuning/rcp26_2100_top20.csv", index=False)

    rcp85_2050 = pd.read_csv("gridsearchcv_results/cvResults_param_grid_tw_rcp85-2050.csv")
    rcp85_2050_top20 = rcp85_2050[rcp85_2050["rank_test_score"] <= 20].sort_values(by=['rank_test_score'])
    rcp85_2050_top20 = rcp85_2050_top20[["param_max_depth",	"param_max_features", "param_min_samples_leaf",
                                         "param_min_samples_split",	"param_n_estimators", "mean_test_score",
                                         "rank_test_score"]]
    rcp85_2050_top20.to_csv("../data/new_csv/hyperparameter_tuning/rcp85_2050_top20.csv", index=False)

    rcp85_2075 = pd.read_csv("gridsearchcv_results/cvResults_param_grid_tw_rcp85-2075.csv")
    rcp85_2075_top20 = rcp85_2075[rcp85_2075["rank_test_score"] <= 20].sort_values(by=['rank_test_score'])
    rcp85_2075_top20 = rcp85_2075_top20[["param_max_depth",	"param_max_features", "param_min_samples_leaf",
                                         "param_min_samples_split",	"param_n_estimators", "mean_test_score",
                                         "rank_test_score"]]
    rcp85_2075_top20.to_csv("../data/new_csv/hyperparameter_tuning/rcp85_2075_top20.csv", index=False)

    rcp85_2100 = pd.read_csv("gridsearchcv_results/cvResults_param_grid_tw_rcp85-2100.csv")
    rcp85_2100_top20 = rcp85_2100[rcp85_2100["rank_test_score"] <= 20].sort_values(by=['rank_test_score'])
    rcp85_2100_top20 = rcp85_2100_top20[["param_max_depth",	"param_max_features", "param_min_samples_leaf",
                                         "param_min_samples_split",	"param_n_estimators", "mean_test_score",
                                         "rank_test_score"]]
    rcp85_2100_top20.to_csv("../data/new_csv/hyperparameter_tuning/rcp85_2100_top20.csv", index=False)


    # plot max_depth
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

    # plot min_samples_leaf
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

    # plot min_samples_split
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

    # plot n_estimators
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