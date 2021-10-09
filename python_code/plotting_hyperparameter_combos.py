import matplotlib.pyplot as plt
import pandas as pd
import pprint
from python_code.sklearn_slr_classification import feature_color_dict


def plotting_accuracies():
    fig, axs = plt.subplots(2, 2)
    file_path = "./forests/forest_performance/"
    file_prefixes = ["new_hyperparams_", "new_hyperparams_2_", "new_hyperparams_3_",
                     "new_hyperparams_min_leaf_1_", "new_hyperparams_min_leaf_4_", "new_hyperparams_4_"]
    file_suffix = "_performance_measures.csv"
    rcp_list = ['rcp26', 'rcp85']
    years = [n for n in range(2020, 2151, 5)]

    accuracy_dict = {}
    for prefix in file_prefixes:
        accuracy_dict[prefix] = {}
        for rcp in rcp_list:
            accuracy_dict[prefix][rcp] = {"Training Accuracy": [], "Test Accuracy": []}


    accuracy_dict["new_hyperparams_"]["Params"] = "max_depth=18, max_features=15, " \
                                                      "min_samples_leaf=2, " \
                                                      "min_samples_split=4, n_estimators=250"
    accuracy_dict["new_hyperparams_2_"]["Params"] = "max_depth=14, max_features=15, " \
                                                      "min_samples_leaf=1, " \
                                                      "min_samples_split=4, n_estimators=500"
    accuracy_dict["new_hyperparams_3_"]["Params"] = "max_depth=12, max_features=15, " \
                                                        "min_samples_leaf=3, " \
                                                        "min_samples_split=4, n_estimators=500"
    accuracy_dict["new_hyperparams_min_leaf_1_"]["Params"] = "max_depth=18, max_features=15, " \
                                                      "min_samples_leaf=1, " \
                                                      "min_samples_split=4, n_estimators=250"
    accuracy_dict["new_hyperparams_min_leaf_4_"]["Params"] = "max_depth=18, max_features=15, " \
                                                                 "min_samples_leaf=4, " \
                                                                 "min_samples_split=4, n_estimators=250"
    accuracy_dict["new_hyperparams_4_"]["Params"] = "max_depth=14, max_features=15, " \
                                                    "min_samples_leaf=7, " \
                                                    "min_samples_split=10, n_estimators=750"

    color_dict = feature_color_dict(file_prefixes)

    for rcp in rcp_list:
        for yr in years:
            for prefix in file_prefixes:
                df = pd.read_csv(file_path + prefix + rcp + "_" + str(yr) +file_suffix)
                accuracy_dict[prefix][rcp]["Training Accuracy"].append(df['train_accuracy'][0])
                accuracy_dict[prefix][rcp]["Test Accuracy"].append(df['test_accuracy'][0])

    #pprint.pprint(accuracy_dict)

    for key in accuracy_dict:
        params_label = accuracy_dict[key]["Params"]
        # column 0 = RCP 2.6    column 1 = RCP 8.5
        # row 0 = Training   row 1 = Testing
        axs[0, 0].plot(years, accuracy_dict[key]["rcp26"]["Training Accuracy"], label=params_label,
                       color=color_dict[key])
        axs[1, 0].plot(years, accuracy_dict[key]["rcp26"]["Test Accuracy"], label=params_label,
                       color=color_dict[key])
        axs[0, 1].plot(years, accuracy_dict[key]["rcp85"]["Training Accuracy"], label=params_label,
                       color=color_dict[key])
        axs[1, 1].plot(years, accuracy_dict[key]["rcp85"]["Test Accuracy"], label=params_label,
                       color=color_dict[key])

    axs[0, 0].legend(bbox_to_anchor=(.1, 1.07))
    axs[0, 0].set_title("RCP 2.6")
    axs[0, 1].set_title("RCP 8.5")
    axs[0, 0].set(ylabel='Training Accuracy')
    axs[1, 0].set(ylabel='Testing Accuracy')
    axs[0, 0].set_ylim(bottom=.89, top=1)
    axs[0, 1].set_ylim(bottom=.89, top=1)
    axs[1, 0].set_ylim(bottom=.89, top=1)
    axs[1, 1].set_ylim(bottom=.89, top=1)
    plt.subplots_adjust(top=.82, bottom=.045, left=.08, right=.97)
    plt.show()


if __name__ == '__main__':
    plotting_accuracies()