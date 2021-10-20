import matplotlib.pyplot as plt
import pandas as pd
import pprint
import numpy as np
from python_code.sklearn_slr_classification import feature_color_dict


def plotting_accuracies():
    fig, axs = plt.subplots(3, 2)
    file_path = "./forests/forest_performance/"
    file_prefixes = ["new_hyperparams_", "new_hyperparams_2_", "new_hyperparams_3_",
                     "new_hyperparams_min_leaf_1_", "new_hyperparams_min_leaf_4_", "new_hyperparams_4_",
                     "new_hyperparams_5_", "new_hyperparams_6_"]
    file_prefix_no_min_leaf_1 = ["new_hyperparams_", "new_hyperparams_3_", "new_hyperparams_min_leaf_4_",
                               "new_hyperparams_4_", "new_hyperparams_5_", "new_hyperparams_6_"]
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
    accuracy_dict["new_hyperparams_5_"]["Params"] = "max_depth=16, max_features=15, " \
                                                    "min_samples_leaf=3, " \
                                                    "min_samples_split=13, n_estimators=1000"
    accuracy_dict["new_hyperparams_6_"]["Params"] = "max_depth=16, max_features=15, " \
                                                    "min_samples_leaf=2, " \
                                                    "min_samples_split=16, n_estimators=250"

    means_dict = {}

    for rcp in rcp_list:
        mean_per_year = []
        for yr in years:
            sum_per_year = 0
            for prefix in file_prefixes:
                df = pd.read_csv(file_path + prefix + rcp + "_" + str(yr) +file_suffix)
                accuracy_dict[prefix][rcp]["Training Accuracy"].append(df['train_accuracy'][0])
                accuracy_dict[prefix][rcp]["Test Accuracy"].append(df['test_accuracy'][0])
                sum_per_year += df['test_accuracy'][0]

                #if yr == 2100:
                #    print(rcp, accuracy_dict[prefix]["Params"], df['train_accuracy'][0], df['test_accuracy'][0], sep="\t\t")

            mean_per_year.append(sum_per_year / len(file_prefixes))
        means_dict[rcp] = mean_per_year

    #pprint.pprint(accuracy_dict)
    #pprint.pprint(means_dict)

    for key in accuracy_dict:
        params_label = accuracy_dict[key]["Params"]
        # column 0 = RCP 2.6    column 1 = RCP 8.5
        # row 0 = Training   row 1 = Testing    row 2 = Normalized Testing
        axs[0, 0].plot(years, accuracy_dict[key]["rcp26"]["Training Accuracy"], label=params_label)
        axs[1, 0].plot(years, accuracy_dict[key]["rcp26"]["Test Accuracy"], label=params_label)
        axs[2, 0].plot(years, [accuracy_dict[key]["rcp26"]["Test Accuracy"][i] - means_dict['rcp26'][i] for i
                               in range(len(years))], label=params_label)
        print("RCP 2.6", accuracy_dict[key]["Params"],
              np.asarray([accuracy_dict[key]["rcp26"]["Test Accuracy"][i] -
                          means_dict['rcp26'][i] for i in range(len(years))]).mean(), sep="\t")
        print("RCP 8.5", accuracy_dict[key]["Params"], np.asarray(
            [accuracy_dict[key]["rcp85"]["Test Accuracy"][i] - means_dict['rcp85'][i] for i in
             range(len(years))]).mean(), sep="\t")

        axs[0, 1].plot(years, accuracy_dict[key]["rcp85"]["Training Accuracy"], label=params_label)
        axs[1, 1].plot(years, accuracy_dict[key]["rcp85"]["Test Accuracy"], label=params_label)
        axs[2, 1].plot(years, [accuracy_dict[key]["rcp85"]["Test Accuracy"][i] - means_dict['rcp85'][i] for i
                               in range(len(years))], label=params_label)

    list_10_yrs = []
    for yr in range(2020, 2151, 10):
        list_10_yrs.append(yr)

    for i in [0, 1]:
        axs[2, i].set_ylim(bottom=-.015, top=.015)      # +/- 0.013 limit for no min_samples_leaf = 1
        axs[2, i].grid(b=True)
        axs[2, i].set_xlim(left=2020, right=2150)
        axs[2, i].set_xticks(list_10_yrs)
        for j in [0, 1]:
            axs[i, j].set_ylim(bottom=.89, top=1)
            axs[i, j].grid(b=True)
            axs[i, j].set_xlim(left=2020, right=2150)
            axs[i, j].set_xticks(list_10_yrs)
            axs[i, j].set_xticklabels(list_10_yrs)

    #axs[0, 0].legend(bbox_to_anchor=(.1, 1.115), ncol=2)       # legend loc for no min_samples_leaf = 1
    axs[0, 0].legend(bbox_to_anchor=(.1, 1.14), ncol=2)       # with min_samples_leaf = 1
    axs[0, 0].set_ylabel('Training Accuracy', fontsize=12)
    axs[1, 0].set_ylabel('Testing Accuracy', fontsize=12)
    axs[2, 0].set_ylabel('Normalized Testing Accuracy', fontsize=12)
    #plt.subplots_adjust(top=.85, bottom=.045, left=.08, right=.97)     # subplot adjustments for no min_samples_leaf = 1
    plt.subplots_adjust(top=.8, bottom=.045, left=.08, right=.97)      # with min_samples_leaf = 1
    fig.suptitle("Accuracy for Random Forests using Various Hyperparameter Combinations", fontsize=16)
    #fig.text(.27, .855, "RCP2.6", fontsize=14, ha='center')     # for no min_samples_leaf = 1
    #fig.text(.785, .855, "RCP8.5", fontsize=14, ha='center')    # for no min_samples_leaf = 1
    fig.text(.27, .81, "RCP2.6", fontsize=14, ha='center')      # with min_samples_leaf = 1
    fig.text(.785, .81, "RCP8.5", fontsize=14, ha='center')     # with min_samples_leaf = 1
    axs[0, 0].set_title("(a)", loc='left', x=.01, y=.865, fontsize=14)      # y = .885 for no min_samples_leaf = 1
    axs[0, 1].set_title("(b)", loc='left', x=.01, y=.865, fontsize=14)
    axs[1, 0].set_title("(c)", loc='left', x=.01, y=.865, fontsize=14)
    axs[1, 1].set_title("(d)", loc='left', x=.01, y=.865, fontsize=14)
    axs[2, 0].set_title("(e)", loc='left', x=.01, y=.865, fontsize=14)
    axs[2, 1].set_title("(f)", loc='left', x=.01, y=.865, fontsize=14)
    plt.show()


if __name__ == '__main__':
    plotting_accuracies()