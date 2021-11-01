from python_code.sklearn_slr_classification import MODEL_DICT, feature_color_dict, PARAMETER_DICT, load_forests
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from pprint import pprint
from sklearn.inspection import permutation_importance
import pandas as pd


def slr_stacked_dif_importances_plot(param_sample_df, rcp85_forest_list, years, importance_threshold=.04):
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
    name = "RCP8.5"
    importances_info_list = []

    for i in range(2):
        forest_list = rcp85_forest_list
        importances_info = {}
        for j in range(len(forest_list)):
            forest = forest_list[j]
            yr = years[j]

            # stacked importances dictionary
            if i == 0:
                importances = forest.feature_importances_
            else:
                X_test = pd.read_csv("../data/new_csv/rcp85_" + str(yr) +"_Xtest.csv")
                y_test = pd.read_csv("../data/new_csv/rcp85_" + str(yr) + "_ytest.csv", header=None)
                importances = permutation_importance(forest, X_test, y_test).importances_mean
                importances = importances / importances.sum()

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

    pprint(importances_info_list)

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
        if i == 0:
            axs[i].set_ylabel("Relative Gini Importance", fontsize=14)
        elif i == 1:
            axs[i].set_ylabel("Relative Mean Permutation Importance", fontsize=14)
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
        title = title_label + " " + name + " Feature Importances"
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

    plt.figlegend(handles=label_values, labels=label_keys, bbox_to_anchor=(.99, .995), fontsize=12)
    plt.subplots_adjust(left=.105, right=.815, top=.96, bottom=.065, hspace=.258)
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv("../data/new_csv/RData_parameters_sample.csv")
    slr_rcp85_5step = pd.read_csv("../data/new_csv/slr_rcp85_5yrstep.csv")
    yrs_rcp85 = slr_rcp85_5step.columns.tolist()
    rcp85_forest_list = load_forests(yrs_rcp85, "new_hyperparams_rcp85")
    slr_stacked_dif_importances_plot(df, rcp85_forest_list, yrs_rcp85)