import matplotlib.pyplot as plt
import math
from sklearn_slr_classification import classify_data, slr_forest



def max_features(param_samples_df, slr_df, max_d=None, min_samp_leaf=1, max_samp=None, print_threshold=True):
    """
    Create forests with using different max_features values and plot the training and validation accuracies vs the
    max_feature values
    :param param_samples_df: dataframe of the input feature values
    :param slr_df: dataframe of the output year values
    :param max_d: the maximum depth of the tree as an integer -- default = None
    :param min_samp_leaf: the minimum integer number of samples required to be in a leaf node -- default = 1
    :param max_samp: the integer number of samples to draw from the training data to train each tree (since the forest
    will be bootstrapped) -- default = None which means all of the training samples will be used to train each tree
    :param print_threshold: boolean that controls whether the 90th percentile value of each year should be printed
    :return: None
    """
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
    """
    Create forests with using different max_depth values and plot the training and validation accuracies vs the
    max_depth values
    :param param_samples_df: dataframe of the input feature values
    :param slr_df: dataframe of the output year values
    :param max_feat: integer number of features to consider when determining the best split -- default = "auto" which
    takes the square root of the total number of features
    :param min_samp_leaf: the minimum integer number of samples required to be in a leaf node -- default = 1
    :param max_samp: the integer number of samples to draw from the training data to train each tree (since the forest
    will be bootstrapped) -- default = None which means all of the training samples will be used to train each tree
    :param print_threshold: boolean that controls whether the 90th percentile value of each year should be printed
    :return: None
    """
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
    """
    Create forests with using different min_samples_leaf values and plot the training and validation accuracies vs the
    min_samples_leaf values
    :param param_samples_df: dataframe of the input feature values
    :param slr_df: dataframe of the output year values
    :param max_feat:  integer number of features to consider when determining the best split -- default = "auto" which
    takes the square root of the total number of features
    :param max_d: the maximum depth of the tree as an integer -- default = None
    :param max_samp: the integer number of samples to draw from the training data to train each tree (since the forest
    will be bootstrapped) -- default = None which means all of the training samples will be used to train each tree
    :param print_threshold: boolean that controls whether the 90th percentile value of each year should be printed
    :return: None
    """
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
    """
    Create forests with using different max_samples values and plot the training and validation accuracies vs the
    max_samples values
    :param param_samples_df: dataframe of the input feature values
    :param slr_df: dataframe of the output year values
    :param max_feat: integer number of features to consider when determining the best split -- default = "auto" which
    takes the square root of the total number of features
    :param max_d: the maximum depth of the tree as an integer -- default = None
    :param min_samp_leaf: the minimum integer number of samples required to be in a leaf node -- default = 1
    :param print_threshold: boolean that controls whether the 90th percentile value of each year should be printed
    :return: None
    """
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


def parameter_loop_max_features(param_samples_df, slr_df):
    """
    Loops through running the max_features(), max_depth(), and max_samples() functions by using the user-defined values
    of the parameters based on the previous iteration in the loop.
    :param param_samples_df: dataframe of the input feature values
    :param slr_df: dataframe of the output year values
    :return: None
    """
    stop = 1
    counter = 1
    max_depth_val = None
    max_samples_val = None
    while stop != 0:
        print("Iteration", counter)
        max_features(param_samples_df, slr_df, max_d=max_depth_val, max_samp=max_samples_val, print_threshold=False)
        max_features_val = int(input("Max features: "))
        max_depth(param_samples_df, slr_df, max_feat=max_features_val, max_samp=max_samples_val, print_threshold=False)
        max_depth_val = int(input("Max depth: "))
        max_samples(param_samples_df, slr_df, max_feat=max_features_val, max_d=max_depth_val, print_threshold=False)
        max_samples_val = int(input("Max samples: "))
        print("Iteration", counter, "Summary")
        print("\tMax features =", max_features_val)
        print("\tMax depth = ", max_depth_val)
        print("\tMax samples =", max_samples_val)
        stop = int(input("Enter 0 to stop"))
        counter += 1


def parameter_loop_min_samples_leaf(param_samples_df, slr_df):
    """
    Loops through running the max_features(), min_samples_leaf(), and max_samples() functions by using the user-defined
    values of the parameters based on the previous iteration in the loop.
    :param param_samples_df: dataframe of the input feature values
    :param slr_df: dataframe of the output year values
    :return: None
    """
    stop = 1
    counter = 1
    min_samples_leaf_val = 1
    max_samples_val = None
    while stop != 0:
        print("Iteration", counter)
        max_features(param_samples_df, slr_df, min_samp_leaf=min_samples_leaf_val, max_samp=max_samples_val,
                     print_threshold=False)
        max_features_val = int(input("Max features: "))
        min_samples_leaf(param_samples_df, slr_df, max_feat=max_features_val, max_samp=max_samples_val,
                         print_threshold=False)
        min_samples_leaf_val = int(input("Min samples leaf: "))
        max_samples(param_samples_df, slr_df, max_feat=max_features_val, min_samp_leaf=min_samples_leaf_val,
                    print_threshold=False)
        max_samples_val = int(input("Max samples: "))
        print("Iteration", counter, "Summary")
        print("\tMax features =", max_features_val)
        print("\tMin samples leaf = ", min_samples_leaf_val)
        print("\tMax samples =", max_samples_val)
        stop = int(input("Enter 0 to stop"))
        counter += 1