import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from sklearn.model_selection import train_test_split


def classify_data(slr_df, print_threshold=False, percentile=.9):
    """
    Takes a dataframe of sea-level rise values for various years and classifies the values as "low" or "high"
    depending on if the value is above or below the 90th percentile of the data for each year
    :param slr_df: dataframe of the output sea-level rise values, where the columns are years
    :param print_threshold: boolean that controls whether the percentile value of each year should be printed
    :param percentile: decimal value of the percentile to be used as the threshold of "low" and "high"
    :return: df_classify -- a dataframe where all of the values are either "low" or "high"
    """

    # classify data
    years = slr_df.columns.tolist()
    threshold = slr_df.quantile(q=percentile)
    if print_threshold:
        print("Threshold:\n", threshold)
    row_list = []
    for i in range(slr_df.shape[0]):
        row = []
        for j in range(len(years)):
            if slr_df.iloc[i, j] >= threshold.iloc[j]:
                row.append("high")
            else:
                row.append("low")
        row_list.append(row)
    df_classify = pd.DataFrame(row_list, columns=years)
    return df_classify


def split_train_validation_test(df, feature_list, rcp, year, print_ratios, file_path):
    """
    Splits the data into training, validation, and testing data sets.  The splits are 60% training, 20% validation,
    and 20% testing.  These data sets are saved as CSVs into the file_path folder.
    :param df: oversampled and classified dataframe
    :param feature_list: list of the parameter sample features
    :param rcp: RCP string without spaces/punctuation (ex: 'rcp26')
    :param year: string of the year to split the data for
    :param print_ratios: boolean that controls whether to print the amount of non-high-end and high-end data
    points are in each data set that was just split
    :param file_path: the file path to save the CSVs of the newly split data sets into
    :return: None
    """

    x = df[feature_list]
    y = df[year]
    x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.4)  # train= 60%, rest= 40%
    x_validation, x_test, y_validation, y_test = train_test_split(x_rest, y_rest,
                                                                  test_size=0.5)  # validation = 20%, test= 20%

    if print_ratios:
        print(year, " train --- low (non-high-end):", y_train[y_train == 'low'].shape[0], "high:",
              y_train[y_train == 'high'].shape[0])
        print(year, " validation --- low (non-high-end):", y_validation[y_validation == 'low'].shape[0], "high:",
              y_validation[y_validation == 'high'].shape[0])
        print(year, " test --- low (non-high-end):", y_test[y_test == 'low'].shape[0], "high:",
              y_test[y_test == 'high'].shape[0])

    x_train_file_path = file_path + rcp + "_" + str(year) + "_Xtrain.csv"
    x_train.to_csv(x_train_file_path, index=False)
    y_train_file_path = file_path + rcp + "_" + str(year) + "_ytrain.csv"
    y_train.to_csv(y_train_file_path, index=False, header=False)

    x_val_file_path = file_path + rcp + "_" + str(year) + "_Xvalidation.csv"
    x_test.to_csv(x_val_file_path, index=False)
    y_val_file_path = file_path + rcp + "_" + str(year) + "_yvalidation.csv"
    y_test.to_csv(y_val_file_path, index=False, header=False)

    x_test_file_path = file_path + rcp + "_" + str(year) + "_Xtest.csv"
    x_test.to_csv(x_test_file_path, index=False)
    y_test_file_path = file_path + rcp + "_" + str(year) + "_ytest.csv"
    y_test.to_csv(y_test_file_path, index=False, header=False)


def oversample_data(parameter_sample_df, df_classify, rcp, print_class_count=False, print_ratios=False,
                    file_path="../data/new_csv/preprocessed_data/", skip_years=None, classify_percentile=.9):
    """
    Oversamples the data to make the amount of non-high-end and high-end data points approximately equal.  Calls
    the split_train_validation_test() function to split data into training, validation, and testing and saves them
    as CSVs into the file_path folder.
    :param parameter_sample_df: dataframe of the input feature values
    :param df_classify: dataframe of the classified output year values
    :param rcp: RCP string without spaces/punctuation (ex: 'rcp26')
    :param print_class_count: boolean that controls whether to print the amount of non-high-end and high-end data
    points are in the overall dataset before splitting into training, validaiton, and testing
    :param print_ratios: boolean that controls whether to print the amount of non-high-end and high-end data
    points are in each data set that was split
    :param file_path: the file path to save the CSVs of the newly split data sets into
    :param skip_years: list of the years (as intergers) to not save data for
    :return: None
    """

    features = parameter_sample_df.columns.tolist()
    years = df_classify.columns.tolist()
    oversample_by = int(classify_percentile / (1 - classify_percentile) - 1)

    for yr in years:
        if skip_years is not None and int(yr) in skip_years:
            if print_class_count:
                print("Skipping", yr)
            continue

        # join param values with output values of that year
        yr_df = parameter_sample_df.join(df_classify[yr], how="outer")
        yr_df = yr_df.dropna()

        # oversample "high" GMSLR class
        high_df = yr_df[yr_df[yr] == "high"]
        yr_df = yr_df.append([high_df]*oversample_by, ignore_index=True)    # gives a future warning about df.append

        # shuffle data
        yr_df = yr_df.sample(frac=1)

        if print_class_count:
            print(yr, "--- low (non-high-end):", yr_df[yr_df[yr] == 'low'].shape[0],
                  "high:", yr_df[yr_df[yr] == 'high'].shape[0])

        # split into training, validation, and testing & save to csv
        split_train_validation_test(yr_df, features, rcp, yr, print_ratios=print_ratios, file_path=file_path)


if __name__ == '__main__':
    parameter_sample_df = pd.read_csv("../data/new_csv/RData_parameters_sample.csv")
    slr_rcp26_5step = pd.read_csv("../data/new_csv/slr_rcp26_5yrstep.csv")
    slr_rcp85_5step = pd.read_csv("../data/new_csv/slr_rcp85_5yrstep.csv")

    rcp_dict = {'rcp26': slr_rcp26_5step, 'rcp85': slr_rcp85_5step}
    percentile = .8

    for rcp in rcp_dict.keys():
        classify_df = classify_data(rcp_dict[rcp], percentile=percentile)
        oversample_data(parameter_sample_df, classify_df, rcp, classify_percentile=percentile, print_ratios=True,
                        print_class_count=True, file_path="../data/new_csv/preprocessed_data/80th_percentile/")
