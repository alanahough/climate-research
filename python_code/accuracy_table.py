import pandas as pd


def accuracy_latex_table(years, performance_path):
    rcp26_train = []
    rcp26_val = []
    rcp26_test = []
    rcp85_train = []
    rcp85_val = []
    rcp85_test = []
    for yr in years:
        path26 = performance_path + "rcp26_" + str(yr) + "_performance_measures.csv"
        df_26 = pd.read_csv(path26)
        path85 = performance_path + "rcp85_" + str(yr) + "_performance_measures.csv"
        df_85 = pd.read_csv(path85)

        rcp26_train.append(df_26.train_accuracy[0])
        rcp26_val.append(df_26.val_accuracy[0])
        rcp26_test.append(df_26.test_accuracy[0])

        rcp85_train.append(df_85.train_accuracy[0])
        rcp85_val.append(df_85.val_accuracy[0])
        rcp85_test.append(df_85.test_accuracy[0])

    accuracy_df = pd.DataFrame({'Year': years, "2.6 train": rcp26_train, '2.6 validation': rcp26_val,
                                '2.6 test': rcp26_test, '8.5 train': rcp85_train, '8.5 validation': rcp85_val,
                                '8.5 test': rcp85_test})
    print(accuracy_df.to_latex(index=False))



if __name__ == '__main__':
    years = [x for x in range(2020, 2155, 5)]
    accuracy_latex_table(years, './forests/forest_accuracy/revision_2_')
