import matplotlib.pyplot as plt
import pandas as pd

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
