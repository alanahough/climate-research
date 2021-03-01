## sklearn_slr_classifcation.py

### Imports


```python
from sklearn import tree, metrics, ensemble
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import math
import pylab
import joblib
import os
```

### Global settings


```python
MAX_DEPTH = 14
MAX_FEATURES = "sqrt"
MIN_SAMPLES_LEAF = 4
MIN_SAMPLES_SPLIT = 5
N_ESTIMATORS = 500

plt.rcParams['figure.figsize'] = [10, 9]
```

***
### `classify_data(df, print_threshold=False)`
Takes a dataframe of sea-level rise values for various years and classifies the values as "low" or "high" depending on if the value is above or below the 90th percentile of the data for each year.

**Parameters:**
- *df:* dataframe of the output sea-level rise values, where the columns are years
- *print_threshold:* boolean that controls whether the 90th percentile value of each year should be printed

**Returns:** a dataframe where all of the values are either "low" or "high"

**Definition:**


```python
def classify_data(df, print_threshold=False):
    years = df.columns.tolist()
    threshold = df.quantile(q=.9)
    if print_threshold is True:
        print("Threshold:\n", threshold)
    row_list = []
    for i in range(df.shape[0]):
        row = []
        for j in range(len(years)):
            if df.iloc[i, j] >= threshold.iloc[j]:
                row.append("high")
            else:
                row.append("low")
        row_list.append(row)
    df_classify = pd.DataFrame(row_list, columns=years)
    return df_classify
```

**Usage:**


```python
slr_rcp26_5step = pd.read_csv("../data/new_csv/slr_rcp26_5yrstep.csv")
slr_classify = classify_data(slr_rcp26_5step)
slr_classify.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2020</th>
      <th>2025</th>
      <th>2030</th>
      <th>2035</th>
      <th>2040</th>
      <th>2045</th>
      <th>2050</th>
      <th>2055</th>
      <th>2060</th>
      <th>2065</th>
      <th>...</th>
      <th>2105</th>
      <th>2110</th>
      <th>2115</th>
      <th>2120</th>
      <th>2125</th>
      <th>2130</th>
      <th>2135</th>
      <th>2140</th>
      <th>2145</th>
      <th>2150</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>9995</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>...</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
    </tr>
    <tr>
      <td>9996</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>...</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
    </tr>
    <tr>
      <td>9997</td>
      <td>high</td>
      <td>high</td>
      <td>high</td>
      <td>high</td>
      <td>high</td>
      <td>high</td>
      <td>high</td>
      <td>high</td>
      <td>high</td>
      <td>high</td>
      <td>...</td>
      <td>high</td>
      <td>high</td>
      <td>high</td>
      <td>high</td>
      <td>high</td>
      <td>high</td>
      <td>high</td>
      <td>high</td>
      <td>high</td>
      <td>high</td>
    </tr>
    <tr>
      <td>9998</td>
      <td>high</td>
      <td>high</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>...</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
    </tr>
    <tr>
      <td>9999</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>...</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



***
### `gridsearch(param_samples_df, slr_df, year)`
Perform a gridsearch of the parameters used to create the forests, saves the best parameters to a CSV, and saves the cross validation information/result to another CSV.  The function definition contains the parameter grids I performed a gridsearch of.  Those can be used, or new parameter grids can be created within the function definition.  Before running this function, you will need to do the following:

- Change the parameter grid name in `grid_search` to the parameter grid you wish to use.  
    - For instance, `grid_search = GridSearchCV(estimator=forest, param_grid=` *your parameter grid* `, cv=5, n_jobs=-1, verbose=1, scoring="accuracy")`
- Change the filename of the `best_params_df` CSV file.
    - For instance, `best_params_df.to_csv("./gridsearchcv_results/` *new filename* `.csv", index=False)`
- Change the filename of the `score_df` CSV file.
    - For instance, `score_df.to_csv("./gridsearchcv_results/` *new filename* `.csv", index=False)`

**Parameters:**
- *param_samples_df:* dataframe of the input feature values
- *param slr_df:* dataframe of the output year values
- *year:* string of the year for the data to use when making the forests in the gridsearch

**Returns:** None

**Definition:**


```python
def gridsearch(param_samples_df, slr_df, year):
    slr_classify = classify_data(slr_df)
    df_slr = param_samples_df.join(slr_classify, how="outer")
    df_slr = df_slr.dropna()
    features = param_samples_df.columns.tolist()
    x = df_slr[features]
    y = df_slr[year]

    # 80% training, 20% testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    param_grid_1 = {
        'max_depth': [2, 4, 6, 8, 10],
        'max_features': ["sqrt", 10, 15, 20, 25, 30, 35],
        'max_samples': [1000, 2000, 3000, 4000, 4800],
        'min_samples_leaf': [1, 15, 30, 45, 60, 75, 90, 105],
        'min_samples_split': [15, 30, 45, 60, 75, 90, 105]
    }   #took like 14.5 hours and I forgot to print the best params
    param_grid_2 = {
        'max_depth': [3, 4, 5, 6],
        'max_features': ["sqrt", 10, 15, 20, 25, 30],
        'max_samples': [1000, 2000, 3000, 4000],
        'min_samples_leaf': [1, 20, 40, 60, 80, 100],
        'min_samples_split': [2, 20, 40, 60, 80, 100]
    }   #took around 4.5 hours -- {'max_depth': 6, 'max_features': 30, 'max_samples': 3000, 'min_samples_leaf': 1, 
    #                              'min_samples_split': 20}
    param_grid_3 = {
        'max_depth': [4, 5, 6, 7],
        'max_features': [20, 25, 30, 35],
        'max_samples': [2000, 3000, 4000],
        'min_samples_leaf': [1, 2, 4, 8, 12, 16],
        'min_samples_split': [5, 10, 15, 20, 25, 30]
    }   #took around 4 hours -- {'max_depth': 7, 'max_features': 35, 'max_samples': 4000, 'min_samples_leaf': 4, 
    #                            'min_samples_split': 20}
    param_grid_4 = {
        'max_depth': [6, 7, 8, 9, 10],
        'max_features': [25, 30, 32, 35, 38],
        'max_samples': [3000, 3500, 4000, 4500, 4800],
        'min_samples_leaf': [2, 4, 8],
        'min_samples_split': [10, 15, 20, 25, 30]
    }   #took around 7.5 hours -- {'max_depth': 10, 'max_features': 38, 'max_samples': 4800, 'min_samples_leaf': 8, 
    #                              'min_samples_split': 15}
    param_grid_5 = {
        'max_depth': [7, 10, 13, 16, 19],
        'max_features': [30, 32, 35, 38],
        'max_samples': [3500, 4000, 4500, 4800],
        'min_samples_leaf': [2, 4, 8, 10, 12],
        'min_samples_split': [10, 15, 20, 25, 30]
    }   #took around 11 hours -- {'max_depth': 13, 'max_features': 35, 'max_samples': 4500, 'min_samples_leaf': 2, 
    #                             'min_samples_split': 10}
    param_grid_6 = {
        'max_depth': [12, 13, 14],
        'max_features': [35, 36, 37, 38],
        'max_samples': [4000, 4500, 4800],
        'min_samples_leaf': [2, 4, 6, 8],
        'min_samples_split': [10, 13, 16, 20]
    }   #took around 3 hours -- {'max_depth': 13, 'max_features': 38, 'max_samples': 4800, 'min_samples_leaf': 8, 
    #                            'min_samples_split': 13}
        # BEST PARAMETERS:
        # {'max_depth': 14, 'max_features': 37, 'max_samples': 4500, 'min_samples_leaf': 4, 'min_samples_split': 10}
        # Mean cross-validated score of the best_estimator:  0.9333333333333333

    n_estimators_param_grid = {
        'n_estimators': range(100, 1001, 100),
        'max_depth': [14],
        'max_features': [37],
        'max_samples': [4500],
        'min_samples_leaf': [4],
        'min_samples_split': [10]
    }   # 'n_estimators': 700, Mean cross-validated score of the best_estimator:  0.9373750000000001
        # n_estimators: 300, Mean cross-validated score of the best_estimator:  0.936875
        
    usage_example_param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [3, 7],
        'max_features': [37],
        'min_samples_leaf': [4],
        'min_samples_split': [10]
    }

    forest = ensemble.RandomForestClassifier()
    grid_search = GridSearchCV(estimator=forest, param_grid=usage_example_param_grid, cv=5, n_jobs=-1, verbose=1,
                               scoring="accuracy")
    grid_search.fit(x_train, y_train)
    print("BEST PARAMETERS:")
    print(grid_search.best_params_)
    print("Mean cross-validated score of the best_estimator: ", grid_search.best_score_)
    best_params_df = pd.DataFrame(grid_search.best_params_, index=[0])
    best_params_df.to_csv("./gridsearchcv_results/usage_example_param_grid.csv", index=False)
    score_df = pd.DataFrame(grid_search.cv_results_)
    score_df.to_csv("./gridsearchcv_results/usage_example_cv_results.csv", index=False)
```

**Usage:**


```python
param_sample_df = pd.read_csv("../data/new_csv/RData_parameters_sample.csv")
slr_rcp26_5step = pd.read_csv("../data/new_csv/slr_rcp26_5yrstep.csv")
gridsearch(param_sample_df, slr_rcp26_5step, "2100")
```

    Fitting 5 folds for each of 4 candidates, totalling 20 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  20 out of  20 | elapsed:  2.7min finished
    

    BEST PARAMETERS:
    {'max_depth': 7, 'max_features': 37, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 300}
    Mean cross-validated score of the best_estimator:  0.95825
    

***
### `slr_forest(feature_list, df, year, max_feat="auto", max_d=None, min_samp_leaf=1, n_estimators=100, min_samples_split=2, print_forest=False)`
Creates a forest using the desired parameters and fits the forest with 60% of the data in the dataframe.  When this function is called, it prints the validation accuracy and training accuracy of the forest.

**Parameters:**
- *feature_list:* list of the input column names as strings

- *df:* dataframe that contains both the input data and the output data, NOT already split into training,
    validation, and testing subsets
- *year:* string of the year to use as the output data

- *max_feat:* integer number of features to consider when determining the best split -- default = "auto" which takes the square root of the total number of features

- *max_d:* the maximum depth of the tree as an integer -- default = None

- *min_samp_leaf:* the minimum integer number of samples required to be in a leaf node -- default = 1
    
- *n_estimators:* the number of trees in the forest as an integer -- default = 100

- *min_samples_split:* the minimum integer number of samples required in a node to be able to split -- default = 2

- *print_forest:* boolean that controls whether the trees in the forest should be printed out in text form

**Returns:** (forest, v_accuracy, t_accuracy)
- forest is the forest that was created and fit with the data
- v_accuracy is the validation accuracy as a decimal value
- t_accuracy is the training accuracy as a decimal value

**Definition:**


```python
def slr_forest(feature_list, df, year, max_feat="auto", max_d=None, min_samp_leaf=1, n_estimators=100,
               min_samples_split=2, print_forest=False):
    # set up subsets
    x = df[feature_list]
    y = df[year]
    x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.4)  # train= 60%, validation + test= 40%
    # split up rest of 40% into validation & test
    x_validation, x_test, y_validation, y_test = train_test_split(x_rest, y_rest,
                                                                  test_size=.5)  # validation= 20%, test= 20%

    # random forest creation
    forest = ensemble.RandomForestClassifier(n_estimators=n_estimators, criterion="entropy", max_features=max_feat,
                                             max_depth=max_d, min_samples_leaf=min_samp_leaf,
                                             min_samples_split=min_samples_split)
    forest = forest.fit(x_train, y_train)

    # print random forest
    if print_forest == True:
        for i in range(0, len(forest.estimators_)):
            print("\nEstimator", i, ":")
            text = tree.export_text(forest.estimators_[i], feature_names=feature_list)
            print(text)

    # random forest validation
    y_predicted = forest.predict(x_validation)
    v_accuracy = metrics.accuracy_score(y_validation, y_predicted)
    print("\nValidation Accuracy =", v_accuracy)

    # random forest training
    y_predicted = forest.predict(x_train)
    t_accuracy = metrics.accuracy_score(y_train, y_predicted)
    print("Training Accuracy =", t_accuracy)

    return forest, v_accuracy, t_accuracy
```

**Usage:**


```python
param_sample_df = pd.read_csv("../data/new_csv/RData_parameters_sample.csv")
slr_rcp26_5step = pd.read_csv("../data/new_csv/slr_rcp26_5yrstep.csv")

slr_classify = classify_data(slr_rcp26_5step)
df_slr_classify = param_sample_df.join(slr_classify, how="outer")
df_slr_classify = df_slr_classify.dropna()
features = param_sample_df.columns.tolist()

forest = slr_forest(features, df_slr_classify, "2100", max_feat="auto", max_d=3, min_samp_leaf=1, n_estimators=5, 
                    min_samples_split=2, print_forest=True)
```

    
    Estimator 0 :
    |--- V0_gsic.slr_brick <= 0.30
    |   |--- V0_simple.slr_brick <= 7.41
    |   |   |--- class: 0.0
    |   |--- V0_simple.slr_brick >  7.41
    |   |   |--- class: 1.0
    |--- V0_gsic.slr_brick >  0.30
    |   |--- alpha.temperature <= 0.98
    |   |   |--- P0_dais.slr_brick <= 0.48
    |   |   |   |--- class: 1.0
    |   |   |--- P0_dais.slr_brick >  0.48
    |   |   |   |--- class: 1.0
    |   |--- alpha.temperature >  0.98
    |   |   |--- Tcrit_dais.slr_brick <= -15.90
    |   |   |   |--- class: 0.0
    |   |   |--- Tcrit_dais.slr_brick >  -15.90
    |   |   |   |--- class: 1.0
    
    
    Estimator 1 :
    |--- slope_dais.slr_brick <= 0.00
    |   |--- S.temperature <= 4.04
    |   |   |--- Tcrit_dais.slr_brick <= -16.07
    |   |   |   |--- class: 1.0
    |   |   |--- Tcrit_dais.slr_brick >  -16.07
    |   |   |   |--- class: 1.0
    |   |--- S.temperature >  4.04
    |   |   |--- offset.Tgav_obs <= -0.04
    |   |   |   |--- class: 1.0
    |   |   |--- offset.Tgav_obs >  -0.04
    |   |   |   |--- class: 1.0
    |--- slope_dais.slr_brick >  0.00
    |   |--- mu_dais.slr_brick <= 10.87
    |   |   |--- a_tee.slr_brick <= 0.12
    |   |   |   |--- class: 1.0
    |   |   |--- a_tee.slr_brick >  0.12
    |   |   |   |--- class: 1.0
    |   |--- mu_dais.slr_brick >  10.87
    |   |   |--- lambda_dais.slr_brick <= 0.01
    |   |   |   |--- class: 1.0
    |   |   |--- lambda_dais.slr_brick >  0.01
    |   |   |   |--- class: 1.0
    
    
    Estimator 2 :
    |--- rho.Tgav_obs <= 0.45
    |   |--- Tcrit_dais.slr_brick <= -15.95
    |   |   |--- diff.temperature <= 2.15
    |   |   |   |--- class: 1.0
    |   |   |--- diff.temperature >  2.15
    |   |   |   |--- class: 1.0
    |   |--- Tcrit_dais.slr_brick >  -15.95
    |   |   |--- b_anto.slr_brick <= 1.25
    |   |   |   |--- class: 1.0
    |   |   |--- b_anto.slr_brick >  1.25
    |   |   |   |--- class: 1.0
    |--- rho.Tgav_obs >  0.45
    |   |--- diff.temperature <= 1.74
    |   |   |--- alpha.temperature <= 1.14
    |   |   |   |--- class: 1.0
    |   |   |--- alpha.temperature >  1.14
    |   |   |   |--- class: 1.0
    |   |--- diff.temperature >  1.74
    |   |   |--- S.temperature <= 3.88
    |   |   |   |--- class: 1.0
    |   |   |--- S.temperature >  3.88
    |   |   |   |--- class: 1.0
    
    
    Estimator 3 :
    |--- S.temperature <= 4.04
    |   |--- S.temperature <= 3.26
    |   |   |--- Tcrit_dais.slr_brick <= -16.33
    |   |   |   |--- class: 1.0
    |   |   |--- Tcrit_dais.slr_brick >  -16.33
    |   |   |   |--- class: 1.0
    |   |--- S.temperature >  3.26
    |   |   |--- Tcrit_dais.slr_brick <= -16.02
    |   |   |   |--- class: 0.0
    |   |   |--- Tcrit_dais.slr_brick >  -16.02
    |   |   |   |--- class: 1.0
    |--- S.temperature >  4.04
    |   |--- Tcrit_dais.slr_brick <= -15.70
    |   |   |--- f0_dais.slr_brick <= 1.40
    |   |   |   |--- class: 0.0
    |   |   |--- f0_dais.slr_brick >  1.40
    |   |   |   |--- class: 0.0
    |   |--- Tcrit_dais.slr_brick >  -15.70
    |   |   |--- beta_simple.slr_brick <= 0.00
    |   |   |   |--- class: 1.0
    |   |   |--- beta_simple.slr_brick >  0.00
    |   |   |   |--- class: 1.0
    
    
    Estimator 4 :
    |--- diff.temperature <= 2.15
    |   |--- slope_dais.slr_brick <= 0.00
    |   |   |--- S.temperature <= 3.60
    |   |   |   |--- class: 1.0
    |   |   |--- S.temperature >  3.60
    |   |   |   |--- class: 1.0
    |   |--- slope_dais.slr_brick >  0.00
    |   |   |--- f0_dais.slr_brick <= 1.14
    |   |   |   |--- class: 1.0
    |   |   |--- f0_dais.slr_brick >  1.14
    |   |   |   |--- class: 1.0
    |--- diff.temperature >  2.15
    |   |--- alpha.temperature <= 0.98
    |   |   |--- alpha.temperature <= 0.88
    |   |   |   |--- class: 1.0
    |   |   |--- alpha.temperature >  0.88
    |   |   |   |--- class: 1.0
    |   |--- alpha.temperature >  0.98
    |   |   |--- sigma.Tgav_obs <= 0.10
    |   |   |   |--- class: 1.0
    |   |   |--- sigma.Tgav_obs >  0.10
    |   |   |   |--- class: 0.0
    
    
    Validation Accuracy = 0.911
    Training Accuracy = 0.9168333333333333
    

***
### `make_forest_and_export(parameter_sample_df, slr_df, yrs_to_output, rcp_str, forest_path, accuracy_path)`
Creates forests for the given years, saves each forest as a file, and saves the validation and training accuracy of each forest in a CSV file.  Once a forest is saved as a file, the function prints the RCP and year, the forest's validation and training accuracies, and the size of file that the forest is saved as.  Each forests is saved as a .joblib file.  For instance, a forest for RCP 8.5 in the year 2075 is saved as rcp85_2075.joblib

**Parameters:**
- *parameter_sample_df:* dataframe of the input feature values

- *slr_df:* dataframe of the output year values

- *yrs_to_output:* list of the years as strings to create and export forests for

- *rcp_str:* RCP name as a string with no spaces (ex: "rcp85")

- *forest_path:* path of the folder to save the forests into (ex: "./forests/")

- *accuracy_path:* path of the folder to save the accuracy CSV files into (ex: "./forests/forest_accuracy/")

**Returns:** None

**Definition:**


```python
def make_forest_and_export(parameter_sample_df, slr_df, yrs_to_output, rcp_str, forest_path, accuracy_path):
    slr_classify = classify_data(slr_df)
    df_slr_classify = parameter_sample_df.join(slr_classify, how="outer")
    df_slr_classify = df_slr_classify.dropna()
    features = parameter_sample_df.columns.tolist()
    for yr in yrs_to_output:
        print("\n", rcp_str, yr)
        forest, v_accuracy, t_accuracy = slr_forest(features, df_slr_classify, yr, max_feat=MAX_FEATURES,
                                                    max_d=MAX_DEPTH, min_samp_leaf=MIN_SAMPLES_LEAF,
                                                    min_samples_split= MIN_SAMPLES_SPLIT, n_estimators=N_ESTIMATORS)
        forest_file_path = forest_path + rcp_str + "_" + yr + ".joblib"
        joblib.dump(forest, forest_file_path, compress=3)
        print(f"Compressed Random Forest: {np.round(os.path.getsize(forest_file_path) / 1024 / 1024, 2)} MB")
        accuracy_df = pd.DataFrame({"Validation Accuracy": [v_accuracy], "Training Accuracy": [t_accuracy]})
        accuracy_file_path = accuracy_path + rcp_str + "_" + yr + "_accuracy.csv"
        accuracy_df.to_csv(accuracy_file_path, index=False)
```

**Usage:**


```python
param_sample_df = pd.read_csv("../data/new_csv/RData_parameters_sample.csv")
slr_rcp26_5step = pd.read_csv("../data/new_csv/slr_rcp26_5yrstep.csv")
slr_rcp85_5step = pd.read_csv("../data/new_csv/slr_rcp85_5yrstep.csv")
yrs_5step = slr_rcp26_5step.columns.tolist()

make_forest_and_export(param_sample_df, slr_rcp85_5step, yrs_5step, "rcp85", "./forests/", 
                       "./forests/forest_accuracy/")
```

    
     rcp85 2020
    
    Validation Accuracy = 0.8995
    Training Accuracy = 0.9006666666666666
    Compressed Random Forest: 0.13 MB
    
     rcp85 2025
    
    Validation Accuracy = 0.8995
    Training Accuracy = 0.9006666666666666
    Compressed Random Forest: 0.13 MB
    
     rcp85 2030
    
    Validation Accuracy = 0.8995
    Training Accuracy = 0.9073333333333333
    Compressed Random Forest: 0.13 MB
    
     rcp85 2035
    
    Validation Accuracy = 0.9015
    Training Accuracy = 0.913
    Compressed Random Forest: 0.13 MB
    
     rcp85 2040
    
    Validation Accuracy = 0.916
    Training Accuracy = 0.9288333333333333
    Compressed Random Forest: 0.13 MB
    
     rcp85 2045
    
    Validation Accuracy = 0.918
    Training Accuracy = 0.9413333333333334
    Compressed Random Forest: 0.13 MB
    
     rcp85 2050
    
    Validation Accuracy = 0.9165
    Training Accuracy = 0.936
    Compressed Random Forest: 0.13 MB
    
     rcp85 2055
    
    Validation Accuracy = 0.911
    Training Accuracy = 0.9276666666666666
    Compressed Random Forest: 0.13 MB
    
     rcp85 2060
    
    Validation Accuracy = 0.9055
    Training Accuracy = 0.9253333333333333
    Compressed Random Forest: 0.13 MB
    
     rcp85 2065
    
    Validation Accuracy = 0.9075
    Training Accuracy = 0.9276666666666666
    Compressed Random Forest: 0.13 MB
    
     rcp85 2070
    
    Validation Accuracy = 0.9175
    Training Accuracy = 0.9241666666666667
    Compressed Random Forest: 0.13 MB
    
     rcp85 2075
    
    Validation Accuracy = 0.9175
    Training Accuracy = 0.9256666666666666
    Compressed Random Forest: 0.13 MB
    
     rcp85 2080
    
    Validation Accuracy = 0.9115
    Training Accuracy = 0.9225
    Compressed Random Forest: 0.13 MB
    
     rcp85 2085
    
    Validation Accuracy = 0.914
    Training Accuracy = 0.923
    Compressed Random Forest: 0.13 MB
    
     rcp85 2090
    
    Validation Accuracy = 0.9155
    Training Accuracy = 0.9266666666666666
    Compressed Random Forest: 0.13 MB
    
     rcp85 2095
    
    Validation Accuracy = 0.927
    Training Accuracy = 0.9238333333333333
    Compressed Random Forest: 0.13 MB
    
     rcp85 2100
    
    Validation Accuracy = 0.9315
    Training Accuracy = 0.9281666666666667
    Compressed Random Forest: 0.12 MB
    
     rcp85 2105
    
    Validation Accuracy = 0.9255
    Training Accuracy = 0.9323333333333333
    Compressed Random Forest: 0.13 MB
    
     rcp85 2110
    
    Validation Accuracy = 0.9245
    Training Accuracy = 0.9253333333333333
    Compressed Random Forest: 0.13 MB
    
     rcp85 2115
    
    Validation Accuracy = 0.9195
    Training Accuracy = 0.9323333333333333
    Compressed Random Forest: 0.13 MB
    
     rcp85 2120
    
    Validation Accuracy = 0.9245
    Training Accuracy = 0.933
    Compressed Random Forest: 0.12 MB
    
     rcp85 2125
    
    Validation Accuracy = 0.9265
    Training Accuracy = 0.9331666666666667
    Compressed Random Forest: 0.12 MB
    
     rcp85 2130
    
    Validation Accuracy = 0.923
    Training Accuracy = 0.9356666666666666
    Compressed Random Forest: 0.13 MB
    
     rcp85 2135
    
    Validation Accuracy = 0.9215
    Training Accuracy = 0.9335
    Compressed Random Forest: 0.12 MB
    
     rcp85 2140
    
    Validation Accuracy = 0.9275
    Training Accuracy = 0.9378333333333333
    Compressed Random Forest: 0.12 MB
    
     rcp85 2145
    
    Validation Accuracy = 0.9265
    Training Accuracy = 0.9378333333333333
    Compressed Random Forest: 0.12 MB
    
     rcp85 2150
    
    Validation Accuracy = 0.925
    Training Accuracy = 0.9381666666666667
    Compressed Random Forest: 0.13 MB
    

***
### `load_forests(year_list, rcp_str)`
Loads forests from a saved forest file into a list.

**Parameters:**
- *year_list:* list of years (string or int) to load forests for
- *rcp_str:* RCP name as a string with no spaces (ex: "rcp85")

**Returns:** a list of the loaded forests

**Definition:**


```python
def load_forests(year_list, rcp_str):
    forests = []
    for yr in year_list:
        path = "./forests/" + rcp_str + "_" + str(yr) + ".joblib"
        forests.append(joblib.load(path))
    return forests
```

**Usage:**


```python
rcp26_forest_list = load_forests(["2050", "2100", "2150"], "rcp26")

forest_rcp26_2100 = rcp26_forest_list[1]
print(forest_rcp26_2100.classes_)
```

    ['high' 'low']
    

***
### `find_forest_splits(forest, feature_names, feature, firstsplit=False)`
Determines the split values from all the trees in the forest for the splits of a specific feature.

**Parameters:**
- *forest:* a forest that has already been fit with training data
- *feature_names:* list of all the feature names from the dataframe used to fit the forest
- *feature:* string of specific feature to find the splits for
- *firstsplit:* boolean value
    - True returns values from only the first occurrence of the feature's split in each tree
    - False returns values from all of the splits of the specified feature from each tree

**Returns:** a list of lists, each inner list contains the desired feature's split values for a tree
- note: the length of the outer list is equal to the number of trees in the forest

**Definition:**


```python
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
```

**Usage:**


```python
param_sample_df = pd.read_csv("../data/new_csv/RData_parameters_sample.csv")
features = param_sample_df.columns.tolist()

forest_rcp26_2100 = load_forests(["2100"], "rcp26")[0]

find_forest_splits(forest_rcp26_2100, features, "S.temperature", firstsplit=True)
```




    [[3.2021313905715942],
     [3.7019248008728027],
     [3.5886601209640503],
     [3.464869737625122],
     [4.044274806976318],
     [3.6618326902389526],
     [3.310426354408264],
     [3.556783437728882],
     [4.046891689300537],
     [3.6310532093048096],
     [3.300024628639221],
     [3.45815646648407],
     [3.847332000732422],
     [3.0708497762680054],
     [3.883669376373291],
     [3.3435288667678833],
     [3.702062964439392],
     [3.8948192596435547],
     [3.8153610229492188],
     [3.815734624862671],
     [3.556783437728882],
     [3.6483975648880005],
     [3.2161539793014526],
     [3.574170231819153],
     [3.621071457862854],
     [3.6313672065734863],
     [3.0711801052093506],
     [3.702062964439392],
     [3.6207584142684937],
     [4.089716911315918],
     [4.043443202972412],
     [3.464869737625122],
     [3.350989580154419],
     [3.3265045881271362],
     [3.814141869544983],
     [3.4722864627838135],
     [3.4639962911605835],
     [3.2161539793014526],
     [3.5844130516052246],
     [3.6298818588256836],
     [3.1686161756515503],
     [3.464869737625122],
     [3.269200921058655],
     [3.5512564182281494],
     [3.348569631576538],
     [3.7019248008728027],
     [3.847299814224243],
     [3.20207679271698],
     [3.0711801052093506],
     [4.045417070388794],
     [3.8122527599334717],
     [3.6203389167785645],
     [3.558834671974182],
     [3.8948192596435547],
     [3.814141869544983],
     [3.3450424671173096],
     [3.6429741382598877],
     [3.6226603984832764],
     [3.310426354408264],
     [3.3275870084762573],
     [3.515202760696411],
     [3.1560885906219482],
     [3.7616225481033325],
     [3.20207679271698],
     [3.2161539793014526],
     [3.846420645713806],
     [3.6310532093048096],
     [3.702062964439392],
     [3.464869737625122],
     [3.054978370666504],
     [3.5489728450775146],
     [3.2021313905715942],
     [3.661621332168579],
     [3.5513330698013306],
     [3.202032446861267],
     [3.713188648223877],
     [3.2562432289123535],
     [3.1560885906219482],
     [3.815734624862671],
     [3.2161539793014526],
     [4.108471870422363],
     [3.7347341775894165],
     [3.3265045881271362],
     [3.7019248008728027],
     [3.549881935119629],
     [3.8974398374557495],
     [3.4638423919677734],
     [3.6483975648880005],
     [3.3267908096313477],
     [3.4741379022598267],
     [3.8153610229492188],
     [3.346712589263916],
     [3.2161539793014526],
     [3.4639962911605835],
     [3.3265045881271362],
     [3.45815646648407],
     [3.558834671974182],
     [3.8803112506866455],
     [4.869840145111084],
     [3.707209825515747]]



***
### `split_stats(split_list, split_feature)`
Calculates the minimum, 1st quartile, median, mean, 3rd quartile, and maximum of the splits in the split_list.  These statistics are also printed when this function is called.

**Parameters:**
- *split_list:* a list of lists, each inner list contains the desired feature's split values for a tree
- *split_feature:* string of the feature whose split values are in split_list

**Returns:** splitdf, quantiles_list 
- splitdf is a pandas dataframe created from split_list.  Each row of the dataframe is a different tree in the forest.
- quantiles_list is a list of the calculated statistics of split_list.  The order of the list is [min, Q1, median, Q3, max, mean]

**Definition:**


```python
def split_stats(split_list, split_feature):
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
```

**Usage:**


```python
param_sample_df = pd.read_csv("../data/new_csv/RData_parameters_sample.csv")
features = param_sample_df.columns.tolist()
forest_rcp26_2100 = load_forests(["2100"], "rcp26")[0]

split_list = find_forest_splits(forest_rcp26_2100, features, "S.temperature")

splitdf, quantiles_list = split_stats(split_list, "S.temperature")
print("\nQuantiles list:", quantiles_list)
splitdf.head()
```

    
    Mean of S.temperature splits: 4.1179916882047465
    Minimum of S.temperature splits: 2.6955171823501587
    Q1 of S.temperature splits: 3.300024628639221
    Median of S.temperature splits: 3.820434808731079
    Q3 of S.temperature splits: 4.675499439239502
    Maximum of S.temperature splits: 8.723577976226807
    
    Quantiles list: [2.6955171823501587, 3.300024628639221, 3.820434808731079, 4.675499439239502, 8.723577976226807, 4.1179916882047465]
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3.202131</td>
      <td>4.123497</td>
      <td>4.374663</td>
      <td>5.616092</td>
      <td>4.375769</td>
      <td>6.598045</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3.701925</td>
      <td>4.183320</td>
      <td>4.615553</td>
      <td>4.907348</td>
      <td>6.358938</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.588660</td>
      <td>4.044196</td>
      <td>4.981603</td>
      <td>6.253417</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3.464870</td>
      <td>2.833562</td>
      <td>4.446724</td>
      <td>5.171238</td>
      <td>6.570989</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>4.044275</td>
      <td>3.083898</td>
      <td>2.820725</td>
      <td>3.085438</td>
      <td>6.376080</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Based on `splitdf` from this example, the first tree in the forest splits on S.temperature six times, the second tree splits on S.temperature five times, etc.

***
### `perform_splits(forest, feature_list, split_feature)`
Runs the `find_forest_splits()` function and the `split_stats()` function.  In other words, this function find splits of a specific feature and calculates the minimum, 1st quartile, median, mean, 3rd quartile, and maximum of the splits.  These statistics are also printed when this function is called.

**Parameters:**
- *forest:* a forest that has already been fit with training data
- *feature_list:* list of all the feature names from the dataframe used to fit the forest
- *split_feature:* string of specific feature to find the splits for

**Returns:** split_df, all, first_only
- splitdf is a pandas dataframe created from split_list.  Each row of the dataframe is a different tree in the forest.
- all is a list of the calculated statistics for all the split values of split_feature.  The order of the list is [min, Q1, median, Q3, max, mean]
- first_only is a list of the calculated statistics for only the first split values of split_feature in each tree.  The order of the list is [min, Q1, median, Q3, max, mean]

**Definition:**


```python
def perform_splits(forest, feature_list, split_feature):
    # find forest splits
    split_list = find_forest_splits(forest, feature_list, split_feature)
    empty = True
    for list in split_list:
        if len(list) != 0:
            empty = False
    if empty:
        return None, None, None
    else:
        split_df, all = split_stats(split_list, split_feature)
        first_split_list = find_forest_splits(forest, feature_list, split_feature, firstsplit=True)
        first_only = split_stats(first_split_list, split_feature)[1]
        return split_df, all, first_only
```

**Usage:**


```python
param_sample_df = pd.read_csv("../data/new_csv/RData_parameters_sample.csv")
features = param_sample_df.columns.tolist()
forest_rcp26_2100 = load_forests(["2100"], "rcp26")[0]

split_df, all_list, first_only_list = perform_splits(forest_rcp26_2100, features, "S.temperature")
print("\nStats list of all S.temperature split values:", all_list)
print("Stats list of first S.temperature split values:", first_only_list)
split_df.head()
```

    
    Mean of S.temperature splits: 4.1179916882047465
    Minimum of S.temperature splits: 2.6955171823501587
    Q1 of S.temperature splits: 3.300024628639221
    Median of S.temperature splits: 3.820434808731079
    Q3 of S.temperature splits: 4.675499439239502
    Maximum of S.temperature splits: 8.723577976226807
    
    Mean of S.temperature splits: 3.5580745589733125
    Minimum of S.temperature splits: 3.054978370666504
    Q1 of S.temperature splits: 3.326719254255295
    Median of S.temperature splits: 3.557809054851532
    Q3 of S.temperature splits: 3.7087045311927795
    Maximum of S.temperature splits: 4.869840145111084
    
    Stats list of all S.temperature split values: [2.6955171823501587, 3.300024628639221, 3.820434808731079, 4.675499439239502, 8.723577976226807, 4.1179916882047465]
    Stats list of first S.temperature split values: [3.054978370666504, 3.326719254255295, 3.557809054851532, 3.7087045311927795, 4.869840145111084, 3.5580745589733125]
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3.202131</td>
      <td>4.123497</td>
      <td>4.374663</td>
      <td>5.616092</td>
      <td>4.375769</td>
      <td>6.598045</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3.701925</td>
      <td>4.183320</td>
      <td>4.615553</td>
      <td>4.907348</td>
      <td>6.358938</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.588660</td>
      <td>4.044196</td>
      <td>4.981603</td>
      <td>6.253417</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3.464870</td>
      <td>2.833562</td>
      <td>4.446724</td>
      <td>5.171238</td>
      <td>6.570989</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>4.044275</td>
      <td>3.083898</td>
      <td>2.820725</td>
      <td>3.085438</td>
      <td>6.376080</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



***
### `splits_table(forest, feature_names)`
Creates a dataframe that shows what fraction of the time in this forest that each feature is used at each split point.

**Parameters:**
- *forest:* a forest that has already been fit with training data
- *feature_names:* list of all the feature names from the dataframe used to fit the forest

**Returns:** dataframe that shows what fraction of the time in this forest that each feature is used at each split point
- note: the rows of the dataframe are the split names (p, l, lr, etc) and the columns are the feature names

**Definition:**


```python
def splits_table(forest, feature_names):
    features = feature_names.copy()
    dict_list=[]
    for i in range(len(forest.estimators_)):
        estimator = forest.estimators_[i]
        tree_feature = estimator.tree_.feature
        feature_new = []
        for node in tree_feature:
            if node == -2:
                feature_new.append('leaf')
            else:
                feature_new.append(features[node])
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

    # for trees w/ max depth = 5
    nodes = ["p", "l", "ll", "lll", "llll", "lllll", "llllr", "lllr", "lllrl", "lllrr", "llr", "llrl", "llrll", "llrlr",
             "llrr", "llrrl", "llrrr", "lr", "lrl", "lrll", "lrlll", "lrllr", "lrlr", "lrlrl", "lrlrr", "lrr", "lrrl",
             "lrrll", "lrrlr", "lrrr", "lrrrl", "lrrrr", "r", "rl", "rll", "rlll", "rllll", "rlllr", "rllr", "rllrl",
             "rllrr", "rlr", "rlrl", "rlrll", "rlrlr", "rlrr", "rlrrl", "rlrrr", "rr", "rrl", "rrll", "rrlll", "rrllr",
             "rrlr", "rrlrl", "rrlrr", "rrr", "rrrl", "rrrll", "rrrlr", "rrrr", "rrrrl", "rrrrr"]

    row_list=[]
    features.append("leaf")

    for node in nodes:
        feature_list = []
        for i in range (len(dict_list)):
            if node in dict_list[i]:
                feature_list.append(dict_list[i][node])
        feature_sums = {}
        for name in features:
            feature_sums[name] = 0
        for f in feature_list:
            feature_sums[f] += 1
        tot=len(feature_list)
        feature_fractions=[]
        for f in feature_sums.keys():
            feature_fractions.append(feature_sums[f] / tot)
        row_list.append(feature_fractions)

    df = pd.DataFrame(row_list, index=nodes, columns=features)
    return df
```

**Usage:**


```python
param_sample_df = pd.read_csv("../data/new_csv/RData_parameters_sample.csv")
features = param_sample_df.columns.tolist()
forest_rcp26_2100 = load_forests(["2100"], "rcp26")[0]

splits_table_df = splits_table(forest_rcp26_2100, features)
splits_table_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>S.temperature</th>
      <th>diff.temperature</th>
      <th>alpha.temperature</th>
      <th>beta0_gsic.slr_brick</th>
      <th>V0_gsic.slr_brick</th>
      <th>n_gsic.slr_brick</th>
      <th>Gs0_gsic.slr_brick</th>
      <th>a_tee.slr_brick</th>
      <th>a_simple.slr_brick</th>
      <th>b_simple.slr_brick</th>
      <th>...</th>
      <th>h0_dais.slr_brick</th>
      <th>c_dais.slr_brick</th>
      <th>b0_dais.slr_brick</th>
      <th>slope_dais.slr_brick</th>
      <th>lambda_dais.slr_brick</th>
      <th>Tcrit_dais.slr_brick</th>
      <th>offset.ocheat_obs</th>
      <th>sigma.ocheat_obs</th>
      <th>rho.ocheat_obs</th>
      <th>leaf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>p</td>
      <td>0.440000</td>
      <td>0.08</td>
      <td>0.310000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.100000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>l</td>
      <td>0.440000</td>
      <td>0.04</td>
      <td>0.170000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.290000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>ll</td>
      <td>0.390000</td>
      <td>0.04</td>
      <td>0.120000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.060000</td>
      <td>0.020000</td>
      <td>0.230000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.020000</td>
    </tr>
    <tr>
      <td>lll</td>
      <td>0.224490</td>
      <td>0.00</td>
      <td>0.122449</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.020408</td>
      <td>0.010204</td>
      <td>0.000000</td>
      <td>0.010204</td>
      <td>0.010204</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.061224</td>
      <td>0.040816</td>
      <td>0.010204</td>
      <td>0.153061</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.193878</td>
    </tr>
    <tr>
      <td>llll</td>
      <td>0.113924</td>
      <td>0.00</td>
      <td>0.037975</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.012658</td>
      <td>0.037975</td>
      <td>0.000000</td>
      <td>0.012658</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.012658</td>
      <td>0.012658</td>
      <td>0.012658</td>
      <td>0.012658</td>
      <td>0.050633</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.531646</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>



Based on `splits_table_df` from this example, S.temperature is the parent node (row p) 44% of the time and alpha.temperature is the leftmost node at a depth of 3 (row ll) 12% of the time.

***
### `tree_splits(param_sample_df, response, rcp, forests_list, year_list, folder_path)`
Runs the `perform_splits()` function and the `splits_table()` function for each forest, and creates a plot of the feature importances of each forest in the same pop-up.  The S.temperature split values are saved into a separate CSV for each forest, the split breakdown from the `splits_table()` function are saved into a separate CSV for each forest, the statistics of all the S.temperature splits for each forest/year are saved into one CSV, and the statistics of the first split value of the S.temeprature splits for each forest/year are saved into one CSV.

**Parameters:**
- *param_sample_df:* dataframe of the input feature values
- *response:* "SLR" or "Tgav"
- *rcp:* RCP name as a string (ex: "RCP 8.5")
- *forests_list:* a list of already fit forests for this function to be perfomred on
- *year_list:* list of the years (as integers) that correspond to years of the forests in forest_list
- *folder_path:* path to the folder where the CSV files will be saved

**Returns:** None

**Definition:**


```python
def tree_splits(param_sample_df, response, rcp, forests_list, year_list, folder_path):
    fig, axs = plt.subplots(1, len(year_list))
    features = param_sample_df.columns.tolist()
    first_quartile_data = []
    all_quartile_data = []
    rcp_no_space = rcp.replace(" ", "")
    rcp_no_space_no_period = rcp_no_space.replace(".", "")

    for j in range(len(year_list)):
        yr = str(year_list[j])
        all_label = response + " " + yr + " all splits"
        first_label = response + " " + yr + " first split"
        all = (all_label,)
        first = (first_label,)
        forest = forests_list[j]
        split_df, all_quartiles, first_quartiles=perform_splits(forest, features,"S.temperature")
        if isinstance(split_df, pd.DataFrame):
            pass
        else:
            continue

        split_file_path = folder_path + rcp_no_space_no_period + "_" + yr + "_splits.csv"
        split_df.to_csv(split_file_path, index=False)
        for n in range(len(all_quartiles)):
            all += (all_quartiles[n],)
            first += (first_quartiles[n],)
        first_quartile_data.append(first)
        all_quartile_data.append(all)

        table_df = splits_table(forest, features)
        split_table_file_path = folder_path + rcp_no_space_no_period + "_" + yr + "_split_table.csv"
        table_df.to_csv(split_table_file_path, index=True)

        # importances plot
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        importances_list = []
        std_list = []
        features_list = []
        for idx in indices:
            if importances[idx] > .025:
                importances_list.append(importances[idx])
                std_list.append(std[idx])
                features_list.append(features[idx])
        axs[j].bar(range(len(importances_list)), importances_list, color="tab:blue",
                       yerr=std_list, align="center")
        title = yr
        axs[j].set_title(title)
        axs[j].set_xticks(range(len(importances_list)))
        axs[j].set_xticklabels(features_list, rotation=90)
        axs[j].set_ylim(top=1.0)
        axs[j].set_ylim(bottom= 0.0)
    main_title = response + " " + rcp + " Feature Importances"
    fig.suptitle(main_title, fontsize=15)
    fig.text(0.52, 0.04, 'Features', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Relative Importance', va='center', rotation='vertical', fontsize=12)
    plt.show()

    df_first_quartile = pd.DataFrame(first_quartile_data, columns=["Name", "0%", "25%", "50%", "75%", "100%", "Mean"])
    first_file_path = folder_path + rcp_no_space_no_period + "_first_splits.csv"
    df_first_quartile.to_csv(first_file_path, index=False)

    df_all_quartile = pd.DataFrame(all_quartile_data, columns=["Name", "0%", "25%", "50%", "75%", "100%", "Mean"])
    all_file_path = folder_path + rcp_no_space_no_period + "_all_splits.csv"
    df_all_quartile.to_csv(all_file_path, index=False)
```

**Usage:**


```python
param_sample_df = pd.read_csv("../data/new_csv/RData_parameters_sample.csv")
forest_2025 = joblib.load("./forests/rcp85_2025.joblib")
forest_2100 = joblib.load("./forests/rcp85_2100.joblib")
path = "../data/new_csv/SLR_splits/classification_forest/"

tree_splits(param_sample_df, "SLR", "RCP 8.5", [forest_2025, forest_2100], [2025, 2100], path)
```

    
    Mean of S.temperature splits: 4.022029463439546
    Minimum of S.temperature splits: 1.7192018628120422
    Q1 of S.temperature splits: 2.7376991510391235
    Median of S.temperature splits: 3.4285292625427246
    Q3 of S.temperature splits: 5.125203549861908
    Maximum of S.temperature splits: 9.687826156616211
    
    Mean of S.temperature splits: 3.3464225000805325
    Minimum of S.temperature splits: 2.121634364128113
    Q1 of S.temperature splits: 2.7627662420272827
    Median of S.temperature splits: 3.2573286294937134
    Q3 of S.temperature splits: 3.483919382095337
    Maximum of S.temperature splits: 9.226417064666748
    
    Mean of S.temperature splits: 4.234018252470952
    Minimum of S.temperature splits: 2.1573214530944824
    Q1 of S.temperature splits: 2.956520676612854
    Median of S.temperature splits: 3.7053385972976685
    Q3 of S.temperature splits: 5.2994232177734375
    Maximum of S.temperature splits: 9.239834785461426
    
    Mean of S.temperature splits: 3.460056948661804
    Minimum of S.temperature splits: 2.157975196838379
    Q1 of S.temperature splits: 2.989344835281372
    Median of S.temperature splits: 3.298148214817047
    Q3 of S.temperature splits: 3.644248813390732
    Maximum of S.temperature splits: 9.081339836120605
    


![png](output_46_1.png)


***
### `feature_color_dict(features_list)`
Assigns each feature in the features_list a color and stores it in a dictionary.

**Parameters:**
- *features_list:* list of feature names to associate colors with

**Returns:** color_dict 
- color_dict is a dictionary where the keys are the features and the values are the colors


```python
def feature_color_dict(features_list):
    color_map = pylab.get_cmap('terrain')
    color_dict = {}
    for i in range(len(features_list)):
        color = color_map(i/len(features_list))
        color_dict[features_list[i]] = color
    return color_dict
```

**Usage:** Used within the `slr_stacked_importances_plot()` function

***
### `slr_stacked_importances_plot(param_sample_df, rcp26_forest_list, rcp85_forest_list, years, importance_threshold)`
Create a stacked histogram of the feature importances over time for each RCP.  The x-axis contains the years and the y-axis is the feature importance.  The stacked bars for each year shows the breakdown of the feature importances of the forest for that year.

**Parameters:**
- *param_sample_df:* dataframe of the input feature values

- *rcp26_forest_list:* a list of already fit forests created from RCP 2.6 data

- *rcp85_forest_list:* a list of already fit forests created from RCP 8.5 data

- *years:* list of the years (as strings) that correspond to years of the forests in rcp26_forest_list and rcp85_forest_list

- *importance_threshold:* decimal value where every feature whose feature importance is under this threshold will be added into the "Other" category on the plot

**Returns:** None

**Definition:**


```python
def slr_stacked_importances_plot(param_sample_df, rcp26_forest_list, rcp85_forest_list, years, importance_threshold):
    features = param_sample_df.columns.tolist()
    name = ["RCP2.6", "RCP8.5"]
    forest_masterlist=[rcp26_forest_list, rcp85_forest_list]
    importances_info_list = []

    for i in range(len(forest_masterlist)):
        forest_list = forest_masterlist[i]
        importances_info = {}
        for j in range(len(forest_list)):
            forest = forest_list[j]

            # stacked importances dictionary
            importances = forest.feature_importances_
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
                            list = [0 for n in range(0, j)]
                            list.append(importance)
                            importances_info[feature] = list
                        sum += importance
                importances_info['Other'].append(1 - sum)
                for f in importances_info:
                    if len(importances_info[f]) < (j + 1):
                        importances_info[f].append(0)
        importances_info_list.append(importances_info)

    # set color for each feature
    features_on_plot = []
    for importances_info in importances_info_list:
        for feature in importances_info:
            if feature not in features_on_plot:
                features_on_plot.append(feature)
    color_dict = feature_color_dict(features_on_plot)

    # plotting
    fig, axs = plt.subplots(2, 1)
    for i in range(len(importances_info_list)):
        importances_info = importances_info_list[i]
        # stacked importances plot
        x = np.arange(len(years))
        bottom = None
        for feature in importances_info:
            color = color_dict[feature]
            if feature == "Other":
                pass
            else:
                if bottom is None:
                    axs[i].bar(x, importances_info[feature], label=feature, color=color)
                    bottom = np.array(importances_info[feature])
                else:
                    axs[i].bar(x, importances_info[feature], bottom=bottom, label=feature, color=color)
                    bottom += np.array(importances_info[feature])
        percent_label = "Other (< " + str(importance_threshold*100) + "%)"
        axs[i].bar(x, importances_info["Other"], bottom=bottom, label=percent_label)
        axs[i].set_ylabel("Relative Importances")
        yrs_10=[]
        for yr in years:
            if int(yr) % 10 == 0:
                yrs_10.append(yr)
            else:
                yrs_10.append("")
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(yrs_10)
        axs[i].set_xlabel('Year')
        title = "SLR " + name[i] + " Feature Importances"
        axs[i].set_title(title, fontsize=14)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.figlegend(by_label.values(), by_label.keys(), bbox_to_anchor=(.98, .62))
    plt.show()
```

**Usage:**


```python
param_sample_df = pd.read_csv("../data/new_csv/RData_parameters_sample.csv")
rcp26_forest_list = load_forests(yrs_rcp26, "rcp26")
rcp85_forest_list = load_forests(yrs_rcp85, "rcp85")

list_5_yrs = []
for yr in range(2020, 2151, 5):
    list_5_yrs.append(str(yr))


slr_stacked_importances_plot(param_sample_df, rcp26_forest_list, rcp85_forest_list, list_5_yrs, .05)  #threshold = 5%
```


![png](output_53_0.png)


***
### `Stemp_histograms(year_list, rcp, first_only=False)`
Opens the saved CSV files of the S.temperature splits for each year in the year_list and creates histograms of the S.temperature splits in each tree for each year

**Parameters:**
- *year_list:* list of the years (string or int) for the dataframes in split_df_list

- *rcp:* RCP name as a string (ex: "RCP 8.5")

- *first_only:* boolean that controls whether to only plot the values of the first S.temperture split in the trees

**Returns:** None

**Definition:**


```python
def Stemp_histograms(year_list, rcp, first_only=False):
    rcp_no_space = rcp.replace(" ", "")
    rcp_no_space_no_period = rcp_no_space.replace(".", "")
    split_df_list = []
    for yr in year_list:
        file_path = "../data/new_csv/SLR_splits/classification_forest/" + rcp_no_space_no_period + "_" + str(yr) \
                    + "_splits.csv"
        df = pd.read_csv(file_path)
        split_df_list.append(df)

    split_list = []
    if first_only is True:
        for df in split_df_list:
            split_list.append(df["0"].dropna().values.tolist())
    else:
        for df in split_df_list:
            split_list.append(df.stack().tolist())

    fig, axs = plt.subplots(1, len(year_list))
    i = 0
    bin_seq=np.arange(0, 10, step=.5)
    for i in range(len(split_list)):
        data = split_list[i]
        yr = year_list[i]
        axs[i].hist(data, bins=bin_seq, edgecolor='white')
        axs[i].set_title(yr)
        if first_only is True:
            axs[i].set_ylim(top=500)
        else:
            axs[i].set_ylim(top=1000)
        axs[i].set_ylim(bottom=0)
        axs[i].set_xlim(right=10)
        axs[i].set_xlim(left=1)
        i += 1
    main_title = "SLR " + rcp + " Histogram of S.temperature Split Values"
    if first_only is True:
        main_title += " (First Split Only)"
    fig.suptitle(main_title)
    fig.text(0.52, 0.04, 'S.temperature Split', ha='center')
    fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')
    plt.show()
```

**Usage:**


```python
Stemp_histograms([2050, 2100, 2150], "RCP 8.5")
```


![png](output_57_0.png)


***
### `Stemp_max_split_histogram(year_list, rcp)`
Opens the saved CSV files of the S.temperature splits for each year in the year_list and creates histograms of the highest S.temperature split in each tree for each year

**Parameters:**
- *year_list:* list of the years (string or int) for the dataframes in split_df_list

- *rcp:* RCP name as a string (ex: "RCP 8.5")

**Returns:** None
 
**Definition:**


```python
def Stemp_max_split_histogram(year_list, rcp):
    rcp_no_space = rcp.replace(" ", "")
    rcp_no_space_no_period = rcp_no_space.replace(".", "")
    df_dict={}
    for yr in year_list:
        file_path = "../data/new_csv/SLR_splits/classification_forest/" + rcp_no_space_no_period + "_" + str(yr) \
                    + "_splits.csv"
        df = pd.read_csv(file_path)
        max_list = df.max(axis=1).tolist()
        df_dict[str(yr)] = max_list

    fig, axs = plt.subplots(1, len(year_list))
    i = 0
    bin_seq = np.arange(0, 10, step=.5)
    for yr in df_dict:
        axs[i].hist(df_dict[yr], bins=bin_seq, edgecolor='white')
        axs[i].set_title(yr)
        axs[i].set_ylim(bottom=0)
        axs[i].set_ylim(top=200)
        axs[i].set_xlim(right=10)
        axs[i].set_xlim(left=1)
        i += 1
    main_title = "SLR " + rcp + " Histogram of Highest S.temperature Split Values of each Tree"
    fig.suptitle(main_title)
    fig.text(0.52, 0.04, 'S.temperature Split', ha='center')
    fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')
    plt.show()
```

**Usage:**


```python
Stemp_max_split_histogram([2100, 2150], "RCP 8.5")
```


![png](output_61_0.png)


***
### `Stemp_boxplots(year_list, rcp, first_only=False, show_outliers=True)`
Opens the saved CSV files of the S.temperature splits for each year in the year_list and creates a plot of boxplots of the S.temperature splits

**Parameters:**
- *year_list:* list of the years (string or int) for the dataframes in split_df_list

- *rcp:* RCP name as a string (ex: "RCP 8.5")

- *first_only:* boolean that controls whether to only plot the values of the first S.temperture split in the trees

- *show_outliers:* boolean that controls whether to show outliers on the plot

**Returns:** None

**Definition:**


```python
def Stemp_boxplots(year_list, rcp, first_only=False, show_outliers=True):
    rcp_no_space = rcp.replace(" ", "")
    rcp_no_space_no_period = rcp_no_space.replace(".", "")
    split_df_list = []
    for yr in year_list:
        file_path = "../data/new_csv/SLR_splits/classification_forest/" + rcp_no_space_no_period + "_" + str(yr) \
                    + "_splits.csv"
        df = pd.read_csv(file_path)
        split_df_list.append(df)

    split_list = []
    if first_only is True:
        split_str = "(First Split Only)"
        for df in split_df_list:
            split_list.append(df["0"].dropna().values.tolist())
    else:
        split_str = ""
        for df in split_df_list:
            split_list.append(df.stack().tolist())
    fig, ax = plt.subplots()
    ax.boxplot(split_list, showfliers=show_outliers, patch_artist=True, medianprops=dict(color="black"),
               flierprops=dict(markeredgecolor='silver'), labels=[str(yr) for yr in year_list])
    title = "SLR " + rcp + " Boxplots of S.temperature Split Values " + split_str
    plt.ylabel("S.temperature Split Value")
    plt.xlabel("Year")
    plt.title(title, fontsize=15)
    plt.grid(b=True, axis='y', color='gray')
    plt.show()
```

**Usage:**


```python
list_20_yrs = []
for yr in range(2020, 2151, 20):
    list_20_yrs.append(yr)
Stemp_boxplots(list_20_yrs, "RCP 8.5", first_only=True, show_outliers=False)
```


![png](output_65_0.png)


***
### `Stemp_max_split_boxplots(year_list, rcp, show_outliers=True)`
Opens the saved CSV files of the S.temperature splits for each year in the year_list and creates boxplots of the highest S.temperature split in each tree for each year.

**Parameters:**
- *year_list:* list of the years (string or int) for the dataframes in split_df_list

- *rcp:* RCP name as a string (ex: "RCP 8.5")

- *show_outliers:* boolean that controls whether to show outliers on the plot

**Returns:** None

**Definition:**


```python
def Stemp_max_split_boxplots(year_list, rcp, show_outliers=True):
    rcp_no_space = rcp.replace(" ", "")
    rcp_no_space_no_period = rcp_no_space.replace(".", "")
    split_lists=[]
    for yr in year_list:
        file_path = "../data/new_csv/SLR_splits/classification_forest/" + rcp_no_space_no_period + "_" + str(yr) \
                    + "_splits.csv"
        df = pd.read_csv(file_path)
        max_list = df.max(axis=1).dropna().tolist()
        split_lists.append(max_list)

    fig, ax = plt.subplots()
    ax.boxplot(split_lists, showfliers=show_outliers, patch_artist=True, medianprops=dict(color="black"),
               flierprops=dict(markeredgecolor='silver'), labels=[str(yr) for yr in year_list])
    title = "SLR " + rcp + " Boxplots of Highest S.temperature Split Values of each Tree"
    plt.ylabel("S.temperature Split Value")
    plt.xlabel("Year")
    plt.title(title, fontsize=15)
    plt.grid(b=True, axis='y', color='gray')
    plt.show()
```

**Usage:**


```python
list_20_yrs = []
for yr in range(2020, 2151, 20):
    list_20_yrs.append(yr)
Stemp_max_split_boxplots(list_20_yrs, "RCP 8.5", show_outliers=True)
```


![png](output_69_0.png)



```python

```
