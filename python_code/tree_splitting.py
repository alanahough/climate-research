"""
9/13/20-9/15/20
Writing functions to calculate entropy, calculate information gain, and determine the best split based on
information gain.
9/22/20
Writing generate tennis function, update entropy function to work with continuous case, calculated
conditional probabilities.
9/29/20
Changing generate tennis function to generate tennis column of dataframe using conditional probability of other
features.  Conditional probabilities were calculated by creating a random dataframe WITH the tennis column (using
generate tennis function) and then calculating the conditional probabilities from that (done in the probabilities()
function).
"""

import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Union
import turtle as t

@dataclass
class ColumnData:
    __slots__ = "column_name", "data_type", "values"
    column_name: str
    data_type: str      #categorical, discrete, continuous
    values: list        #list of possible values for categorical, range of numbers for discrete and continuous

@dataclass
class Tree:
    __slots__ = "node", "split", "left_val", "left", "right"
    node: str
    split: str
    left_val: Union[float, str, int]
    left: Union[float, 'Tree']
    right: Union[float, 'Tree']

def generate_tennis(col_info, nrows, condition):
    """
    Randomly generates a tennis dataset with nrows number of rows, where tennis is determined by conditional
    probabilities.
    :param col_info: list of ColumnData dataclasses
    :param nrows: number of rows to generate (int)
    :param condition: string of the features to make tennis conditional on when determining each row's tennis value
    :return: df: tennis dataframe
    """

    #changed sun to .9 then to .6
    conditional_probs={'sun': 0.6, 'wind': 0.7142857142857143, 'humidity': 0.5454545454545454,
                       'season': 0.6, 'time': 0.7142857142857143, 'temp': 0.6470588235294118, 'travel': 0.6470588235294118}

    col_names=[]
    for col in col_info:
        if col.column_name == condition:
            condition_val = col.values[0]
        col_names.append(col.column_name)
    col_names.append("tennis")
    row_list=[]

    for r in range (0, nrows):
        row=()
        for col in col_info:
            if col.data_type == "categorical" and col.column_name != "tennis":
                val= np.random.choice(col.values, 1)
                val = val[0]
                if col.column_name == condition:
                    if val == condition_val:
                        tennis = np.random.choice(["tennis", "no tennis"], 1,
                                                  p=[conditional_probs[condition], 1 - conditional_probs[condition]])
                    else:
                        tennis = np.random.choice(["tennis", "no tennis"], 1,
                                                  p=[1 - conditional_probs[condition], conditional_probs[condition]])
            elif col.data_type == "discrete":
                val= np.random.randint(col.values[0], col.values[1] + 1, 1)     #does not include "high" value w/o +1
                val = val[0]
            else:   #if continuous
                val= np.random.uniform(col.values[0], col.values[1], 1)
                val=round(val[0], 2)    #round float value to 2 decimal places
            row += (val ,)
        row += (tennis[0] ,)
        row_list.append(row)
    df=pd.DataFrame(row_list, columns=col_names)
    print("Conditional probability of", condition, "to make tennis:", conditional_probs[condition])
    return df


def entropy(df, feature):
    """
    Determines the entropy of a feature.
    :param df: dataframe that contains data for the feature
    :param feature: string of feature name
    :return: entropy
    """

    f = df[feature].tolist()
    tot = len(f)
    entropy = 0
    dict = {}
    bins=3

    #if empty dataframe: entropy=0
    if len(f) == 0:
        return 0

    #make dictionary of values and how many times they occur
    if isinstance(f[0], float) == True:     #if continuous feature
        nmin= min(f)
        nmax= max(f)
        bin_size = (nmax - nmin) / bins
        for i in range (0, bins):
            dict[int(nmin+i*bin_size)] = 0
        lastbin=int(nmin+(bins-1)*bin_size)
        for val in f:
            prev_key = nmin-1
            #each key has counts of values that are in the interval [current key, next key)
            for key in dict.keys():
                if val > lastbin:
                    dict[lastbin] += 1
                    break
                elif val > key:
                    prev_key= key
                else:
                    dict[prev_key] += 1
                    break
    else:       #if categorical or discrete feature
        for ft in f:
            if ft in dict:
                dict[ft] += 1
            else:
                dict[ft] = 1

    #calculate entropy
    for key in dict:
        if dict[key] > 0:
            pc = dict[key]/tot
            entropy -= pc*math.log2(pc)

    return entropy


def info_gain(parent, left, right, target):
    """
    Determines the information gain of a potential feature split.
    :param parent: dataframe of the parent node
    :param left: dataframe of the left child node
    :param right: dataframe of the right child node
    :param target: string of target feature name
    :return: information gain
    """

    par_len = parent.shape[0]
    l_len = left.shape[0]
    r_len = right.shape[0]

    ig = entropy(parent, target) - l_len / par_len * entropy(left, target) - r_len / par_len * entropy(right, target)

    return ig


def best_split(df, target):
    """
    Determines the split that will give the largest information gain.
    :param df: dataframe
    :param target: string of target feature name
    :return: tuple of (feature, information gain, split value/string) for the best split
    """
    ig_dict={}
    #loop through each feature and calculate infornmation gain
    for feature in df:
        first_element = df[feature].tolist()[0]
        if feature == target:   #do not want to include the information gain of target (bc =1)
            pass
        elif isinstance(first_element, str) == True:   #if 1st element in column is a string, feature is categorical
            left= first_element
            gain=info_gain(df, df[df[feature] == left], df[df[feature] != left], target)
            ig_dict[feature] = (feature, gain, left)    #add feature, gain, split value to dictionary
        else:      #if 1st element in column is a number (either int or float)
            ig=0
            split_val= 0
            min= df[feature].min()             #minimum value in column
            max= df[feature].max()             #maximum value in column
            step_size = .1                          #size of step for looking at potential split values
            steps = int((max - min) / step_size)    #number of steps to get from min to max with given step size
            ran= np.linspace(min + step_size, max, steps)   #1D array of every value to test as split value

            #test possible split values for best information gain
            for val in ran:
                val=round(val, 2)
                gain = info_gain(df, df[df[feature] < val], df[df[feature] >= val], target)
                if gain > ig:   #if this split value's gain is higher than highest so far, update ig and split_val
                     ig= gain
                     split_val= val
            ig_dict[feature]= (feature, ig, split_val)  #add feature, gain, split value to dictionary

    split=(None, 0, None)
    #loop through dictionary of all features and their info gain and split value
    for key in ig_dict:
        ig= ig_dict[key][1]
        if ig > split[1]:       #if this info gain is higher than highest so far, update split tuple
            split=ig_dict[key]

    return split


def test1():
    tennis=pd.DataFrame({'sun': ["sunny", "sunny", "not sunny", "sunny"],
                     "wind": ["windy", "not windy", "not windy", "windy"],
                     "humidity": ["not humid", "not humid", "humid", "humid"],
                     "tennis": ["tennis", "tennis", "no tennis", "no tennis"]})

    print("sun entropy =", entropy(tennis, "sun"))
    print("tennis entropy =", entropy(tennis, "tennis"))

    print(entropy(tennis[tennis["sun"] == "sunny"], "tennis"))

    print("sun info gain =", info_gain(tennis, tennis[tennis["sun"] == 'sunny'], tennis[tennis["sun"] != "sunny"], "tennis"))
    print("wind info gain =", info_gain(tennis, tennis[tennis["wind"] == 'windy'], tennis[tennis["wind"] != "windy"], "tennis"))
    print("humidity info gain =", info_gain(tennis, tennis[tennis["humidity"] == 'humid'], tennis[tennis["humidity"] != "humid"], "tennis"))

    disc=pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,],
                       "y": [1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7, 8, 9, 10],
                       "z": [1, 2, 8, 8, 5, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]})

    print("x entropy =", entropy(disc, "x"))
    print("y entropy =", entropy(disc, "y"))
    print("z entropy =", entropy(disc, "z"))

    for col in disc:
        dict = {}
        f = disc[col].tolist()
        for ft in f:
            if ft in dict:
                dict[ft] += 1
            else:
                dict[ft] = 1
        n=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        d={}
        for num in n:
            if num in dict.keys():
                d[num]=dict[num]
            else:
                d[num] = 0
        l_label= col + ", entropy = " + str("{0:.4f}".format(entropy(disc, col)))
        plt.plot(n, d.values(), label= l_label)
    plt.legend(bbox_to_anchor=(0.02, .97), loc='upper left', borderaxespad=0.)
    plt.axis([1, 10, 0, 17])
    plt.show()

    print(best_split(tennis, "tennis"))
    print(best_split(disc, "z"))
    print(best_split(disc, "y"))


def test2():
    #best split with continuous case
    sun = ColumnData("sun", "categorical", ["sunny", "not sunny"])
    wind = ColumnData("wind", "categorical", ["windy", "not windy"])
    humidity = ColumnData("humidity", "categorical", ["humid", "not humid"])
    season = ColumnData("season", "categorical", ["summer", "fall"])
    time = ColumnData("time", "categorical", ["AM", "PM"])
    temp = ColumnData("temp", "continuous", [50, 100])  # in degrees F
    travel = ColumnData("travel", "discrete", [0, 30])  # travel time to tennis court in minutes
    col_info=[sun, wind, humidity, season, time, temp, travel]
    pd.set_option('display.max_columns', None)
    df = generate_tennis(col_info, 10)
    print(df)
    #x=entropy(df, "temp")
    #print(x)
    best=best_split(df, "tennis")
    print(best)

def probabilities():
    sun = ColumnData("sun", "categorical", ["sunny", "not sunny"])
    wind = ColumnData("wind", "categorical", ["windy", "not windy"])
    humidity = ColumnData("humidity", "categorical", ["humid", "not humid"])
    season = ColumnData("season", "categorical", ["summer", "fall"])
    time = ColumnData("time", "categorical", ["AM", "PM"])
    temp = ColumnData("temp", "continuous", [50, 100])  # in degrees F
    travel = ColumnData("travel", "discrete", [0, 30])  # travel time to tennis court in minutes
    # need to include tennis as random input to get conditional probabilities to use in generate_tennis() func
    # changed tennis column_name to "tennis?" instead of "tennis" so that it actually generates a random
    # tennis column
    tennis = ColumnData("tennis?", "categorical", ["tennis", "no tennis"])
    col_info = [sun, wind, humidity, season, time, temp, travel, tennis]
    rows = 30

    # set seed to get same dataframe & probabilities every time
    #np.random.seed(27)
    pd.set_option('display.max_columns', None)
    df = generate_tennis(col_info, rows, "sun")
    print(df.loc[: , :"tennis?"])

    feature_list=[]
    for feature in df:
        feature_list.append(feature)
    f_list_copy = feature_list.copy()

    tennis = len(df.loc[df["tennis?"] == "tennis",])
    prob_tennis = tennis / rows
    print("probability of playing tennis =", "{0:.5f}".format(prob_tennis), "\n")

    # conditional probabilities
    # P(A|B)=P(A & B) / P(B)

    #conditional probabilities
    conditional_probs = {}
    for feature1 in df:
        del f_list_copy[0]
        for dclass in col_info:
            if dclass.column_name == feature1:
                col1 = dclass
                break
        #calculating conditional probability
        if feature1 == "tennis?" or feature1 == "tennis":
            pass
        elif col1.data_type == "categorical":
            left1 = len(df.loc[df[feature1] == col1.values[0],])
            left_tennis1 = len(df.loc[(df[feature1] == col1.values[0]) & (df["tennis?"] == "tennis"),])
            prob = left_tennis1 / left1
            print(feature1, ": probability of playing tennis if it's", col1.values[0], "=", "{0:.5f}".format(prob))
            conditional_probs[feature1] = prob
        else:
            mid1 = (col1.values[1] - col1.values[0]) / 2 + col1.values[0]
            left1 = len(df.loc[df[feature1] <= mid1,])
            left_tennis1 = len(df.loc[(df[feature1] <= mid1) & (df["tennis?"] == "tennis"),])
            prob = left_tennis1 / left1
            print(feature1, ": probability of playing tennis if", feature1, "is less than or equal to", mid1,
                  "=", "{0:.5f}".format(prob))
            conditional_probs[feature1] = prob

        for feature2 in f_list_copy:
            for dclass in col_info:
                if dclass.column_name == feature2:
                    col2 = dclass
                    break
            if (col1.data_type == "discrete" or col1.data_type == "continuous") and col2.data_type == "categorical":
                #makes sure f1 is always categorical if at least one feature categorical
                f1= col2
                f2= col1
            else:
                f1= col1
                f2= col2
            # calculating conditional probability
            if feature1 == "tennis?" or feature1 == "tennis" or feature2 == "tennis?" or feature2 == "tennis":
                pass
            elif f1.data_type == "categorical" and f2.data_type == "categorical":
                left2 = len(df.loc[(df[f1.column_name] == f1.values[0]) & (df[f2.column_name] == f2.values[0]),])
                left_tennis2 = len(df.loc[(df[f1.column_name] == f1.values[0]) & (df[f2.column_name] == f2.values[0])
                                          & (df["tennis?"] == "tennis"),])
                prob = left_tennis2 / left2
                feature= f1.column_name + " " + f2.column_name
                print(feature, ": probability of playing tennis if it's", f1.values[0], "and", f2.values[0],
                      "=", "{0:.5f}".format(prob))
                conditional_probs[feature] = prob
            elif f1.data_type == "categorical":
                mid = (f2.values[1] - f2.values[0]) / 2 + f2.values[0]
                left2 = len(df.loc[(df[f1.column_name] == f1.values[0]) & (df[f2.column_name] <= mid),])
                left_tennis2 = len(df.loc[(df[f1.column_name] == f1.values[0]) & (df[f2.column_name] <= mid)
                                          & (df["tennis?"] == "tennis"),])
                prob = left_tennis2 / left2
                feature = f1.column_name + " " + f2.column_name
                print(feature, ": probability of playing tennis if it's", f1.values[0], "and", f2.column_name,
                      "is less than or equal to", mid, "=", "{0:.5f}".format(prob))
                conditional_probs[feature] = prob
            else:
                mid2 = (f2.values[1] - f2.values[0]) / 2 + f2.values[0]
                left2 = len(df.loc[(df[f1.column_name] <= mid1) & (df[f2.column_name] <= mid2),])
                left_tennis2 = len(df.loc[(df[f1.column_name] <= mid1) & (df[f2.column_name] <= mid2) &
                                          (df["tennis?"] == "tennis"),])
                prob = left_tennis2 / left2
                feature = f1.column_name + " " + f2.column_name
                print(feature, ": probability of playing tennis if", f1.column_name, "is less than or equal to", mid1,
                      "and", f2.column_name, "is less than or equal to", mid2, "=", "{0:.5f}".format(prob))
                conditional_probs[feature] = prob
    print("\n", conditional_probs)


def gen_tennis_conditional(rows):
    sun = ColumnData("sun", "categorical", ["sunny", "not sunny"])
    wind = ColumnData("wind", "categorical", ["windy", "not windy"])
    humidity = ColumnData("humidity", "categorical", ["humid", "not humid"])
    season = ColumnData("season", "categorical", ["summer", "fall"])
    time = ColumnData("time", "categorical", ["AM", "PM"])
    temp = ColumnData("temp", "continuous", [50, 100])  # in degrees F
    travel = ColumnData("travel", "discrete", [0, 30])  # travel time to tennis court in minutes
    col_info = [sun, wind, humidity, season, time, temp, travel]
    condition= "sun"
    one_condition = True
    col1= sun
    col2= humidity
    # set seed to get same dataframe & probabilities every time
    #np.random.seed(27)
    pd.set_option('display.max_columns', None)
    df = generate_tennis(col_info, rows, condition)
    if one_condition == True:
        for dclass in col_info:
            if dclass.column_name == condition:
                col = dclass
                break
        if col.data_type == "categorical":
            left = len(df.loc[df[condition] == col.values[0],])
            left_tennis = len(df.loc[(df[condition] == col.values[0]) & (df["tennis"] == "tennis"),])
            prob = left_tennis / left
            print(condition, ": probability of playing tennis if it's", col.values[0], "=", "{0:.5f}".format(prob))
        else:
            mid = (col.values[1] - col.values[0]) / 2 + col.values[0]
            left = len(df.loc[df[condition] <= mid,])
            left_tennis = len(df.loc[(df[condition] <= mid) & (df["tennis"] == "tennis"),])
            prob = left_tennis / left
            print(condition, ": probability of playing tennis if", condition, "is less than or equal to", mid,
                  "=", "{0:.5f}".format(prob))
    else:
        if (col1.data_type == "discrete" or col1.data_type == "continuous") and col2.data_type == "categorical":
            # makes sure f1 is always categorical if at least one feature categorical
            f1 = col2
            f2 = col1
        else:
            f1 = col1
            f2 = col2
        # calculating conditional probability
        if f1.data_type == "categorical" and f2.data_type == "categorical":
            left2 = len(df.loc[(df[f1.column_name] == f1.values[0]) & (df[f2.column_name] == f2.values[0]),])
            left_tennis2 = len(df.loc[(df[f1.column_name] == f1.values[0]) & (df[f2.column_name] == f2.values[0])
                                      & (df["tennis"] == "tennis"),])
            prob = left_tennis2 / left2
            feature = f1.column_name + " " + f2.column_name
            print(feature, ": probability of playing tennis if it's", f1.values[0], "and", f2.values[0],
                  "=", "{0:.5f}".format(prob))
        elif f1.data_type == "categorical":
            mid = (f2.values[1] - f2.values[0]) / 2 + f2.values[0]
            left2 = len(df.loc[(df[f1.column_name] == f1.values[0]) & (df[f2.column_name] <= mid),])
            left_tennis2 = len(df.loc[(df[f1.column_name] == f1.values[0]) & (df[f2.column_name] <= mid)
                                      & (df["tennis"] == "tennis"),])
            prob = left_tennis2 / left2
            feature = f1.column_name + " " + f2.column_name
            print(feature, ": probability of playing tennis if it's", f1.values[0], "and", f2.column_name,
                  "is less than or equal to", mid, "=", "{0:.5f}".format(prob))
        else:
            mid1 = (f1.values[1] - f1.values[0]) / 2 + f1.values[0]
            mid2 = (f2.values[1] - f2.values[0]) / 2 + f2.values[0]
            left2 = len(df.loc[(df[f1.column_name] <= mid1) & (df[f2.column_name] <= mid2),])
            left_tennis2 = len(df.loc[(df[f1.column_name] <= mid1) & (df[f2.column_name] <= mid2) &
                                      (df["tennis"] == "tennis"),])
            prob = left_tennis2 / left2
            feature = f1.column_name + " " + f2.column_name
            print(feature, ": probability of playing tennis if", f1.column_name, "is less than or equal to", mid1,
                  "and", f2.column_name, "is less than or equal to", mid2, "=", "{0:.5f}".format(prob))
    return df


def write_csv():
    sun = ColumnData("sun", "categorical", ["sunny", "not sunny"])
    wind = ColumnData("wind", "categorical", ["windy", "not windy"])
    humidity = ColumnData("humidity", "categorical", ["humid", "not humid"])
    season = ColumnData("season", "categorical", ["summer", "fall"])
    time = ColumnData("time", "categorical", ["AM", "PM"])
    temp = ColumnData("temp", "continuous", [50, 100])  # in degrees F
    travel = ColumnData("travel", "discrete", [0, 30])  # travel time to tennis court in minutes
    col_info = [sun, wind, humidity, season, time, temp, travel]

    df= generate_tennis(col_info, 30, 'sun')
    df.to_csv(r'C:\Users\hough\Documents\research\data\new_csv\tennis\tennis30.csv', index=False)
    df = generate_tennis(col_info, 100, 'sun')
    df.to_csv(r'C:\Users\hough\Documents\research\data\new_csv\tennis\tennis100.csv', index=False)
    df = generate_tennis(col_info, 500, 'sun')
    df.to_csv(r'C:\Users\hough\Documents\research\data\new_csv\tennis\tennis500.csv', index=False)
    df = generate_tennis(col_info, 1000, 'sun')
    df.to_csv(r'C:\Users\hough\Documents\research\data\new_csv\tennis\tennis1000.csv', index=False)


def decision_tree(df, target, levels, node= "parent", prev_split_prob=0):
    """
    :param df: dataframe
    :param target: target variable trying to predict
    :param levels: number of levels of splits the tree should make
    :param node: parent, left, right, etc
    :param prev_split_prob: conditional probability of playing tennis with the prior splits
    :return: Tree dataclass
    """
    node_list=node.split()
    t.down()
    for i in range (1, len(node_list)):
        if node_list[i] == "parent":
            pass
        elif node_list[i] == "left":
            t.right(75)
            t.forward(90)
            t.up()
            t.forward(10)
            t.left(75)
            t.down()
        else:
            t.left(75)
            t.forward(90)
            t.up()
            t.forward(10)
            t.right(75)
            t.down()

    if prev_split_prob >= .5:
        ten = "tennis"
    else:
        ten = "no tennis"

    if levels == 0:
        print("[", node, "]", "prob of tennis =", prev_split_prob, "->", ten)
        t.write(ten, align= "center")     #turtle
        for i in range(len(node_list) - 1, 0, -1):
            if node_list[i] == "parent":
                pass
            elif node_list[i] == "left":
                t.up()
                t.right(75)
                t.backward(100)
                t.left(75)
            else:
                t.up()
                t.left(75)
                t.backward(100)
                t.right(75)
        return prev_split_prob
    else:
        split, ig, val = best_split(df, target)
        if split == None:
            print("[", node, "]", "prob of tennis =", prev_split_prob, "->", ten)
            t.write(ten, align="center")      #turtle
            for i in range(len(node_list)-1, 0, -1):
                if node_list[i] == "parent":
                    pass
                elif node_list[i] == "left":
                    t.up()
                    t.right(75)
                    t.backward(100)
                    t.left(75)
                else:
                    t.up()
                    t.left(75)
                    t.backward(100)
                    t.right(75)
            return prev_split_prob
        else:
            if isinstance(val, str) == True:
                print("[", node, "]", "split :", split, "{ left child =", val, "}")
                t.write(split, align="center")
                t.up()
                t.forward(10)
                t.right(90)
                t.forward(10)
                t.write(val, align="right")
                t.backward(10)
                t.left(90)
                t.backward(10)
                dfl= df.loc[df[split] == val, ]
                dfr= df.loc[df[split] != val, ]
                lefttot= len(df.loc[df[split] == val, ])
                lefttarget=len(df.loc[(df[split] == val) & (df[target] == target), ]) #assuming target value is same as target column name
                probleft= lefttarget / lefttot
                righttot= len(df.loc[df[split] != val, ])
                righttarget = len(df.loc[(df[split] != val) & (df[target] == target),])  # assuming target value is same as target column name
                probright= righttarget/righttot
            else:
                print("[", node, "]", "split :", split, "{ left child =", split, "<", val, "}")
                t.write(split, align="center")
                t.up()
                t.forward(10)
                t.right(90)
                t.forward(10)
                l_val = ">" + str(val)
                t.write(l_val, align="right")
                t.backward(10)
                t.left(90)
                t.backward(10)
                dfl= df.loc[df[split] < val, ]
                dfr = df.loc[df[split] >= val, ]
                lefttot = len(df.loc[df[split] < val,])
                lefttarget = len(df.loc[(df[split] < val) & (df[target] == target),])  # assuming target value is same as target column name
                probleft = lefttarget / lefttot
                righttot = len(df.loc[df[split] >= val,])
                righttarget = len(df.loc[(df[split] >= val) & (df[target] == target),])  # assuming target value is same as target column name
                probright = righttarget / righttot
            dfl = dfl.drop(columns=[split])
            dfr = dfr.drop(columns=[split])
            node_l = node + " left"
            node_r = node + " right"
            for i in range(len(node_list)-1, 0, -1):
                if node_list[i] == "parent":
                    pass
                elif node_list[i] == "left":
                    t.up()
                    t.right(75)
                    t.backward(100)
                    t.left(75)
                else:
                    t.up()
                    t.left(75)
                    t.backward(100)
                    t.right(75)
            tree= Tree(node, split, val, decision_tree(dfl, target, levels-1, node_l, probleft),
                       decision_tree(dfr, target, levels - 1, node_r, probright))
            return tree


def validation(df, tree):
    df = df.reset_index(drop= True)
    correct = 0
    len_df= df.shape[0]
    # loop through rows
    for i in range(0, len_df):
        tree_copy= tree
        while isinstance(tree_copy, float) == False:    #if float, then leaf node bc has probability instead of another branch
            if isinstance(tree_copy.left_val, str) == True:
                if df.loc[i, tree_copy.split] == tree_copy.left_val:
                    tree_copy = tree_copy.left
                else:
                    tree_copy = tree_copy.right
            else:   #continuous/discrete feature
                if df.loc[i, tree_copy.split] < tree_copy.left_val:
                    tree_copy = tree_copy.left
                else:
                    tree_copy = tree_copy.right
        if (tree_copy >= .5 and df.loc[i, "tennis"] == "tennis") or (tree_copy < .5 and df.loc[i, "tennis"] != "tennis"):
            correct += 1
    print("correct:", correct, "incorrect:", len_df - correct)
    accuracy = correct / len_df
    return accuracy


def main():
    np.random.seed(957)
    #test1()
    #test2()
    #probabilities()
    #print(gen_tennis_conditional(30))
    #write_csv()

    t.hideturtle()
    t.speed(0)
    t.right(90)

    df30 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/tennis/tennis30.csv")
    #print("df30 tree w/ 3 levels")
    #tree30 = decision_tree(df30, "tennis", 3)

    df100 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/tennis/tennis100.csv")
    #print("\ndf100 tree w/ 3 levels")
    #tree100_3 = decision_tree(df100, "tennis", 3)
    #print("\ndf100 tree w/ 5 levels")
    #tree100 = decision_tree(df100, "tennis", 5)
    #print("df100 tree w/ 7 levels")
    #tree100_7 = decision_tree(df100, "tennis", 7)

    #print("\n30 row df, 30 row tree w/ 3 levels")
    #print("accuracy =", validation(df30, tree30))
    #print("\n100 row df, 100 row tree w/ 3 levels")
    #print("accuracy =", validation(df100, tree100_3))
    #print("\n100 row df, 100 row tree w/ 5 levels")
    #print("accuracy =", validation(df100, tree100))
    #print("\n100 row df, 100 row tree w/ 7 levels")
    #print("accuracy =", validation(df100, tree100_7))
    #print("\n100 row df, 30 row tree w/ 3 levels")
    #print("accuracy =", validation(df100, tree30))

    #df500 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/tennis/tennis500.csv")
    #train = df500.loc[ :249, ]
    #validate = df500.loc[250: ,]
    #for i in range(1, 8):
    #    tree500=decision_tree(train, "tennis", i)
    #    print("2nd 250 rows of 500 df, 1st 250 row tree w/", i, "levels")
    #    print("accuracy =", validation(validate, tree500), "\n")

    df1000 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/tennis/tennis1000.csv")
    train = df1000.loc[:449, ]
    validate = df1000.loc[500:, ]
    #for i in range(1, 8):
    #    tree1000=decision_tree(train, "tennis", i)
    #    print("2nd 500 rows of 1000 df, 1st 500 row tree w/", i, "levels")
    #    print("accuracy =", validation(validate, tree100), "\n")

    #tree1000_1= decision_tree(df1000, "tennis", 1)
    #print("\n1000 row df, 1000 row tree w/ 1 levels")
    #print("accuracy =", validation(df1000, tree1000_1))

    tree1000_7 = decision_tree(df1000, "tennis", 7)
    print("\n1000 row df, 1000 row tree w/ 7 levels")
    print("accuracy =", validation(df1000, tree1000_7))
    t.done()
    print("tree data class", tree1000_7)


main()