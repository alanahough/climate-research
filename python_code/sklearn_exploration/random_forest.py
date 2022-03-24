"""
9/15/20-
"""

import pandas as pd
import random
import math
import turtle as t
from tree_splitting import decision_tree    #make sure main() run is commented out so it doesnt run it in this code

def random_forest(df, num_samples, num_trees, target, levels):
    rows= df.shape[0]
    features= df.shape[1]
    m= int(math.sqrt(features))
    trees=[]

    #creating trees
    for i in range (0, num_trees):
        #creating a training subset
        training_rows=[]
        for i in range (0, num_samples):
            training_rows.append(random.randint(0, rows - 1))
        #print(training_rows)
        training = pd.DataFrame(columns=df.columns.values.tolist())
        for row in training_rows:
            training=training.append(df.iloc[row])
        print(training)
        decision_tree(training, target, levels)


        #getting subset of features
        #m_features=[]
        #for i in range (0, m):
        #    n = random.randint(0, features - 1)
        #    m_features.append(df.columns.values[n])
        #print(m_features)

    #need to take out tennis first? calc m without tennis in list?


def test1():
    tennis = pd.DataFrame({'sun': ["sunny", "sunny", "not sunny", "sunny"],
                           "wind": ["windy", "not windy", "not windy", "windy"],
                           "humidity": ["not humid", "not humid", "humid", "humid"],
                           "tennis": ["tennis", "tennis", "no tennis", "no tennis"]})
    random_forest(tennis, 2, 3, "tennis", 2)

def test2():
    df100 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/tennis/tennis100.csv")
    random_forest(df100, 10, 5, "tennis", 3)

t.hideturtle()
t.speed(0)
t.right(90)
test1()
test2()