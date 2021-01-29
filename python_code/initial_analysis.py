"""
8/28/20-9/1/20
Importing the parameters sample data, determining correlation coefs between parameters, and making scatter plots of some parameters.
"""

import pandas as pd
import matplotlib.pyplot as plt

def main():
    df= pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/RData_parameters_sample.csv")
    print(df.head())
    #print(df.columns)

    #create correlation coef matrix
    cor_mat = df[:].corr()
    #pd.set_option("display.max_rows", 38, "display.max_columns", 38)
    #print(cor_mat)

    #filter correlatoin coef matrix
    cutoff= .5
    high_cor=cor_mat[(cor_mat < -cutoff) | (cor_mat > cutoff)]
    #print(high_cor)

    #create dataframe with only correlations that meet the cutoff defined above
    l=[]
    for i in range(0, 38):          #38= # of parameters
        for n in range (0, 38):
            if pd.notna(high_cor.iloc[i, n]) == True:
                if high_cor.columns.values[i] != high_cor.index.values[n]:
                    if (high_cor.columns.values[n], high_cor.index.values[i], high_cor.iloc[i, n]) not in l:
                        l.append((high_cor.columns.values[i], high_cor.index.values[n], high_cor.iloc[i, n]))
    high_cor=pd.DataFrame(l, columns=['Parameter 1', 'Parameter 2', 'Correlation'])
    pd.set_option("display.max_rows", 12, "display.max_columns", 3)
    print(high_cor)

    #plots
    for i in range (0, len(high_cor.index)):
        param_name1 = str(high_cor.iloc[i, 0])
        param_name2 = str(high_cor.iloc[i, 1])
        correlation = str("{0:.4f}".format(high_cor.iloc[i, 2]))
        plt.scatter(df.loc[:, [param_name1]], df.loc[:, [param_name2]])
        plt.xlabel(param_name1)
        plt.ylabel(param_name2)
        title= param_name2 + " vs " + param_name1 + " (corr = " + correlation + ")"
        plt.title(title)
        plt.show()



if __name__ == '__main__':
    main()