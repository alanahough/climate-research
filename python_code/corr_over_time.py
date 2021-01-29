"""
9/3/20-9/8/20
Importing the parameters sample data; Tgav data for rcp 26 and 85 for 2025, 2050, 2075, 2100, 2125, and 2150;
and slr data for rcp 26 and 85 for 2025, 2050, 2075, 2100, 2125, and 2150.  Determining correlation coefs between
parameters and the output data.  Making scatter plots of some of the correlations.  Making plots of high
correlations over time.
"""

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

def main():
    df= pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/RData_parameters_sample.csv")
    slr_rcp26= pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/slr_rcp26.csv")
    slr_rcp85 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/slr_rcp85.csv")
    Tgav_rcp26 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/Tgav_rcp26.csv")
    Tgav_rcp85 = pd.read_csv("C:/Users/hough/Documents/research/data/new_csv/Tgav_rcp85.csv")

    df_slr_rcp26 = df.join(slr_rcp26, how="outer")
    df_slr_rcp85 = df.join(slr_rcp85, how="outer")
    df_Tgav_rcp26 = df.join(Tgav_rcp26, how="outer")
    df_Tgav_rcp85 = df.join(Tgav_rcp85, how="outer")

    df_list=[df_slr_rcp26, df_slr_rcp85, df_Tgav_rcp26, df_Tgav_rcp85]
    years=[2025, 2050, 2075, 2100, 2125, 2150]
    years_str=["2025", "2050", "2075", "2100", "2125", "2150"]
    names=["slr rcp26", "slr rcp85", "Tgav rcp26", "Tgav rcp85"]
    c=0

    for df in df_list:
        #create correlation coef matrix
        cor_mat = df[:].corr()

        # filter correlatoin coef matrix
        cutoff = .55
        high_cor = cor_mat[(cor_mat < -cutoff) | (cor_mat > cutoff)]

        # create dataframe with only correlations that meet the cutoff defined above
        l = []
        for i in range(0, 44):  # 38 parameters & 6 years_str
            for n in range(0, 44):
                if pd.notna(high_cor.iloc[i, n]) == True:
                    if high_cor.columns.values[i] != high_cor.index.values[n]:
                        if high_cor.columns.values[i] in years_str or high_cor.index.values[n] in years_str:
                            if high_cor.columns.values[i] in years_str and high_cor.index.values[n] in years_str:
                                pass
                            elif (high_cor.columns.values[n], high_cor.index.values[i], high_cor.iloc[i, n]) not in l:
                                l.append((high_cor.columns.values[i], high_cor.index.values[n], high_cor.iloc[i, n]))
        high_cor = pd.DataFrame(l, columns=['Parameter', names[c], 'Correlation'])
        print(high_cor)

        #scatter plots- one popup for each plot
        #for i in range(0, len(high_cor.index)):
        #    param_name1 = str(high_cor.iloc[i, 0])
        #    param_name2 = str(high_cor.iloc[i, 1])
        #    correlation = str("{0:.4f}".format(high_cor.iloc[i, 2]))
        #    plt.scatter(df.loc[:, [param_name1]], df.loc[:, [param_name2]])
        #    plt.xlabel(param_name1)
        #    plt.ylabel(param_name2)
        #    title = names[c] + " year " + param_name2 + " vs " + param_name1 + " (corr = " + correlation + ")"
        #    plt.title(title)
        #    plt.show()

        #scatter plots- one model in one popup
        m=math.ceil(len(high_cor.index)/2)
        fig, axs= plt.subplots(m, 2)
        for i in range(0, len(high_cor.index)):
            param_name1 = str(high_cor.iloc[i, 0])
            param_name2 = str(high_cor.iloc[i, 1])
            correlation = str("{0:.4f}".format(high_cor.iloc[i, 2]))
            title = names[c] + " year " + param_name2 + " vs " + param_name1 + " (corr = " + correlation + ")"
            x=df[param_name1].tolist()
            y=df[param_name2].tolist()
            n=i//2
            if i % 2 == 0:
                axs[n, 0].scatter(x, y)
                axs[n, 0].set_title(title)
                axs[n, 0].set(xlabel= param_name1, ylabel= names[c] + " year " + param_name2)
            else:
                axs[n, 1].scatter(x, y)
                axs[n, 1].set_title(title)
                axs[n, 1].set(xlabel=param_name1, ylabel= names[c] + " year " + param_name2)
        fig.tight_layout(pad=1)
        plt.show()

        #high correlation over time plots
        corr_param=high_cor["Parameter"].tolist()
        c_list=[]
        for param in corr_param:
            if param in c_list:
                pass
            else:
                c_list.append(param)

        l_cutoff = []
        for i in range (0, len(years)):
            l_cutoff.append(cutoff)

        for param in c_list:
            cor = []
            for i in range(0, 44):
                for n in range(0, 44):
                    if cor_mat.columns.values[n] == param:
                        if cor_mat.index.values[i] in years_str:
                            cor.append(cor_mat.iloc[i,n])

            plt.plot(years, l_cutoff, '--k')
            plt.plot(years, cor,'-r', marker='o')
            plt.xticks(np.arange(2025, 2175, step=25))
            plt.yticks(np.arange(0, 1.1, step=0.1))
            plt.xlabel("Year")
            plt.ylabel("Correlation")
            title = param + " & " + names[c] + " Correlation over Time"
            plt.title(title)
            for a, b in zip(years, cor):
                plt.text(a, b, str("{0:.4f}".format(b)), horizontalalignment='right', verticalalignment='bottom')
            plt.show()

        c += 1



if __name__ == '__main__':
    main()