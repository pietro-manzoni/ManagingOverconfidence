# Perform analysis of results from RRN_Overconfidence project
# Last modified: 24Jun24

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities import read_results_10_runs

pd.set_option('display.max_columns', None)

# Define BaseFolder for files importation
base_folderA = "NN_RESULTS/CRPS/Test=2012/Lambda=000/"
base_folderB = "NN_RESULTS/CRPS/Test=2012/Lambda=015/"
base_folderC = "NN_RESULTS/CWS/Test=2012/Lambda=000/"
base_folderD = "NN_RESULTS/CWS/Test=2012/Lambda=030/"
base_folderE = "NN_RESULTS/MLE/Test=2012/Lambda=000/"

# Coverage Plot Settings
labels = ["A", "B", "C", "D", "E"]
line_styles = ["dotted", "dashed", "dotted", "dashed", "dotted"]
marker_types = ["o", "s", "^", "v", "X"]
colours = ["dodgerblue", "green", "hotpink", "tan", "darkorange"]
y_lim_plot = [-3.5, 3.5]

# store all results in a dataFrame
final_results = pd.DataFrame(columns=["MAPE", "se(MAPE)", "RMSE", "se(RMSE)",
                                      "APL", "se(APL)", "AACE", "se(AACE)",
                                      "Cover90", "se(Cover90)", "Cover95", "se(Cover95)"])

final_results.loc["A"], coverage_A = read_results_10_runs(base_folderA)
final_results.loc["B"], coverage_B = read_results_10_runs(base_folderB)
final_results.loc["C"], coverage_C = read_results_10_runs(base_folderC)
final_results.loc["D"], coverage_D = read_results_10_runs(base_folderD)
final_results.loc["E"], coverage_E = read_results_10_runs(base_folderE)

# print Tables of Results
print(final_results.round(2))

# Create Coverage Plots

# save all coverage results in a list
coverage_list = [coverage_A, coverage_B, coverage_C, coverage_D, coverage_E]
nominal = np.arange(90, 100)


def plot_means(idx, coverage):
    mean_ce = np.array(coverage["mean"], dtype=float)
    plt.plot(nominal, mean_ce, color=colours[idx], linestyle=line_styles[idx], marker=marker_types[idx], zorder=-1)


def plot_bands(idx, coverage):
    dw_ce = np.array(coverage["mean"] - coverage["se"], dtype=float)
    up_ce = np.array(coverage["mean"] + coverage["se"], dtype=float)
    plt.fill_between(nominal, dw_ce, up_ce, color=colours[idx], alpha=0.25, zorder=0)


plt.figure()
for (i, cov) in enumerate(coverage_list):
    plot_means(i, cov)
plt.plot(nominal, 0 * nominal, color='black', linestyle='solid', zorder=-1)
for (i, cov) in enumerate(coverage_list):
    plot_bands(i, cov)

labels.append('Target')
plt.legend(labels, handlelength=2.5, ncol=3, columnspacing=0.9)
plt.xlabel("Nominal Coverage [%]")
plt.ylabel("Coverage Error [%]")
plt.ylim(y_lim_plot)
plt.yticks(np.arange(np.ceil(y_lim_plot[0]), np.ceil(y_lim_plot[1]), 1.0))

plt.show()
