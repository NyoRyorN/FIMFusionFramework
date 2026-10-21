from reports.ablation_boxplot.ablation_plots_each_class import ablationPlotsEachClass
from reports.ablation_boxplot.ablation_sum_plots import ablationSumPlots
import os

if __name__ == "__main__":
    rootDir = input("Input the directory path: ")
    if not os.path.exists(rootDir):
        print(f"Directory {rootDir} does not exist.")
    else:
        ablationSumPlots(rootDir)
        print(f"Ablation summary plots generated in {rootDir}.")
        ablationPlotsEachClass(rootDir)
        print(f"Ablation plots for each class generated in {rootDir}.")
