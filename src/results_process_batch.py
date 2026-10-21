from src.results_process import ResultsProcess
import glob
import os

rootDir = input("Enter the root path: ")
dir_list = [d for d in glob.glob(rootDir + "/*") if os.path.isdir(d)]
mode = input("Input the mode (N:[1-5, 6, 7], A:[1-5, 6-7], C:[1-6, 7]): ")
mode = mode.upper()
if mode not in ['N', 'A', 'C']:
    raise ValueError("Invalid mode. Please enter 'N', 'A', or 'C'.")
flag_deleteAfter = input("Do you want to apply 'delete_after.py'? (y/n): ")
flag_summaryOutputs = input(
    "Do you want to apply 'summary_outputs.py'? (y/n): ")
flag_detailPlots = input(
    "Do you want to apply 'box_plots_each_class.py'? (y/n): ")


for d in dir_list:
    if "AblationPlots" in d:
        continue
    print(f"Processing {d} ...")
    ResultsProcess(d, mode, flag_deleteAfter,
                   flag_summaryOutputs, flag_detailPlots)
