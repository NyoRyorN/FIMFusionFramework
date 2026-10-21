from src.utils.delete_after import DeleteAfter
from src.plot import SummaryOutPuts, BoxPlotsEachClass, all_calc_indicator


def ResultsProcess(rootPath, mode, flag_deleteAfter=True, flag_summaryOutputs=True, flag_detailPlots=True, range_detailPlots="0-100"):
    if flag_deleteAfter == "y":
        DeleteAfter(rootPath)

    if flag_summaryOutputs == "y":
        SummaryOutPuts(rootPath)
        if mode == 'C':
            all_calc_indicator(rootPath)
            print(f"Saved {rootPath} all_calc_indicator")

    if flag_detailPlots == "y":
        BoxPlotsEachClass(rootPath, mode, range_detailPlots)


if __name__ == "__main__":
    rootPath = input("Enter the root path: ")
    mode = input("Input the mode (N:[1-5, 6, 7], A:[1-5, 6-7], C:[1-6, 7]): ")
    mode = mode.upper()
    if mode not in ['N', 'A', 'C']:
        raise ValueError("Invalid mode. Please enter 'N', 'A', or 'C'.")
    flag_deleteAfter = input("Do you want to apply 'delete_after.py'? (y/n): ")
    flag_summaryOutputs = input(
        "Do you want to apply 'summary_outputs.py(N, A) or all_calc_indicator.py(C)'? (y/n): ")
    flag_detailPlots = input(
        "Do you want to apply 'box_plots_each_class.py'? (y/n): ")
    range_detailPlots = input(
        "Do you want to use the range 0-100 for detail plots? (y/n) : "
    )
    ResultsProcess(rootPath, mode, flag_deleteAfter=flag_deleteAfter,
                   flag_summaryOutputs=flag_summaryOutputs, flag_detailPlots=flag_detailPlots, range_detailPlots=range_detailPlots)
