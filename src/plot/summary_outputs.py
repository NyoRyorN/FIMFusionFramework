# conclution specify, each Accuracies.csv load, Summary.csv create
import pandas as pd
import os
from src.utils import FIMitems, setPltFont
import matplotlib.pyplot as plt
from src.utils import plotTasks, Tasks


def PlotCrossItemPlots(data: pd.DataFrame, column: str, save_path: str):
    """specifycolumn each action createfunction(eachSeed Item and value Seed standard deviation calculate)
    Args:
        data (pd.DataFrame): data frame
        column (str): column names
        save_path (str): save path
    Returns:
        None
    """
    task_dict = {}
    all_tasks = Tasks()[:-1]
    all_plot_tasks = plotTasks()[:-1]
    for i in range(len(all_tasks)):
        task_dict[all_tasks[i]] = all_plot_tasks[i]

    data_df = data[[column, "FromAction", "Seed"]].copy()

    data_df["FromAction"] = data_df["FromAction"].apply(
        lambda x: task_dict.get(x, x))

    # eachSeed Item and value(valuecolumn specify)
    data_df = data_df.groupby(["FromAction", "Seed"]).agg({
        column: 'mean',
    }).reset_index()

    data_df.to_csv(os.path.join(
        save_path, f"CrossItem_{column}_Data.csv"), index=False)

    # eachFromAction for calculate(Seed standard deviation)
    stats_df = data_df.groupby("FromAction")[column].agg(
        mean="mean", std=lambda values: values.std(ddof=1)).reset_index()

    # all_tasks order and same order get
    action_order = [
        task for task in all_plot_tasks if task in stats_df["FromAction"].unique()]

    # action_order
    stats_df = stats_df.set_index("FromAction").reindex(
        action_order).reset_index()
    stats_df.to_csv(os.path.join(
        save_path, f"CrossItem_{column}_Stats.csv"), index=False)

    # bar plot create(mean value and error bars)
    fig, ax = plt.subplots(figsize=(40, 10))

    bars = ax.bar(stats_df["FromAction"], stats_df["mean"],
                  yerr=stats_df["std"], capsize=10,
                  color='skyblue', edgecolor='black', linewidth=1.5,
                  error_kw={'linewidth': 2, 'ecolor': 'red'})

    # each bar abovemean value display(two decimal places)
    for i, (bar, mean_val) in enumerate(zip(bars, stats_df["mean"])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + stats_df["std"].iloc[i] + 0.5,
                f'{mean_val:.2f}', ha='center', va='bottom',
                fontsize=32, fontweight='bold')  # adjust

    # all mean value points display
    overall_mean = data_df[column].mean()
    ax.axhline(overall_mean, color='red', linestyle='--', linewidth=2)
    ax.text(len(stats_df)-0.5, overall_mean + 0.5,
            f'Overall Mean: {overall_mean:.2f}', color='red', fontsize=28, fontweight='bold', ha='right')

    ax.set_title(f"Cross-Item {column} by Action (Mean ± Std)", fontsize=40)
    ax.set_xlabel("Action", fontname="Arial", fontsize=40)
    ax.set_ylabel(f"{column} (%)", fontname="Arial", fontsize=40)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)

    # Y-axis range minimum input value-2.5 from maximum value+2.5 set
    y_min = stats_df["mean"].min() - 2.5 - stats_df["std"].max()
    y_max = stats_df["mean"].max() + 2.5 + stats_df["std"].max()
    plt.ylim(y_min, y_max)

    plt.tight_layout()

    # save
    for idx in ["png", "pdf", "svg"]:
        save_file = os.path.join(
            save_path, f"CrossItem_{column}_BarPlot_MeanStd.{idx}")
        plt.savefig(save_file, bbox_inches='tight')

    plt.close(fig)


def CalcAllStd(data: pd.DataFrame, column: str) -> float:
    """specifycolumn allstandard deviation calculatefunction

    Args:
        data (pd.DataFrame): data frame
        column (str): standard deviation calculatecolumn names

    Returns:
        float: specifycolumn standard deviation
    """
    seed_list = data["Seed"].unique()
    all_data = []
    for seed in seed_list:
        seed_data = data[data["Seed"] == seed]
        all_data.append(seed_data[column].mean())

    return pd.Series(all_data).std(ddof=1)


def CalcMaxAction(data: pd.DataFrame, column: str) -> tuple:
    """specifycolumn each action maximum value calculatefunction

    Args:
        data (pd.DataFrame): data frame
        column (str): maximum value calculatecolumn names

    Returns:
        tuple: (maximum value, maximum value, standard deviation)
    """
    action_list = data["FromAction"].unique()
    all_data = []
    for action in action_list:
        action_data = data[data["FromAction"] == action]
        all_data.append(action_data[column].mean())
    all_data = pd.Series(all_data)
    max_action = action_list[all_data.idxmax()]
    max_action_data = data[data["FromAction"] == max_action]
    max_std = max_action_data[column].std(ddof=1)
    return max_action, all_data.max(), max_std


def CalcAllMaxActionStd(data: pd.DataFrame, item_list: list[str], max_action_list: list[str], column: str) -> float:
    """specifycolumn allstandard deviation calculatefunction

    Args:
        data (pd.DataFrame): data frame
        column (str): standard deviation calculatecolumn names

    Returns:
        float: specifycolumn standard deviation
    """
    all_data = pd.DataFrame()
    for item, action in zip(item_list, max_action_list):
        item_data = data[(data["ToFIM"] == item) &
                         (data["FromAction"] == action)]
        all_data = pd.concat([all_data, item_data])

    all_accuracy = []
    for seed in all_data["Seed"].unique():
        seed_data = all_data[all_data["Seed"] == seed]
        all_accuracy.append(seed_data[column].mean())

    return pd.Series(all_accuracy).mean(), pd.Series(all_accuracy).std(ddof=1)


def SummaryOutPuts(rootPath, ensemble=False):
    setPltFont()

    items = FIMitems()[:-1]

    save_path = rootPath
    # savedirectory create
    plot_save_path = os.path.join(rootPath, "results")
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    df = pd.read_csv(os.path.join(rootPath, "Accuracies.csv"), header=0) \
        if not ensemble else pd.read_csv(os.path.join(rootPath, "ensemble_results.csv"), header=0)
    df = df[df["Seed"] != "Mean"]

    # FromAction case create
    if "FromAction" in df.columns:
        PlotCrossItemPlots(df, "Balanced_Accuracy", plot_save_path)

    result_list = []
    new_columns = []

    if "FromAction" in df.columns:
        new_columns = ["FIMItem", "Accuracy_Mean", "Accuracy_Std", "Acc_Max_Action", "Acc_Max_Accuracy", "Acc_Max_Std",
                       "Balanced_Accuracy_Mean", "Balanced_Accuracy_Std", "Max_Balanced_Acc_Action", "Max_Balanced_Acc_Accuracy", "Max_Balanced_Acc_Std"
                       ]
        max_action_list = []
        max_balanced_action_list = []
        for item in items:
            temp_list = [item]
            data = df[df["ToFIM"] == item]
            max_action, max_accuracy, max_std = CalcMaxAction(data, "Accuracy")
            max_action_list.append(max_action)
            temp_list.append(data["Accuracy"].mean())
            temp_list.append(CalcAllStd(data, "Accuracy"))
            temp_list.append(max_action)
            temp_list.append(max_accuracy)
            temp_list.append(max_std)
            max_action, max_accuracy, max_std = CalcMaxAction(
                data, "Balanced_Accuracy")
            max_balanced_action_list.append(max_action)
            temp_list.append(data["Balanced_Accuracy"].mean())
            temp_list.append(CalcAllStd(data, "Balanced_Accuracy"))
            temp_list.append(max_action)
            temp_list.append(max_accuracy)
            temp_list.append(max_std)
            result_list.append(temp_list)
        temp_list = ["All"]
        data = df
        temp_list.append(data["Accuracy"].mean())
        temp_list.append(CalcAllStd(data, "Accuracy"))
        all_max_mean, all_max_std = CalcAllMaxActionStd(
            data, items, max_action_list, "Accuracy")
        temp_list.append("-")
        temp_list.append(all_max_mean)
        temp_list.append(all_max_std)
        all_balanced_max_mean, all_balanced_max_std = CalcAllMaxActionStd(
            data, items, max_balanced_action_list, "Balanced_Accuracy")
        temp_list.append(data["Balanced_Accuracy"].mean())
        temp_list.append(CalcAllStd(data, "Balanced_Accuracy"))
        temp_list.append("-")
        temp_list.append(all_balanced_max_mean)
        temp_list.append(all_balanced_max_std)
        result_list.append(temp_list)
    else:
        new_columns = ["FIMItem"]
        for column in [c for c in df.columns if c not in ["Seed", "ToFIM"]]:
            new_columns.append(column + "_Mean")
            new_columns.append(column + "_Std")
        for item in items:
            temp_list = [item]
            data = df[df["ToFIM" if "ToFIM" in df.columns else "Item"] == item]
            for column in [c for c in df.columns if c not in ["Seed", "ToFIM"]]:
                temp_list.append(data[column].mean())
                temp_list.append(data[column].std(ddof=1))
                # temp_list.append(data[column].max())
            result_list.append(temp_list)
        temp_list = ["All"]
        data = df
        for column in [c for c in df.columns if c not in ["Seed", "ToFIM"]]:
            temp_list.append(data[column].mean())
            temp_list.append(CalcAllStd(data, column))
        result_list.append(temp_list)

    df_all = pd.DataFrame(result_list, columns=new_columns)
    # df_all.loc["All"] = pd.Series(dtype="float64")
    # df_all.loc["All", "FIMItem"] = "All"
    # df_all.loc["All", "Accuracy_Mean"] = df_all["Accuracy_Mean"].mean()
    # df_all.loc["All", "Accuracy_Max"] = df_all["Accuracy_Max"].max()
    # df_all.loc["All", "Balanced_Accuracy_Mean"] = df_all["Balanced_Accuracy_Mean"].mean()
    # df_all.loc["All",
    #            "Balanced_Accuracy_Max"] = df_all["Balanced_Accuracy_Max"].max()
    df_all.dropna(inplace=True, thresh=2)
    df_all.to_csv(os.path.join(save_path, "Summary.csv"), index=False) \
        if not ensemble else pd.read_csv(os.path.join(save_path, "ensemble_summary.csv"), header=0)
    print(f"Saved {rootPath} summary")
    print("==============================================")
    print("SummaryOutPuts is finished.")
    print("==============================================")


if __name__ == "__main__":
    rootPath = input("Enter the root path: ")
    SummaryOutPuts(rootPath)
