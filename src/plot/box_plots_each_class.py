import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import Tasks, plotTasks, setPltFont


def BoxPlotsEachClass(root_path, mode, range_detailPlots):
    save_path = os.path.join(root_path, "results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    setPltFont()
    fontsize = 40

    colors = ["lightcoral", "Yellow", "Lime"]
    if mode == "N" or mode == "n":
        mode = "Normal"
    # elif mode == "A" or mode == "a":
    #     mode = "Assitance"
    #     colors = ["lightcoral", "Lime"]
    elif mode == "C" or mode == "c":
        mode = "Complete_independence"
        colors = ["Yellow", "Lime"]
    else:
        ValueError("Invalid mode. Choose from 'N', 'A', or 'C'.")

    df = pd.read_csv(os.path.join(root_path, "Accuracies.csv"), header=0)
    df = df[df["Seed"] != "Mean"].copy()

    items = df["ToFIM"].unique()

    if "FromAction" in df.columns:
        task_dict = {}
        all_tasks = Tasks()[:-1]
        all_plot_tasks = plotTasks()[:-1]
        for i in range(len(all_tasks)):
            task_dict[all_tasks[i]] = all_plot_tasks[i]

        df["FromAction"] = df["FromAction"].apply(
            lambda x: task_dict.get(x, x))

        plt.rcParams['font.size'] = fontsize
        plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Arial']})

        for item in items:
            item_df = df[df["ToFIM"] == item].copy()

            # all_tasks order and same order get
            all_plot_tasks = plotTasks()[:-1]
            action_order = [
                task for task in all_plot_tasks if task in item_df["FromAction"].unique()]

            fig, ax = plt.subplots(figsize=(40, 10))
            sns.boxplot(
                data=item_df,
                x="FromAction",
                y="Balanced_Accuracy",
                order=action_order,  # order specify
                showmeans=True,
                medianprops=dict(color='black',
                                 linewidth=3,
                                 linestyle='dashed'),
                whiskerprops=dict(linewidth=2),
                meanprops=dict(color='black',
                               marker='+',
                               markeredgecolor='red',
                               markerfacecolor='red',
                               markeredgewidth=5,
                               markersize=20),
                fliersize=10)
            ax.set_title(f"{item}", fontsize=fontsize)
            ax.set_xlabel("Action", fontname="Arial", fontsize=fontsize)
            ax.set_ylabel("Balanced_Accuracy (%)",
                          fontname="Arial", fontsize=fontsize)
            plt.xticks()

            for idx in ["png", "pdf", "svg"]:
                save_file = os.path.join(save_path, f"{item}.{idx}")
                plt.savefig(save_file, bbox_inches='tight')
            plt.close(fig)

            # mean value and error bars bar plot
            fig, ax = plt.subplots(figsize=(40, 10))

            # all_tasks order and same order get
            all_plot_tasks = plotTasks()[:-1]
            action_order = [
                task for task in all_plot_tasks if task in item_df["FromAction"].unique()]

            # eachFromAction for calculate
            stats_df = item_df.groupby("FromAction")["Balanced_Accuracy"].agg(
                mean="mean", std=lambda values: values.std(ddof=1)).reset_index()

            # action_order
            stats_df = stats_df.set_index("FromAction").reindex(
                action_order).reset_index()

            # bar plot create(mean value and error bars)
            bars = ax.bar(stats_df["FromAction"], stats_df["mean"],
                          yerr=stats_df["std"], capsize=10,
                          color='skyblue', edgecolor='black', linewidth=1.5,
                          error_kw={'linewidth': 2, 'ecolor': 'red'})

            # each bar abovemean value display(two decimal places)
            for i, (bar, mean_val) in enumerate(zip(bars, stats_df["mean"])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + stats_df["std"].iloc[i] + 0.5,
                        f'{mean_val:.2f}', ha='center', va='bottom',
                        fontsize=fontsize*0.8, fontweight='bold')

            ax.set_title(
                f"{item} - Balanced Accuracy by Action (Mean ± Std)", fontsize=fontsize)
            ax.set_xlabel("Action", fontname="Arial", fontsize=fontsize)
            ax.set_ylabel("Balanced Accuracy (%)",
                          fontname="Arial", fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)

            # all mean value points display
            overall_mean = item_df["Balanced_Accuracy"].mean()
            ax.axhline(overall_mean, color='red', linestyle='--', linewidth=2)
            ax.text(len(stats_df)-0.5, overall_mean + 0.5,
                    f'Overall Mean: {overall_mean:.2f}', color='red', fontsize=28, fontweight='bold', ha='right')

            # Y-axis range minimum input value-2.5 from maximum value+2.5 set
            y_min = item_df["Balanced_Accuracy"].min() - 2.5
            y_max = item_df["Balanced_Accuracy"].max() + 2.5
            plt.ylim(y_min, y_max)

            plt.tight_layout()

            # save
            for idx in ["png", "pdf", "svg"]:
                save_file = os.path.join(
                    save_path, f"{item}_BarPlot_MeanStd.{idx}")
                plt.savefig(save_file, bbox_inches='tight')
            plt.close(fig)

            # class_cols = [col for col in item_df.columns if "Class" in col]
            item_df = item_df[[c for c in item_df.columns if "Class" in c or c in [
                "ToFIM", "FromAction"]]].copy()
            if mode == "Complete_independence":
                item_df.rename(columns={"Class6_Accuracy": "Class:1-6",
                                        "Class7_Accuracy": "Class:7"}, inplace=True)
            elif mode == "Normal":
                item_df.rename(columns={"Class5_Accuracy": "Class:1-5",
                                        "Class6_Accuracy": "Class:6",
                                        "Class7_Accuracy": "Class:7"}, inplace=True)
            class_cols = [col for col in item_df.columns if "Class" in col]

            df_all = pd.DataFrame(
                columns=["ToFIM", "FromAction", "Class", "Accuracy"])
            for col in class_cols:
                temp_df = item_df[["ToFIM", "FromAction", col]].copy()
                temp_df.rename(columns={col: "Accuracy"}, inplace=True)
                temp_df["Class"] = col
                if df_all.empty:
                    df_all = temp_df.copy()
                else:
                    df_all = pd.concat([df_all, temp_df], axis=0)

            fig = plt.figure(figsize=(40, 10))
            ax = fig.add_subplot(111)
            bp = sns.boxplot(data=df_all,
                             x="FromAction",
                             y="Accuracy",
                             hue="Class",
                             patch_artist=True,
                             showmeans=True,
                             medianprops=dict(color='black',
                                              linewidth=3,
                                              linestyle='dashed'),
                             whiskerprops=dict(linewidth=2),
                             widths=0.25 if len(class_cols) == 3 else 0.35,
                             meanprops=dict(color='black',
                                            marker='+',
                                            markeredgecolor='red',
                                            markerfacecolor='red',
                                            markeredgewidth=5,
                                            markersize=20),
                             palette=colors,
                             fliersize=10,
                             )
            plt.xlabel("Action", fontname="Arial", fontsize=fontsize)
            plt.ylabel("Class-wise Accuracy [%]",
                       fontname="Arial", fontsize=fontsize)
            plt.xticks(ticks=range(
                len(df_all["FromAction"].unique())), labels=df_all["FromAction"].unique(), fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            if range_detailPlots == "y":
                plt.ylim(0, 100)
            else:
                y_min = df_all["Accuracy"].min()
                y_max = df_all["Accuracy"].max()
                ylim_min = (y_min // 5) * 5
                ylim_max = ((y_max // 5) + 1) * 5
                plt.ylim(ylim_min, ylim_max)
            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left',
                       borderaxespad=0, fontsize=fontsize)
            plt.tight_layout()

            for idx in ["png", "pdf", "svg"]:
                save_file = os.path.join(
                    save_path, f"{item}_ClassWiseAccuracies.{idx}")
                plt.savefig(save_file, bbox_inches='tight')
            plt.show()
            plt.close()
    else:
        fig, ax = plt.subplots(figsize=(40, 10))
        sns.boxplot(
            data=df,
            x="ToFIM",
            y="Balanced_Accuracy",
            showmeans=True,
            medianprops=dict(color='black',
                             linewidth=3,
                             linestyle='dashed'),
            whiskerprops=dict(linewidth=2),
            meanprops=dict(color='black',
                           marker='+',
                           markeredgecolor='red',
                           markerfacecolor='red',
                           markeredgewidth=5,
                           markersize=20),
            fliersize=10)
        ax.set_title("Each Item Balanced Accuracy", fontsize=fontsize)
        ax.set_xlabel("FIMItem", fontname="Arial", fontsize=fontsize)
        ax.set_ylabel("Balanced_Accuracy (%)",
                      fontname="Arial", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        for idx in ["png", "pdf", "svg"]:
            save_file = os.path.join(
                save_path, f"EachItemBalancedAccuracies.{idx}")
            plt.savefig(save_file, bbox_inches='tight')
        plt.close(fig)

        # class_cols = [col for col in item_df.columns if "Class" in col]
        df = df[[c for c in df.columns if "Class" in c or c in ["ToFIM"]]].copy()
        if mode == "Complete_independence":
            df.rename(columns={"Class6_Accuracy": "Class:1-6",
                               "Class7_Accuracy": "Class:7"}, inplace=True)
        elif mode == "Normal":
            df.rename(columns={"Class5_Accuracy": "Class:1-5",
                               "Class6_Accuracy": "Class:6",
                               "Class7_Accuracy": "Class:7"}, inplace=True)
        class_cols = [col for col in df.columns if "Class" in col]

        df_all = pd.DataFrame(columns=["ToFIM", "Class", "Accuracy"])
        for col in class_cols:
            temp_df = df[["ToFIM", col]].copy()
            temp_df.rename(columns={col: "Accuracy"}, inplace=True)
            temp_df["Class"] = col
            if df_all.empty:
                df_all = temp_df.copy()
            else:
                df_all = pd.concat([df_all, temp_df], axis=0)

        fig = plt.figure(figsize=(40, 10))
        # plt.rcParams['font.size'] = fontsize
        # plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Arial']})
        ax = fig.add_subplot(111)
        bp = sns.boxplot(data=df_all,
                         x="ToFIM",
                         y="Accuracy",
                         hue="Class",
                         patch_artist=True,
                         showmeans=True,
                         medianprops=dict(color='black',
                                          linewidth=3,
                                          linestyle='dashed'),
                         whiskerprops=dict(linewidth=2),
                         widths=0.25 if len(class_cols) == 3 else 0.35,
                         meanprops=dict(color='black',
                                        marker='+',
                                        markeredgecolor='red',
                                        markerfacecolor='red',
                                        markeredgewidth=5,
                                        markersize=20),
                         palette=colors,
                         fliersize=10,
                         )
        plt.xlabel("FIM Evaluation Item", fontname="Arial", fontsize=fontsize)
        plt.ylabel("Class-wise Accuracy [%]",
                   fontname="Arial", fontsize=fontsize)
        plt.xticks(ticks=range(
            len(df_all["ToFIM"].unique())), labels=df_all["ToFIM"].unique(), fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        if range_detailPlots == "y":
            plt.ylim(0, 100)
        else:
            y_min = df_all["Accuracy"].min()
            y_max = df_all["Accuracy"].max()
            ylim_min = (y_min // 5) * 5
            ylim_max = ((y_max // 5) + 1) * 5
            plt.ylim(ylim_min, ylim_max)

        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left',
                   borderaxespad=0, fontsize=fontsize)
        plt.tight_layout()

        for idx in ["png", "pdf", "svg"]:
            save_file = os.path.join(
                save_path, f"All_ClassWiseAccuracies.{idx}")
            plt.savefig(save_file, bbox_inches='tight')
        plt.show()
        plt.close()

    print("===============================================")
    print("BoxPlotsEachClass is finished.")
    print("===============================================")
