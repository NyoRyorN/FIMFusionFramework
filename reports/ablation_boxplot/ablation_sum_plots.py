import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import japanize_matplotlib
import pandas as pd
from src.utils import Tasks, JPNTasks, FIMitems, setPltFont


def calMaxActionforSingleTask(path: str, item: str, df: pd.DataFrame, hue: str) -> int:
    temp_path = os.path.join(path, 'results.csv')
    if not os.path.exists(temp_path):
        print(f"File {temp_path} does not exist.")
        return None
    temp_df = pd.read_csv(temp_path, header=0)
    if "FromAction" not in df.columns:
        return None
    new_df = df.copy()
    temp_df = temp_df[["FromAction", "ToFIM", "Balanced_Accuracy"]]
    new_df = new_df.reindex(
        columns=["Seed", "ToFIM", "Balanced_Accuracy", "FromAction"])

    item_df = temp_df[temp_df["ToFIM"] == item].copy()
    new_df = new_df[new_df["ToFIM"] == item].copy()
    max_action = item_df[item_df["Balanced_Accuracy"].max(
    ) == item_df["Balanced_Accuracy"]]["FromAction"].values[0]
    new_df = new_df[new_df["FromAction"] == max_action].copy()
    new_df = new_df.copy()  # copy
    new_df["situation"] = hue + "_MaxAction"
    new_df["hue"] = hue
    new_df = new_df.drop(columns=["FromAction"])
    new_df = new_df[["Seed", "ToFIM", "Balanced_Accuracy", "situation", "hue"]]

    return new_df


def calMeanActionforSingleTask(item: str, df: pd.DataFrame, hue: str) -> int:
    if "FromAction" not in df.columns:
        return None
    temp_df = df[["Seed", "FromAction", "ToFIM", "Balanced_Accuracy"]].copy()
    temp_df = temp_df[temp_df["Seed"] != "Mean"]
    new_df = None
    seed_list = temp_df["Seed"].unique()
    new_df = pd.DataFrame(
        columns=["ToFIM", "Balanced_Accuracy", "situation", "hue"])
    for seed in seed_list:
        item_df = temp_df[temp_df["Seed"] == seed].copy()
        item_df = item_df[item_df["ToFIM"] == item].copy()
        if new_df.empty:
            new_row = pd.DataFrame({
                "Seed": [seed],
                "ToFIM": [item],
                "Balanced_Accuracy": [item_df["Balanced_Accuracy"].mean()],
                "situation": [hue + "_MeanAction"],
                "hue": [hue]
            })
            new_df = new_row
        else:
            new_row = pd.DataFrame({
                "Seed": [seed],
                "ToFIM": [item],
                "Balanced_Accuracy": [item_df["Balanced_Accuracy"].mean()],
                "situation": [hue + "_MeanAction"],
                "hue": [hue]
            })
            new_df = pd.concat([new_df, new_row], ignore_index=True)

    return new_df


def ablationSumPlots(rootDir: str):
    """
    Generate ablation summary plots for the given directory.
    """
    save_dir = os.path.join(rootDir, "AblationPlots")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    setPltFont()

    # tasks = Tasks()[:-1]
    # jpn_tasks = JPNTasks()[:-1]
    fontsize = 50

    items_all = FIMitems()
    items = items_all[8:-1]

    order_df = pd.read_csv(os.path.join(rootDir, "order.csv"), header=None)
    hue_order = order_df[0].values

    colors = [
        # Translated note
        "c", "m", "y", "g", "b", "r", "k",
        "orange",     # orange
        "purple",     # purple
        "brown",      # brown
        "pink",       # pink
        "gray",       # gray
        "olive",      # olive
        "navy",       # navy
        "crimson"     # crimson(deep red)
    ]
    colors = colors[:len(hue_order)]

    all_action_df = pd.DataFrame()

    new_hue_order = []

    for item in items:
        df = None
        # load
        for hue in hue_order:
            columns = ["ToFIM", "Balanced_Accuracy", "Seed", "FromAction"]
            # path = os.path.join(rootDir, hue)
            try:
                path_list = glob.glob(os.path.join(rootDir, hue+'*'))
                if not path_list:
                    print(f"No directory found for hue {hue}. Skipping...")
                    continue
                path = path_list[0]
                file_path = os.path.join(path, 'Accuracies.csv')
            except Exception as e:
                print(f"Error finding path for hue {hue}: {e}")
                continue

            # files = glob.glob(os.path.join(path, 'results.csv'))
            if not os.path.exists(file_path):
                print(
                    f"File {file_path} does not exist in {path}. Skipping...")
                continue

            try:
                df_temp = pd.read_csv(file_path, header=0)
                print(
                    f"Processing {file_path} for item {item} with hue {hue}.")

                # column
                available_columns = [
                    c for c in columns if c in df_temp.columns]
                df_temp = df_temp[available_columns]
                df_temp = df_temp[df_temp["Seed"] != "Mean"]
                df_mean = df_temp.copy()

                if "FromAction" in df_temp.columns:
                    df_mean = calMeanActionforSingleTask(item, df_mean, hue)
                    df_temp = calMaxActionforSingleTask(
                        path, item, df_temp, hue)
                    if df_temp is not None and df_mean is not None:
                        df_temp = pd.concat([df_mean, df_temp], axis=0)
                        new_hue_order.extend(
                            [f"{hue}_MeanAction", f"{hue}_MaxAction"])
                    else:
                        print(
                            f"Warning: Could not process action data for {hue}")
                        continue
                else:
                    df_temp = df_temp[df_temp["ToFIM"] == item].copy()
                    df_temp = df_temp[["Seed", "ToFIM",
                                       "Balanced_Accuracy"]].copy()
                    df_temp.loc[:, "situation"] = hue
                    df_temp.loc[:, "hue"] = hue
                    new_hue_order.append(hue)

                if df is None:
                    df = df_temp
                else:
                    df = pd.concat([df, df_temp], axis=0, ignore_index=True)

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

        # case processing row
        if df is None or df.empty:
            print(
                f"No data available for item {item}. Skipping plot generation.")
            continue

        df.to_csv(os.path.join(
            save_dir, f"result_all_{item}.csv"), index=False)

        fig = plt.figure(figsize=(40, 15))
        plt.rcParams['font.size'] = fontsize
        ax = fig.add_subplot(111)
        data = df.copy()
        data.loc[:, "situation"] = data["situation"].map(
            lambda x: x.replace("_", "\n"))
        new_hue_order_formatted = [x.replace("_", "\n") for x in new_hue_order]

        # safe sorting process
        try:
            data = data.sort_values("situation", ascending=True, key=lambda x: [
                                    new_hue_order_formatted.index(i) if i in new_hue_order_formatted else len(new_hue_order_formatted) for i in x])
        except Exception as e:
            print(f"Warning: Could not sort data for {item}: {e}")
            # sort case default sort
            data = data.sort_values("situation", ascending=True)

        bp = sns.boxplot(data=data,
                         y="Balanced_Accuracy",
                         x="situation",
                         patch_artist=True,
                         showmeans=True,
                         medianprops=dict(color='black',
                                          linewidth=3,
                                          linestyle='dashed'),
                         whiskerprops=dict(linewidth=2),
                         # widths=width,
                         meanprops=dict(color='white',
                                        marker='+',
                                        markeredgecolor='white',
                                        markerfacecolor='red',
                                        markeredgewidth=5,
                                        markersize=20),
                         hue="hue",
                         hue_order=hue_order,
                         linewidth=3,
                         fliersize=10,
                         palette=colors,
                         legend=False
                         )

        # eachbox plot top mean value display
        unique_situations = data["situation"].unique()
        for i, situation in enumerate(unique_situations):
            situation_data = data[data["situation"]
                                  == situation]["Balanced_Accuracy"]
            mean_val = situation_data.mean()
            max_val = situation_data.max()

            # box plot top(maximum value plus offset) mean value display
            ax.text(i, max_val + 1.0, f'{mean_val:.2f}',
                    ha='center', va='bottom',
                    fontsize=fontsize-10, fontweight='bold',
                    color='red')

        # plt.xlabel("ToFIM")
        # plt.xticks(ticks=data["ToFIM"].unique(), fontsize=fontsize)
        plt.ylabel("Balanced Accuracy (%)", fontsize=fontsize)
        plt.title(f"{item}'s Ablation Results", fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ylim(df["Balanced_Accuracy"].min() - 0.5,
                 df["Balanced_Accuracy"].max() + 0.5)
        plt.grid(axis='y', linestyle='--', linewidth=1, alpha=0.7)

        plt.savefig(os.path.join(
            save_dir, f"Balanced_Accuracy_{item}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(
            save_dir, f"Balanced_Accuracy_{item}.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(
            save_dir, f"Balanced_Accuracy_{item}.svg"), bbox_inches='tight')
        plt.close()

        # bar chart+standard deviation create
        stats_df = df.groupby(["situation", "hue"], observed=False)["Balanced_Accuracy"].agg(
            mean="mean", std=lambda values: values.std(ddof=1)).reset_index()

        # new_hue_order sort(remove duplicates)
        unique_new_hue_order = list(dict.fromkeys(new_hue_order))
        stats_df["situation"] = pd.Categorical(
            stats_df["situation"], categories=unique_new_hue_order, ordered=True)
        stats_df = stats_df.sort_values("situation")

        fig_bar = plt.figure(figsize=(40, 15))
        plt.rcParams['font.size'] = fontsize
        ax_bar = fig_bar.add_subplot(111)

        # huebased onassign colors
        bar_colors = []
        for _, row in stats_df.iterrows():
            hue_idx = list(hue_order).index(
                row["hue"]) if row["hue"] in hue_order else 0
            bar_colors.append(colors[hue_idx])

        # bar chart create(error bars)
        x_positions = range(len(stats_df))
        bars = ax_bar.bar(x_positions, stats_df["mean"],
                          yerr=stats_df["std"], capsize=10,
                          color=bar_colors, edgecolor='black', linewidth=1.5,
                          error_kw={'linewidth': 2, 'ecolor': 'red'})

        # each bar abovemean value display
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, stats_df["mean"], stats_df["std"])):
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2.,
                        height + std_val + 0.5,
                        f'{mean_val:.2f}', ha='center', va='bottom',
                        fontsize=fontsize-10, fontweight='bold')

        # axis labels and title
        ax_bar.set_xticks(x_positions)
        situation_labels = [s.replace("_", "\n")
                            for s in stats_df["situation"]]
        ax_bar.set_xticklabels(situation_labels, fontsize=fontsize)
        ax_bar.set_ylabel("Balanced Accuracy (%)", fontsize=fontsize)
        ax_bar.set_title(
            f"{item}'s Ablation Results (Mean ± Std)", fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        # Y-axis range set
        y_min = stats_df["mean"].min() - 2.5 - stats_df["std"].max()
        y_max = stats_df["mean"].max() + 2.5 + stats_df["std"].max()
        plt.ylim(y_min, y_max)
        plt.grid(axis='y', linestyle='--', linewidth=1, alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(
            save_dir, f"Balanced_Accuracy_{item}_BarPlot.png"), bbox_inches='tight')
        plt.savefig(os.path.join(
            save_dir, f"Balanced_Accuracy_{item}_BarPlot.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(
            save_dir, f"Balanced_Accuracy_{item}_BarPlot.svg"), bbox_inches='tight')
        plt.close()

        # CSV save
        stats_df.to_csv(os.path.join(
            save_dir, f"Balanced_Accuracy_{item}_Stats.csv"), index=False)

        all_action_df = pd.concat(
            [all_action_df, df], axis=0) if all_action_df.empty is False else df

    # AllAction Balanced Accuracy
    # all_action_df = df.copy()
    if not all_action_df.empty and "Seed" in all_action_df.columns and "situation" in all_action_df.columns:
        all_action_df = all_action_df.groupby(
            ["Seed", "situation", "hue"]).agg({"Balanced_Accuracy": "mean"}).reset_index()
        all_action_df = all_action_df[[
            "Seed", "situation", "Balanced_Accuracy", "hue"]]
        all_action_df["hue"] = pd.Categorical(
            all_action_df["hue"], categories=hue_order, ordered=True)
        all_action_df = all_action_df.sort_values("hue")
    else:
        print("Warning: Required columns not found for cross-action analysis")
        return

    fig = plt.figure(figsize=(40, 15))
    plt.rcParams['font.size'] = fontsize
    ax = fig.add_subplot(111)
    # bar plot save
    all_action_df_bar = all_action_df.copy()
    data = all_action_df.copy()
    data.loc[:, "situation"] = data["situation"].map(
        lambda x: x.replace("_", "\n"))
    new_hue_order_formatted = [x.replace("_", "\n") for x in new_hue_order]
    # safe sorting process
    try:
        data = data.sort_values("situation", ascending=True, key=lambda x: [
                                new_hue_order_formatted.index(i) if i in new_hue_order_formatted else len(new_hue_order_formatted) for i in x])
    except Exception as e:
        print(f"Warning: Could not sort cross-action data: {e}")
        data = data.sort_values("situation", ascending=True)

    bp = sns.boxplot(data=data,
                     y="Balanced_Accuracy",
                     x="situation",
                     patch_artist=True,
                     showmeans=True,
                     medianprops=dict(color='black',
                                      linewidth=3,
                                      linestyle='dashed'),
                     whiskerprops=dict(linewidth=2),
                     # widths=width,
                     meanprops=dict(color='white',
                                    marker='+',
                                    markeredgecolor='white',
                                    markerfacecolor='red',
                                    markeredgewidth=5,
                                    markersize=20),
                     hue="hue",
                     hue_order=hue_order,
                     linewidth=3,
                     fliersize=10,
                     palette=colors,
                     legend=False
                     )

    # eachbox plot top mean value display
    unique_situations = data["situation"].unique()
    for i, situation in enumerate(unique_situations):
        situation_data = data[data["situation"]
                              == situation]["Balanced_Accuracy"]
        mean_val = situation_data.mean()
        max_val = situation_data.max()

        # box plot top(maximum value plus offset) mean value display
        ax.text(i, max_val + 1.0, f'{mean_val:.2f}',
                ha='center', va='bottom',
                fontsize=fontsize-10, fontweight='bold',
                color='red')

    # plt.xlabel("ToFIM")
    # plt.xticks(ticks=data["ToFIM"].unique(), fontsize=fontsize)
    plt.ylabel("Balanced Accuracy (%)", fontsize=fontsize)
    plt.title(f"Cross-Action Ablation Results", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylim(data["Balanced_Accuracy"].min() - 0.5,
             data["Balanced_Accuracy"].max() + 0.5)
    plt.grid(axis='y', linestyle='--', linewidth=1, alpha=0.7)

    plt.savefig(os.path.join(
        save_dir, f"Balanced_Accuracy_CrossActions.png"), bbox_inches='tight')
    plt.savefig(os.path.join(
        save_dir, f"Balanced_Accuracy_CrossActions.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(
        save_dir, f"Balanced_Accuracy_CrossActions.svg"), bbox_inches='tight')
    plt.close()

    # Cross-Action bar chart+standard deviation create
    # all_action_df_bar (_format)
    stats_df_cross = all_action_df_bar.groupby(["situation", "hue"], observed=True)["Balanced_Accuracy"].agg(
        mean="mean", std=lambda values: values.std(ddof=1)).reset_index()

    # new_hue_order sort(remove duplicates)
    unique_new_hue_order = list(dict.fromkeys(new_hue_order))
    stats_df_cross["situation"] = pd.Categorical(
        stats_df_cross["situation"], categories=unique_new_hue_order, ordered=True)
    stats_df_cross = stats_df_cross.sort_values(
        "situation").dropna(subset=["situation"])
    # print(stats_df_cross)

    fig_bar_cross = plt.figure(figsize=(40, 15))
    plt.rcParams['font.size'] = fontsize
    ax_bar_cross = fig_bar_cross.add_subplot(111)

    # huebased onassign colors
    bar_colors_cross = []
    for _, row in stats_df_cross.iterrows():
        hue_idx = list(hue_order).index(
            row["hue"]) if row["hue"] in hue_order else 0
        bar_colors_cross.append(colors[hue_idx])

    # bar chart create(error bars)
    x_positions = range(len(stats_df_cross))
    bars = ax_bar_cross.bar(x_positions, stats_df_cross["mean"],
                            yerr=stats_df_cross["std"], capsize=10,
                            color=bar_colors_cross, edgecolor='black', linewidth=1.5,
                            error_kw={'linewidth': 2, 'ecolor': 'red'})

    # each bar abovemean value display
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, stats_df_cross["mean"], stats_df_cross["std"])):
        height = bar.get_height()
        ax_bar_cross.text(bar.get_x() + bar.get_width()/2.,
                          height + std_val + 0.5,
                          f'{mean_val:.2f}', ha='center', va='bottom',
                          fontsize=fontsize-10, fontweight='bold')

    # axis labels and title
    ax_bar_cross.set_xticks(x_positions)
    # situation from row from append(row)
    situation_labels_cross = [str(s).replace("\n", "_").replace("_", "\n")
                              for s in stats_df_cross["situation"]]
    ax_bar_cross.set_xticklabels(situation_labels_cross, fontsize=fontsize)
    ax_bar_cross.set_ylabel("Balanced Accuracy (%)", fontsize=fontsize)
    ax_bar_cross.set_title(
        f"Cross-Action Ablation Results (Mean ± Std)", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Y-axis range set
    y_min_cross = stats_df_cross["mean"].min(
    ) - 2.5 - stats_df_cross["std"].max()
    y_max_cross = stats_df_cross["mean"].max(
    ) + 2.5 + stats_df_cross["std"].max()
    plt.ylim(y_min_cross, y_max_cross)
    plt.grid(axis='y', linestyle='--', linewidth=1, alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(
        save_dir, f"Balanced_Accuracy_CrossActions_BarPlot.png"), bbox_inches='tight')
    plt.savefig(os.path.join(
        save_dir, f"Balanced_Accuracy_CrossActions_BarPlot.pdf"), bbox_inches='tight')
    plt.savefig(os.path.join(
        save_dir, f"Balanced_Accuracy_CrossActions_BarPlot.svg"), bbox_inches='tight')
    plt.close()

    # CSV save
    stats_df_cross.to_csv(os.path.join(
        save_dir, f"Balanced_Accuracy_CrossActions_Stats.csv"), index=False)

    all_action_df = all_action_df.copy()
    all_action_df.loc[:, "situation"] = all_action_df["situation"].str.replace(
        "\n", "_")
    all_action_df.to_csv(os.path.join(
        save_dir, "result_all_CrossActions.csv"), index=False)


if __name__ == "__main__":
    rootDir = input("Input the directory path: ")
    if not os.path.exists(rootDir):
        print(f"Directory {rootDir} does not exist.")
    else:
        ablationSumPlots(rootDir)
        print(f"Ablation summary plots generated in {rootDir}.")
