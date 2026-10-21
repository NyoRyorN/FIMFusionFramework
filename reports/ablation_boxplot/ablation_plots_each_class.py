import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import japanize_matplotlib
import pandas as pd
from src.utils import Tasks, JPNTasks, FIMitems, setPltFont


def calMaxActionforSingleTask(path: str, item: str, df: pd.DataFrame, hue: str) -> int:
    temp_path = os.path.join(path, 'Summary.csv')
    if not os.path.exists(temp_path):
        print(f"File {temp_path} does not exist.")
        return None
    # read Summary.csv
    temp_df = pd.read_csv(temp_path, header=0)
    if "FromAction" not in df.columns:
        return None
    # Accuracies.csvbased onmaximum accuracy row
    new_df = df.copy()
    action = temp_df[temp_df["FIMItem"] ==
                     item]["Max_Balanced_Acc_Action"].values[0]
    new_df = new_df[(new_df["FromAction"] == action) & (
        new_df["ToFIM"] == item) & (new_df["Seed"] != "Mean")].copy()

    return_df = None

    columns = [c for c in new_df.columns if "Class" in c]
    for col in columns:
        temp_df = new_df[["ToFIM", col]].copy()
        temp_df.loc[:, "situation"] = col.replace("_Accuracy", "") if col != columns[0] else col.replace(
            "Class", "Class1-").replace("_Accuracy", "")
        temp_df.loc[:, "hue"] = hue + "_MaxAction"
        temp_df.rename(
            columns={col: "Class-wise_Accuracy"}, inplace=True)

        if return_df is None:
            return_df = temp_df
        else:
            return_df = pd.concat([return_df, temp_df],
                                  axis=0, ignore_index=True)

    return return_df


def calMeanActionforSingleTask(item: str, df: pd.DataFrame, hue: str) -> int:
    if "FromAction" not in df.columns:
        return None
    # Accuracies.csvbased onmaximum accuracy row
    columns = [c for c in df.columns if "Class" in c]
    new_df = df.copy()
    new_df = new_df[(new_df["ToFIM"] == item) &
                    (new_df["Seed"] != "Mean")].copy()
    new_df = new_df.drop(columns=["FromAction"])
    new_df = new_df.groupby(["Seed"], as_index=False).mean(numeric_only=True)
    return_df = None
    temp_df = pd.DataFrame()
    for col in columns:
        temp = new_df[["ToFIM", col]
                      ].copy() if "ToFIM" in new_df.columns else new_df[[col]].copy()
        if "ToFIM" not in new_df.columns:
            temp.loc[:, "ToFIM"] = item
        temp.loc[:, "situation"] = col.replace("_Accuracy", "") if col != columns[0] else col.replace(
            "Class", "Class1-").replace("_Accuracy", "")
        temp.loc[:, "hue"] = hue + "_MeanAction"
        temp.rename(
            columns={col: "Class-wise_Accuracy"}, inplace=True)

        if return_df is None:
            return_df = temp
        else:
            return_df = pd.concat([return_df, temp], axis=0, ignore_index=True)

    return return_df


def ablationPlotsEachClass(rootDir: str):
    """
    Generate ablation summary plots for the given directory.
    """
    save_dir = os.path.join(rootDir, "AblationPlotsEachClass")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    setPltFont()

    # tasks = Tasks()[:-1]
    # jpn_tasks = JPNTasks()[:-1]
    fontsize = 50

    items_all = FIMitems()
    items = items_all[8:-1]

    df = pd.read_csv(os.path.join(rootDir, "order.csv"), header=None)
    hue_order = df[0].values

    colors = [
        "green",
        "blue",
        "red",
        "black",
        "cyan",  # Translated note
        "magenta",  # Translated note
        "yellow",
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

    for item in items:
        df = None
        # load
        for hue in hue_order:
            # columns = ["ToFIM", "Balanced_Accuracy", "Seed"]
            path = glob.glob(os.path.join(rootDir, hue+'*'))[0]
            file_path = os.path.join(path, 'Accuracies.csv')
            # files = glob.glob(os.path.join(path, 'results.csv'))
            if not os.path.exists(file_path):
                print(
                    f"File {file_path} does not exist in {path}. Skipping...")
                continue

            df_temp = pd.read_csv(file_path, header=0)
            df_temp.drop(columns=["Accuracy"], inplace=True)
            if "FromAction" in df_temp.columns:
                df_max = calMaxActionforSingleTask(
                    path, item, df_temp, hue)
                df_mean = calMeanActionforSingleTask(
                    item, df_temp, hue)
                if df_max is not None:
                    df = df_max if df is None else pd.concat(
                        [df, df_max], axis=0, ignore_index=True)
                if df_mean is not None:
                    df = df_mean if df is None else pd.concat(
                        [df, df_mean], axis=0, ignore_index=True)
            else:
                print(
                    f"Processing {file_path} for item {item} with hue {hue}.")
                columns = [c for c in df_temp.columns if "Class" in c]
                for col in columns:
                    df_each = df_temp[df_temp["Seed"] != "Mean"].copy()
                    df_each = df_each[df_each["ToFIM"] == item].copy()
                    df_each = df_each[["ToFIM", col]].copy()
                    df_each.loc[:, "situation"] = col.replace("_Accuracy", "") if col != columns[0] else col.replace(
                        "Class", "Class1-").replace("_Accuracy", "")
                    df_each.loc[:, "hue"] = hue
                    df_each.rename(
                        columns={col: "Class-wise_Accuracy"}, inplace=True)
                    df = df_each if df is None else pd.concat(
                        [df, df_each], axis=0, ignore_index=True)

            df.to_csv(os.path.join(
                save_dir, f"result_class-wise_{item}.csv"), index=False)

        # ★ hue_order and df["hue"].unique() processing
        unique_hues = df["hue"].unique()

        # hue_order build
        expanded_hue_order = []
        expanded_colors = []

        for i, base_hue in enumerate(hue_order):
            # base_hue hue search
            matching_hues = [h for h in unique_hues if h.startswith(base_hue)]

            if not matching_hues:
                # all
                if base_hue in unique_hues:
                    matching_hues = [base_hue]

            if matching_hues:
                # sortorder (MeanAction, MaxAction)
                matching_hues.sort(key=lambda x: (
                    0 if x == base_hue else
                    1 if "_MeanAction" in x else
                    2 if "_MaxAction" in x else 3
                ))

                for j, matched_hue in enumerate(matching_hues):
                    expanded_hue_order.append(matched_hue)

                    # :base_hue from in
                    base_color = colors[i % len(colors)]
                    if matched_hue == base_hue:
                        expanded_colors.append(base_color)
                    elif "_MeanAction" in matched_hue:
                        # MeanAction in
                        expanded_colors.append(
                            f"light{base_color}" if f"light{base_color}" in plt.colormaps() else base_color)
                    elif "_MaxAction" in matched_hue:
                        # MaxAction in
                        expanded_colors.append(
                            f"dark{base_color}" if f"dark{base_color}" in plt.colormaps() else base_color)
                    else:
                        expanded_colors.append(base_color)
            else:
                print(f"Warning: No matching hue found for {base_hue}")

        # adjust
        while len(expanded_colors) < len(expanded_hue_order):
            expanded_colors.extend(
                colors[:len(expanded_hue_order) - len(expanded_colors)])
        expanded_colors = expanded_colors[:len(expanded_hue_order)]

        fig = plt.figure(figsize=(40, 15))
        plt.rcParams['font.size'] = fontsize
        ax = fig.add_subplot(111)
        data = df
        data["situation"] = data["situation"].map(
            lambda x: x.replace("_", "\n"))

        bp = sns.boxplot(data=data,
                         y="Class-wise_Accuracy",
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
                         hue_order=expanded_hue_order,
                         linewidth=3,
                         fliersize=10,
                         palette=expanded_colors,
                         )

        # plt.xlabel("ToFIM")
        # plt.xticks(ticks=data["ToFIM"].unique(), fontsize=fontsize)
        plt.ylabel("Class-wise Accuracy (%)", fontsize=fontsize)
        plt.title(f"{item}'s Ablation Results", fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.ylim(df["Class-wise_Accuracy"].min() - 2,
                 df["Class-wise_Accuracy"].max() + 2)
        plt.grid(axis='y', linestyle='--', linewidth=1, alpha=0.7)
        plt.legend(fontsize=fontsize, loc='upper right',
                   bbox_to_anchor=(1.25, 1))

        plt.savefig(os.path.join(
            save_dir, f"Class-wise_Accuracy_{item}.png"), bbox_inches='tight')
        plt.savefig(os.path.join(
            save_dir, f"Class-wise_Accuracy_{item}.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(
            save_dir, f"Class-wise_Accuracy_{item}.svg"), bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    rootDir = input("Input the directory path: ")
    if not os.path.exists(rootDir):
        print(f"Directory {rootDir} does not exist.")
    else:
        ablationPlotsEachClass(rootDir)
        print(f"Ablation summary plots generated in {rootDir}.")
