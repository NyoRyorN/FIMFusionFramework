import pandas as pd
from src.utils.names import FIMitems
import os

item_names = FIMitems()

data_path = "data/FilteredData/"
output_dir_path = os.path.join(
    "data", "fim_distribution"
)


def fim_distribution(directory_list: list[str] = None, csv_path: str = None):
    # FIMeach Score saveDataFrame initialize
    fim_df = pd.DataFrame(columns=["Directory"] + item_names)

    if csv_path is None:
        csv_path = "all"
    else:
        csv_path = os.path.basename(csv_path).split('_')[0]

    # Directorycolumn element from display
    for directory in directory_list:
        temp_row = {"Directory": directory}
        temp_path = os.path.join(data_path, directory, "FIM.csv")

        temp_df = pd.read_csv(temp_path, header=0)  # Item, Score
        for item in item_names:
            score = temp_df.loc[temp_df['Item']
                                == item, 'Score'].values
            if len(score) > 0:
                temp_row[item] = score[0] if not pd.isna(
                    score[0]) else -1
            else:
                temp_row[item] = None  # from case None set

        if fim_df.empty:
            fim_df = pd.DataFrame([temp_row])
        else:
            fim_df = pd.concat(
                [fim_df, pd.DataFrame([temp_row])], ignore_index=True)

    # CSV file save
    output_csv_path = os.path.join(
        output_dir_path, f"fim_dist_{csv_path}.csv"
    )
    fim_df.sort_values(by="Directory", inplace=True)
    fim_df.to_csv(output_csv_path, index=False, header=True)

    # each value csv output
    dist_output_csv = os.path.join(
        output_dir_path, f"fim_dist_distribution_{csv_path}.csv"
    )
    dist_df = pd.DataFrame(
        columns=["Item"] + [i for i in range(1, 8)] + ["Missing", "Total"])
    for item in item_names:
        if item == "Sum":
            continue  # Sum skip
        value_counts = fim_df[item].value_counts(dropna=False)
        total_count = len(fim_df)
        missing_count = value_counts.get(-1, 0) + value_counts.get(
            None, 0) + value_counts.get(float('nan'), 0)

        row = {"Item": item, "Total": total_count,
               "Missing": missing_count}
        for score in range(1, 8):
            row[score] = value_counts.get(score, 0)

        if dist_df.empty:
            dist_df = pd.DataFrame([row])
        else:
            dist_df = pd.concat(
                [dist_df, pd.DataFrame([row])], ignore_index=True)

    dist_df.to_csv(dist_output_csv, index=False, header=True)


def main():
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # input from CSV file path get
    csv_path = input("Enter the CSV file path: ").strip()
    if os.path.exists(csv_path):
        # CSV file pandas load
        df = pd.read_csv(csv_path, header=0)

        # Directorycolumn
        if 'Directory' not in df.columns:
            print("Error: 'Directory' column was not found")
            return

        directory_list = df['Directory'].tolist()

        fim_distribution(directory_list=directory_list, csv_path=csv_path)

    else:
        print("Calculating the FIM score distribution for all data. ")

        directory_list = [d for d in os.listdir(
            data_path) if os.path.isdir(os.path.join(data_path, d))]

        fim_distribution(directory_list=directory_list, csv_path=None)


if __name__ == "__main__":
    main()
