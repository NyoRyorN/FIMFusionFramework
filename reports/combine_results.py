import pandas as pd
import glob

# Define the file paths
file_paths = input("Enter the file paths (e.g., /path/to/your/files): ")
csv_paths = sorted(glob.glob(f"{file_paths}/*-*/"))
print(csv_paths)
all_df = pd.DataFrame()
for csv_path in csv_paths:
    csv_file = f"{csv_path}/Accuracies.csv"

    # Read the CSV file
    df = pd.read_csv(csv_file, header=0)
    if "Mean" not in df["Seed"].values:
        # Each Column's Mean
        mean_values = df[df.columns[1:]].mean()

        # Add Mean row to the DataFrame
        mean_row = pd.DataFrame(mean_values).T
        mean_row.insert(0, df.columns[0], 'Mean')
        df = pd.concat([df, mean_row], ignore_index=True)
        # Save the updated DataFrame back to the CSV file
        df.to_csv(csv_file, index=False)
    df["ToFIM"] = csv_path.split("/")[-2].split("-")[-1]

    # Append to the all_df DataFrame
    if all_df.empty:
        all_df = df
    else:
        all_df = pd.concat([all_df, df], ignore_index=True)
    # Save the combined DataFrame to a new CSV file

all_df = all_df.reindex(columns=["ToFIM"] + list(all_df.columns[:-1]))
all_df.to_csv(f"{file_paths}/Accuracies.csv", index=False)
results_df = all_df[all_df["Seed"] == "Mean"]
results_df.drop(columns=["Seed"], inplace=True)
results_df.to_csv(f"{file_paths}/results.csv", index=False)
