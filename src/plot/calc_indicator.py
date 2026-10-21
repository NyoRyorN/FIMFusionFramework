# 2Class case / Fusion
import os
import re
import pandas as pd
import sklearn.metrics as metrics
from src.utils.names import FIMitems, Tasks

headers = [
    'Action', 'FIMItem', 'Seed', 'Fold', 'Accuracy', 'BalancedAccuracy',
    'AccClass6', 'AccClass7', 'F1Score', 'MCC', 'Precision', 'AUCROC'
]
all_items = FIMitems()[:-1]
all_tasks = [t.split('-')[-1] for t in Tasks()[:-1]]


def calc_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate indicators based on the input DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing necessary data.

    Returns:
    pd.DataFrame: DataFrame with calculated indicators.
    """
    temp_df = pd.DataFrame(columns=headers[4:])

    # columns = df.columns.tolist()
    df["RealScore"] = df["RealScore"].apply(lambda x: int(x))

    # Calc Accuracy
    accuracy = metrics.accuracy_score(df["True"], df["Predicted"])
    temp_df.at[0, 'Accuracy'] = accuracy

    # Calc Balanced Accuracy
    balanced_accuracy = metrics.balanced_accuracy_score(
        df["True"], df["Predicted"])
    temp_df.at[0, 'BalancedAccuracy'] = balanced_accuracy

    # Calc AccClass6
    class6_df = df[df["RealScore"] <= 6]
    acc_class6 = metrics.accuracy_score(
        class6_df["True"], class6_df["Predicted"])
    temp_df.at[0, 'AccClass6'] = acc_class6

    # Calc AccClass7
    class7_df = df[df["RealScore"] > 6]
    acc_class7 = metrics.accuracy_score(
        class7_df["True"], class7_df["Predicted"])
    temp_df.at[0, 'AccClass7'] = acc_class7

    # Calc F1 Score
    f1_score = metrics.f1_score(
        df["True"], df["Predicted"], average='binary', pos_label=0)
    temp_df.at[0, 'F1Score'] = f1_score

    # Calc MCC
    mcc = metrics.matthews_corrcoef(df["True"], df["Predicted"])
    temp_df.at[0, 'MCC'] = mcc

    # Calc Precision
    precision = metrics.precision_score(
        df["True"], df["Predicted"], average='binary', pos_label=0)
    temp_df.at[0, 'Precision'] = precision

    # Calc AUCROC
    try:
        # Class 0 is positive, so invert y_true (0->1, 1->0) to use with Prob_0
        auc_roc = metrics.roc_auc_score(1 - df["True"], df["Prob_0"])
    except Exception as e:
        print(f"Error calculating AUC-ROC: {e}")
        auc_roc = float('nan')
    temp_df.at[0, 'AUCROC'] = auc_roc

    # print(df.head(5))
    # print(temp_df.head(1))
    # exit()

    return temp_df


def aggregate_indicators_each_fold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate indicators across folds by calculating the mean for each combination
    of Action, FIMItem, and Seed.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing indicators for each fold.

    Returns:
    pd.DataFrame: Aggregated DataFrame with mean indicators.
    """
    temp_df = df.copy().drop(columns=['Fold'], errors='ignore')
    aggregated_df = temp_df.groupby(
        ['Action', 'FIMItem', 'Seed'], as_index=False, observed=False).mean()
    return aggregated_df


def best_action_per_fimitem(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the best Action for each FIMItem based on the highest Accuracy.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing indicators.
    / Each_Summary_Indicator_Lists.csv

    Returns:
    pd.DataFrame: DataFrame with the best Action per FIMItem.
    """
    # print(df.head(5))
    # exit()
    best_df = pd.DataFrame(columns=df.columns)
    # Ensure FIMItem is a Series, not DataFrame (in case of duplicate column names)
    if isinstance(df['FIMItem'], pd.DataFrame):
        fim_items = df['FIMItem'].iloc[:, 0].drop_duplicates().values
    else:
        fim_items = df['FIMItem'].drop_duplicates().values
    for item in fim_items:
        if isinstance(df['FIMItem'], pd.DataFrame):
            item_df = df[df['FIMItem'].iloc[:, 0] == item]
        else:
            item_df = df[df['FIMItem'] == item]
        best_row = item_df.loc[item_df['BalancedAccuracy_Mean'].idxmax()]
        best_df = pd.concat(
            [best_df, best_row.to_frame().T], ignore_index=True) if not best_df.empty else best_row.to_frame().T
    return best_df


def summary_indicator(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a summary of indicators by calculating the mean across all entries.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing indicators.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: Summary DataFrame with mean indicators and each summary DataFrame.
    """
    summary_columns = [headers[1]]  # 'FIMItem', 'Seed', 'Fold'
    summary_columns.extend([[col+'_Mean', col+'_Std'] for col in headers[4:]])
    summary_df = pd.DataFrame(columns=summary_columns)
    each_summary_df = summary_df.copy()
    temp_df = df.copy().drop(columns=['Fold'], errors='ignore')
    unique_actions = temp_df['Action'].unique()
    if len(unique_actions) == 1:  # All
        # Seed aggregateItem
        cross_item_mean_df = temp_df.drop(columns=['Action', 'FIMItem']).groupby(
            ['Seed'], as_index=False, observed=False).mean()
        # cross_item_mean_df['Action'] = 'All'
        cross_item_mean_df['FIMItem'] = 'MeanOfItems'
        temp_df = pd.concat(
            [temp_df, cross_item_mean_df], ignore_index=True)
        # FIMItem aggregate
        temp_df = temp_df.drop(columns=['Action', 'Seed'], errors='ignore')
        mean_df = temp_df.groupby(
            ['FIMItem'], as_index=False, observed=False).mean().drop(columns=['Action'], errors='ignore')
        mean_df.columns = [
            col+'_Mean' if col not in ['Action', 'FIMItem', 'Seed', 'Fold'] else col for col in mean_df.columns]
        std_df = temp_df.groupby(
            ['FIMItem'], as_index=False, observed=False).std(ddof=1).drop(columns=['Action'], errors='ignore')
        std_df.columns = [
            col+'_Std' if col not in ['Action', 'FIMItem', 'Seed', 'Fold'] else col for col in std_df.columns]
        summary_df = mean_df.merge(
            std_df, on=['FIMItem'], how='inner')
    else:  # Action
        # Action&FIMItem aggregate
        mean_df = temp_df.groupby(
            ['Action', 'FIMItem'], as_index=False, observed=False).mean().drop(columns=['Seed'], errors='ignore')
        mean_df.columns = [
            col+'_Mean' if col not in ['Action', 'FIMItem', 'Seed', 'Fold'] else col for col in mean_df.columns]
        std_df = temp_df.groupby(
            ['Action', 'FIMItem'], as_index=False, observed=False).std(ddof=1).drop(columns=['Seed'], errors='ignore')
        std_df.columns = [
            col+'_Std' if col not in ['Action', 'FIMItem', 'Seed', 'Fold'] else col for col in std_df.columns]
        each_summary_df = mean_df.merge(
            std_df, on=['Action', 'FIMItem'], how='inner')
        best_summary_df = best_action_per_fimitem(each_summary_df)
        best_summary_df = best_summary_df.rename(
            columns={'Action': 'BestAction'})

        # best_summary_df BestAction and FIMItem combination row temp_df from extract
        cross_item_best_df = pd.DataFrame(columns=temp_df.columns)
        for index, row in best_summary_df.iterrows():
            action = row['BestAction']
            fim_item = row['FIMItem']
            matched_rows = temp_df[(temp_df['Action'] == action) &
                                   (temp_df['FIMItem'] == fim_item)]
            cross_item_best_df = pd.concat(
                [cross_item_best_df, matched_rows], ignore_index=True) if not cross_item_best_df.empty else matched_rows
        cross_item_best_df_mean = cross_item_best_df.drop(columns=['Action', 'FIMItem'], errors='ignore').groupby(
            ['Seed'], as_index=False, observed=False).mean()
        # cross_item_best_df_mean['Action'] = 'BestAction'
        cross_item_best_df_mean['FIMItem'] = 'MeanOfItems'
        mean_df = cross_item_best_df_mean.groupby(
            ['FIMItem'], as_index=False, observed=False).mean()
        mean_df.columns = [
            col+'_Mean' if col not in ['Action', 'FIMItem', 'Seed', 'Fold'] else col for col in mean_df.columns]
        std_df = cross_item_best_df_mean.groupby(
            ['FIMItem'], as_index=False, observed=False).std(ddof=1)
        std_df.columns = [
            col+'_Std' if col not in ['Action', 'FIMItem', 'Seed', 'Fold'] else col for col in std_df.columns]
        best_summary_df_cross_item = mean_df.merge(
            std_df, on=['FIMItem'], how='inner')
        best_summary_df_cross_item["BestAction"] = "BestOfActions"
        best_summary_df = pd.concat(
            [best_summary_df, best_summary_df_cross_item], ignore_index=True)

        # Action for FIMItem and aggregate
        temp_df = temp_df.drop(columns=['Action'], errors='ignore')
        temp_mean_df = temp_df.groupby(
            ['FIMItem', 'Seed'], as_index=False, observed=False).mean()

        # Seed aggregateItem
        cross_item_mean_df = temp_mean_df.drop(columns=['Action', 'FIMItem'], errors='ignore').groupby(
            ['Seed'], as_index=False, observed=False).mean()
        # cross_item_mean_df['Action'] = 'MeanOfActions'
        cross_item_mean_df['FIMItem'] = 'MeanOfItems'
        temp_mean_df = pd.concat(
            [temp_mean_df, cross_item_mean_df], ignore_index=True)

        # FIMItem aggregate
        mean_df = temp_mean_df.groupby(
            ['FIMItem'], as_index=False, observed=False).mean().drop(columns=['Seed'], errors='ignore')
        mean_df.columns = [
            col+'_Mean' if col not in ['Action', 'FIMItem', 'Seed', 'Fold'] else col for col in mean_df.columns]
        std_df = temp_mean_df.groupby(
            ['FIMItem'], as_index=False, observed=False).std(ddof=1).drop(columns=['Seed'], errors='ignore')
        std_df.columns = [
            col+'_Std' if col not in ['Action', 'FIMItem', 'Seed', 'Fold'] else col for col in std_df.columns]
        summary_df = mean_df.merge(
            std_df, on=['FIMItem'], how='inner')
        summary_df["BestAction"] = "MeanOfActions"

        # Ensure best_summary_df has the same columns in the same order as summary_df
        best_summary_df = best_summary_df[summary_df.columns]
        summary_df = pd.concat(
            [summary_df, best_summary_df], ignore_index=True)

    return summary_df, each_summary_df


def all_calc_indicator(rootDir: str):
    df = pd.DataFrame(columns=headers)
    output_path = os.path.join(rootDir, 'Indicator_Lists.csv')

    # 08-... / FIMItem or 01-..._to_08-... / Action_to_FIMItem
    temp_dirs_item = sorted([d for d in os.listdir(rootDir) if os.path.isdir(
        os.path.join(rootDir, d)) and re.match(r'^\d{2}-', d)])
    for temp_dir in temp_dirs_item:
        if '_to_' in temp_dir:
            item = temp_dir.split('_to_')[-1].split('-')[-1]
            action = temp_dir.split('_to_')[0].split('-')[-1]
        else:
            item = temp_dir.split('-', 1)[-1]
            action = 'All'
        temp_path_item = os.path.join(rootDir, temp_dir)
        # Stratified_Seed* / Seed
        temp_dirs_seed = sorted([d for d in os.listdir(temp_path_item) if os.path.isdir(
            os.path.join(temp_path_item, d)) and d.startswith('Stratified_Seed')])
        for temp_dir_seed in temp_dirs_seed:
            seed = int(re.match(r'Stratified_Seed(\d+)',
                       temp_dir_seed).group(1))
            temp_path_seed = os.path.join(temp_path_item, temp_dir_seed)
            # Fold* / Fold
            temp_dirs_fold = sorted([d for d in os.listdir(temp_path_seed) if os.path.isdir(
                os.path.join(temp_path_seed, d)) and d.startswith('Fold')])
            for temp_dir_fold in temp_dirs_fold:
                fold = int(re.match(r'Fold(\d+)', temp_dir_fold).group(1))
                temp_path_fold = os.path.join(temp_path_seed, temp_dir_fold)
                # Load CSV
                csv_path = os.path.join(temp_path_fold, 'PredictTruelabel.csv')
                if os.path.exists(csv_path):
                    temp_data_df = pd.read_csv(csv_path, header=0)
                    temp_df = calc_indicator(temp_data_df)
                    temp_df['Action'] = action
                    temp_df['FIMItem'] = item
                    temp_df['Seed'] = seed
                    temp_df['Fold'] = fold
                    temp_df = temp_df[headers]
                    df = pd.concat(
                        [df, temp_df], ignore_index=True) if not df.empty else temp_df
                else:
                    print(f"CSV file not found: {csv_path}")

    # Sort df according to specified order
    # Only use categories that exist in the data
    existing_items = [
        item for item in all_items if item in df['FIMItem'].unique()]
    df['FIMItem'] = pd.Categorical(
        df['FIMItem'], categories=existing_items, ordered=True)
    if 'Action' in df.columns:
        existing_actions = [
            'All'] + [task for task in all_tasks if task in df['Action'].unique()]
        existing_actions = [
            action for action in existing_actions if action in df['Action'].unique()]
        df['Action'] = pd.Categorical(
            df['Action'], categories=existing_actions, ordered=True)
    sort_columns = ['FIMItem']
    if 'Action' in df.columns:
        sort_columns.insert(0, 'Action')
    if 'Seed' in df.columns:
        sort_columns.append('Seed')
    if 'Fold' in df.columns:
        sort_columns.append('Fold')
    df = df.sort_values(by=sort_columns).reset_index(drop=True)

    df.to_csv(output_path, index=False)
    aggregated_df = aggregate_indicators_each_fold(df)

    # Sort aggregated_df according to specified order
    # 1. FIMItem by all_items order (only existing items)
    existing_items = [
        item for item in all_items if item in aggregated_df['FIMItem'].unique()]
    aggregated_df['FIMItem'] = pd.Categorical(
        aggregated_df['FIMItem'], categories=existing_items, ordered=True)
    # 2. Action by all_tasks order (if Action column exists, only existing actions)
    if 'Action' in aggregated_df.columns:
        existing_actions = [
            'All'] + [task for task in all_tasks if task in aggregated_df['Action'].unique()]
        existing_actions = [
            action for action in existing_actions if action in aggregated_df['Action'].unique()]
        aggregated_df['Action'] = pd.Categorical(
            aggregated_df['Action'], categories=existing_actions, ordered=True)
    # 3. Seed in ascending order (100 -> 1000)
    sort_columns = ['FIMItem']
    if 'Action' in aggregated_df.columns:
        sort_columns.insert(0, 'Action')
    if 'Seed' in aggregated_df.columns:
        sort_columns.append('Seed')
    aggregated_df = aggregated_df.sort_values(
        by=sort_columns).reset_index(drop=True)

    aggregated_output_path = os.path.join(
        rootDir, 'Aggregated_Indicator_Lists.csv')
    aggregated_df.to_csv(aggregated_output_path, index=False)

    # Create a copy without categorical types for summary calculation
    summary_input_df = aggregated_df.copy()
    summary_df, each_summary_df = summary_indicator(summary_input_df)

    # Sort summary_df according to specified order (only existing values)
    existing_items = [
        item for item in all_items if item in summary_df['FIMItem'].unique()] + ["MeanOfItems"]
    summary_df['FIMItem'] = pd.Categorical(
        summary_df['FIMItem'], categories=existing_items, ordered=True)
    if 'BestAction' in summary_df.columns:
        existing_actions = ['MeanOfActions'] + \
            [task for task in all_tasks if task in summary_df['BestAction'].unique()] + \
            ["BestOfActions"]
        existing_actions = [
            action for action in existing_actions if action in summary_df['BestAction'].unique()]
        summary_df['BestAction'] = pd.Categorical(
            summary_df['BestAction'], categories=existing_actions, ordered=True)
    sort_columns = ['FIMItem']
    if 'BestAction' in summary_df.columns:
        sort_columns.insert(0, 'BestAction')
    summary_df = summary_df.sort_values(by=sort_columns).reset_index(drop=True)

    # Sort each_summary_df according to specified order (only existing values)
    if not each_summary_df.empty:
        existing_items = [
            item for item in all_items if item in each_summary_df['FIMItem'].unique()]
        each_summary_df['FIMItem'] = pd.Categorical(
            each_summary_df['FIMItem'], categories=existing_items, ordered=True)
        if 'Action' in each_summary_df.columns:
            existing_actions = [
                'All'] + [task for task in all_tasks if task in each_summary_df['Action'].unique()]
            existing_actions = [
                action for action in existing_actions if action in each_summary_df['Action'].unique()]
            each_summary_df['Action'] = pd.Categorical(
                each_summary_df['Action'], categories=existing_actions, ordered=True)
        sort_columns = ['FIMItem']
        if 'Action' in each_summary_df.columns:
            sort_columns.insert(0, 'Action')
        each_summary_df = each_summary_df.sort_values(
            by=sort_columns).reset_index(drop=True)

    # Reorder columns: metadata columns first, then alternating _Mean and _Std for each metric
    metadata_cols = [col for col in summary_df.columns if not col.endswith(
        '_Mean') and not col.endswith('_Std')]
    metric_cols = []
    for metric in headers[4:]:
        if f'{metric}_Mean' in summary_df.columns:
            metric_cols.append(f'{metric}_Mean')
        if f'{metric}_Std' in summary_df.columns:
            metric_cols.append(f'{metric}_Std')
    summary_df = summary_df[metadata_cols + metric_cols]

    # Sort summary_df by FIMItem and BestAction (after column reordering)
    # This ensures MeanOfActions and best action rows are grouped by FIMItem
    existing_items = [
        item for item in all_items if item in summary_df['FIMItem'].unique()] + ["MeanOfItems"]
    summary_df['FIMItem'] = pd.Categorical(
        summary_df['FIMItem'], categories=existing_items, ordered=True)
    if 'BestAction' in summary_df.columns:
        existing_actions = ['MeanOfActions'] + \
            [task for task in all_tasks if task in summary_df['BestAction'].unique()] + \
            ["BestOfActions"]
        existing_actions = [
            action for action in existing_actions if action in summary_df['BestAction'].unique()]
        summary_df['BestAction'] = pd.Categorical(
            summary_df['BestAction'], categories=existing_actions, ordered=True)
        summary_df = summary_df.sort_values(
            by=['FIMItem', 'BestAction']).reset_index(drop=True)
    else:
        summary_df = summary_df.sort_values(
            by='FIMItem').reset_index(drop=True)

    summary_output_path = os.path.join(
        rootDir, 'Summary_Indicator_Lists.csv')
    summary_df.to_csv(summary_output_path, index=False)
    each_summary_output_path = os.path.join(
        rootDir, 'Each_Summary_Indicator_Lists.csv')
    if not each_summary_df.empty:
        # Use only columns that exist in each_summary_df
        each_metadata_cols = [col for col in each_summary_df.columns if not col.endswith(
            '_Mean') and not col.endswith('_Std')]
        each_summary_df = each_summary_df[each_metadata_cols + metric_cols]

        # Sort each_summary_df by FIMItem -> Action (after column reordering)
        existing_items = [
            item for item in all_items if item in each_summary_df['FIMItem'].unique()]
        each_summary_df['FIMItem'] = pd.Categorical(
            each_summary_df['FIMItem'], categories=existing_items, ordered=True)
        if 'Action' in each_summary_df.columns:
            existing_actions = [
                'All'] + [task for task in all_tasks if task in each_summary_df['Action'].unique()]
            existing_actions = [
                action for action in existing_actions if action in each_summary_df['Action'].unique()]
            each_summary_df['Action'] = pd.Categorical(
                each_summary_df['Action'], categories=existing_actions, ordered=True)
            each_summary_df = each_summary_df.sort_values(
                by=['FIMItem', 'Action']).reset_index(drop=True)
        else:
            each_summary_df = each_summary_df.sort_values(
                by='FIMItem').reset_index(drop=True)

        each_summary_df.to_csv(each_summary_output_path, index=False)


if __name__ == "__main__":
    mode = input(
        "Select mode: 1 for single directory, 2 for all directories in a root directory: ")
    if mode == '1':
        dirPath = input("Input the directory path: ")
        if not os.path.exists(dirPath):
            raise ValueError(f"Directory {dirPath} does not exist.")
        all_calc_indicator(dirPath)
        print(f"Processed directory: {dirPath}")
    elif mode == '2':
        rootDir = input("Input the root directory path: ")
        if not os.path.exists(rootDir):
            raise ValueError(f"Directory {rootDir} does not exist.")
        temp_dirs = [d for d in os.listdir(rootDir) if os.path.isdir(
            os.path.join(rootDir, d)) and 'Ablation' not in d]
        for temp_dir in temp_dirs:
            temp_path = os.path.join(rootDir, temp_dir)
            all_calc_indicator(temp_path)
            print(f"Processed directory: {temp_path}")
    else:
        print("Invalid mode selected.")
