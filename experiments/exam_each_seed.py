"""
exam_each_seed.py - Seed-wise single-action experiment control script

This script runs single-action learning experiments for predicting
FIM (Functional Independence Measure) items across multiple seeds and statistically evaluates the results.

Key features:
1. Single-step learning from action tasks to FIM items
2. Experiment reproducibility checks across multiple seeds
3. Support for both K-fold cross-validation and holdout validation
4. Experiment tracking with Weights & Biases
5. Statistical result analysis and confusion-matrix generation

Experiment flow:
Action task -> FIM-item prediction model -> evaluation/statistical analysis
"""

import os
import glob
import yaml
import shutil
import argparse
import numpy as np
import pandas as pd
import torch
import wandb
from src.utils import *
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.training import Run as run
from src.data.feeder.results import OTHER_METRIC_KEYS
from src.data.feeder.results import Results as results
from src.data.feeder.feeder import Feeder
from src.data.dataset.make_single_dataset import makeSingleDataset as makeDataset


def main(configPath: str = "config/config.yaml", outputDir: str = "results"):
    """
    Main processing function for seed-wise single-action experiments.

    Controls the full experiment from config loading to result output:
    1. Load and validate the config file
    2. Resolve experiment parameters and task/item combinations
    3. Run training and evaluation for multiple seeds
    4. Analyze and visualize results statistically
    5. Write CSV outputs and keep the Notion-facing summary path available

    Args:
        configPath: Path to the config file (default: config/config.yaml)
        outputDir: Root directory under which the experiment results are saved (default: results)
    """
    # Configure matplotlib fonts for Japanese labels
    setPltFont()

    # Load the YAML config file
    # Read experiment, model, and data-processing settings
    situations = None

    try:
        with open(configPath, "r") as f:
            situations = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Error: Config file not found at config.yaml")
    except yaml.YAMLError as e:
        raise ValueError(f"Error: Failed to parse YAML file. Details: {e}")

    # Read base experiment settings
    team_name = situations["wandb"]["entity"]  # Weights & Biases team name

    # Resolve target tasks and FIM items
    tasks = Tasks()  # Get the full action-task list
    exam_tasks = [tasks[i]
                  for i in situations["action_task_indices"]]  # Target action task
    items = FIMitems()  # Get the full FIM-item list
    exam_items = [items[i]
                  for i in situations["fim_item_indices"]]  # Target FIM item

    # Resolve the experiment name and W&B project name
    exam_name, wb_project = decideExamName(situations, use_wb=True)

    # Experiment-start notification
    print(situations["device"], "Exam Start. " + exam_name)

    # Define DataFrame columns for overall results
    # Store accuracy results by seed, task, and FIM item
    columns = [
        "Seed",                # Seed value
        "FromAction",          # Input action-task name
        "ToFIM",              # Target FIM-item name
        "Accuracy",           # Overall accuracy
        "Balanced_Accuracy",  # Balanced accuracy (accounts for class imbalance)
        *[f"Class{c}_Accuracy" for c in situations["classes"]],  # Per-class accuracy
        *OTHER_METRIC_KEYS,   # F1/MCC/Precision/Recall/AUCROC
    ]

    # Initialize the DataFrame used to merge overall results
    df_all = pd.DataFrame([], columns=columns)

    # Create the root directory for experiment results
    rootPath = os.path.join(outputDir, exam_name)

    if not os.path.exists(rootPath):
        os.makedirs(rootPath)
        # Copy the config file to preserve experiment reproducibility
        shutil.copy(configPath, os.path.join(rootPath, "situations.yaml"))
        os.chmod(os.path.join(rootPath, "situations.yaml"), 0o444)  # Set the copied config to read-only

    # List for saving all experiment results
    all_results = []

    # Experiment loop over action-task and FIM-item combinations
    # Build and evaluate a prediction model from each action task to each FIM item
    for task in exam_tasks:
        for item in exam_items:
            print("============================================================")
            print("action:", task, "FIMitem:", item, " Start.")  # Experiment-start log
            print("============================================================")

            # Get the FIM-item index as a two-digit zero-padded string
            index = str(items.index(item)).zfill(2)
            # Generate the experiment name for the action-to-FIM-item combination
            each_exam_name = task + "_to_" + str(index) + "-" + item

            # Create the directory for this individual experiment result
            root_target_path = os.path.join(rootPath, each_exam_name)
            if not os.path.exists(root_target_path):
                os.makedirs(root_target_path)

            # Initialize the object for managing this individual experiment result
            # Track per-class accuracy and statistics
            result_class = results(situations["classes"])

            # Loop for checking experiment reproducibility across multiple seeds
            # Train and evaluate independently for each seed and check result stability
            for seed in situations["seeds"]:
                print("============================================================")
                print("seed", seed, " Start.")  # Seed-processing start log
                print("============================================================")

                # Set the seed for data splitting
                # Use 42 in split-fixed mode; otherwise use the configured seed
                decideSeed(
                    seed=42 if situations["seed_fix_mode"].lower() == "split" else seed)

                # Set tags for Weights & Biases
                # Metadata tags for experiment tracking and filtering
                tags = [task, item]  # Base tags (action task and FIM item)
                tags.extend(exam_name.split("_")[1:])  # Extract additional tags from the experiment name

                # Create the per-seed result directory
                target_path = os.path.join(
                    root_target_path, "Stratified_Seed" + str(seed)
                )
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
                    # Create the dataset (train/test split)
                    # Use stratified splitting to preserve each class distribution
                    makeDataset(
                        seed=42 if situations["seed_fix_mode"].lower(
                        ) == "split" else seed,  # Seed for data splitting
                        config=situations,        # Experiment config
                        action_task=task,        # Input action task
                        fim_item=item,           # Target FIM item
                        target_path=target_path,  # Data destination path
                    )

                # Initialize the object for managing per-fold results
                # Manage results for each K-fold fold or the single holdout split
                result_class_fold = results(situations["classes"])

                # Run the five K-fold validation loops
                for i in range(5):
                    # Set the seed for model training
                    # Use 42 in model-fixed mode; otherwise use the configured seed
                    decideSeed(
                        seed=42 if situations["seed_fix_mode"].lower() == "model" else seed)

                    # For K-fold validation, create a per-fold directory
                    target_path = os.path.join(
                        root_target_path,
                        "Stratified_Seed" + str(seed),
                        "Fold" + str(i)
                    )
                    print(
                        "============================================================")
                    print("Fold", i, " Start.")  # Fold-start log
                    print(
                        "============================================================")

                    # Create the Weights & Biases config object
                    # Merge experiment parameters and hyperparameters
                    wandb_config = decideWandbConfig(
                        situations, seed, item, task, i)
                    # Initialize Weights & Biases experiment tracking
                    # Track progress, record parameters, and visualize results
                    wb_run = wandb.init(
                        entity=team_name,                    # Team/organization name
                        project=wb_project,                  # Project name
                        name=each_exam_name + "_Seed" + str(seed),  # Experiment name
                        group=item,                          # Grouping (by FIM item)
                        tags=tags,                          # Metadata tags
                        # Job type (fold number or holdout)
                        job_type="Fold" + str(i),
                        config=wandb_config,                # Hyperparameter settings
                        settings=wandb.Settings(
                            save_code=False,        # Disable source-code saving
                            init_timeout=300)       # Initialization timeout (seconds)
                    ) if situations["wandb"]["use_wandb"] else None

                    # Initialize the DataLoader dictionary
                    data_loader = dict()

                    # Set training-data paths
                    train_path = target_path + "/train/data.npy"        # Training data (features)
                    train_dir_path = target_path + "/train/trainDir.csv"  # Data directory information
                    train_label_path = target_path + "/train/target.npy"  # Training labels

                    # Create the training dataset
                    dataset = Feeder(
                        data_path=train_path,              # Feature-data path
                        label_path=train_label_path,       # Label-data path
                        dir_path=train_dir_path,          # Directory-information path
                        # Data-augmentation usage setting
                        use_transform=situations["preprocess"]["use_data_transform"],
                    )

                    # Calculate class weights for imbalanced data
                    t_label = np.load(train_label_path)  # Load labels
                    label_dist = np.bincount(t_label)    # Calculate class distribution
                    class_weights = 1 / label_dist       # Use inverse weights to compensate for class imbalance
                    class_weights = torch.FloatTensor(class_weights).to(
                        situations["device"]
                    )

                    # Calculate per-sample weights for weighted sampling
                    data_weights = [class_weights[t_label[i]].to("cpu").item()
                                    for i in range(len(t_label))]

                    # Create the weighted random sampler
                    # Sample in a way that accounts for class imbalance
                    sampler = WeightedRandomSampler(
                        weights=data_weights,              # Per-sample weights
                        num_samples=len(data_weights),     # Number of samples
                        replacement=True,                  # Allow replacement sampling
                        generator=torch.Generator().manual_seed(
                            42 if situations["seed_fix_mode"].lower() == "model" else seed),
                    )

                    # Create the training DataLoader
                    if situations["trainings"]["is_weighted_random_sampler"]:
                        # Use weighted random sampling
                        data_loader["train"] = DataLoader(
                            dataset=dataset,
                            batch_size=situations["trainings"]["batch_size"],
                            sampler=sampler,
                            # Drop-last setting for mini-batches
                            drop_last=situations["trainings"]["is_drop_last"],
                        )
                    else:
                        # Use normal shuffle sampling
                        data_loader["train"] = DataLoader(
                            dataset=dataset,
                            batch_size=situations["trainings"]["batch_size"],
                            shuffle=True,                      # Enable random shuffling
                            # Drop-last setting for mini-batches
                            drop_last=situations["trainings"]["is_drop_last"],
                            generator=torch.Generator().manual_seed(
                                42 if situations["seed_fix_mode"].lower() == "model" else seed),
                        )

                    # Create the test DataLoader
                    test_path = target_path + "/test/data.npy"       # Test data (features)
                    test_dir_path = target_path + "/test/testDir.csv"  # Test directory information
                    test_label_path = target_path + "/test/target.npy"  # Test labels
                    data_loader["test"] = DataLoader(
                        dataset=Feeder(
                            data_path=test_path,               # Test feature data
                            label_path=test_label_path,        # Test label data
                            dir_path=test_dir_path,           # Test directory information
                            use_transform=False,              # Disable data augmentation during testing
                        ),
                        batch_size=situations["trainings"]["batch_size"],
                        shuffle=False,                        # Disable shuffling during testing
                    )

                    # Run single-action training and evaluation
                    # Train and evaluate a direct prediction model from the action task to the FIM item
                    (
                        accuracy,              # Overall accuracy
                        balanced_accuracy,     # Balanced accuracy (accounts for class imbalance)
                        each_class_accuracy,   # Per-class accuracy
                        other_metrics,         # F1/MCC/Precision/Recall/AUCROC
                        pred_labels,          # Predicted labels
                        true_labels,          # True labels
                    ) = run(
                        config=situations,              # Experiment config
                        trainloader=data_loader["train"],  # trainingdata
                        testloader=data_loader["test"],   # testdata
                        paths=target_path,              # Model save path
                        wb_run=wb_run,                 # Weights & Biases run instance
                        seed=42 if situations["seed_fix_mode"].lower(
                        ) == "model" else seed,        # Seed value setting
                    )

                    # Append fold results to the result-management object
                    result_class_fold.append(
                        accuracy=accuracy,                    # Overall accuracy
                        balanced_accuracy=balanced_accuracy,  # Balanced accuracy
                        each_class_accuracy=each_class_accuracy,  # Per-class accuracy
                        other_metrics=other_metrics,         # F1/MCC/Precision/Recall/AUCROC
                        pred_labels=pred_labels,             # Predicted labels
                        true_labels=true_labels,             # True labels
                    )

                    # Delete model files after training to save storage
                    model_pths = glob.glob(
                        os.path.join(target_path, "*_*.pth"))  # Find saved model files
                    for model_pth in model_pths:
                        os.remove(model_pth)  # Delete model files

                    # Clean temporary directories
                    temp_dirs = [d for d in glob.glob(
                        os.path.join(target_path, "*")) if os.path.isdir(d)]  # Get the directory list
                    for temp_dir in temp_dirs:
                        # Move CSV files from train/test directories to the parent directory
                        if "train" in temp_dir or "test" in temp_dir:
                            file = glob.glob(
                                os.path.join(temp_dir, "*.csv"))  # Find CSV files
                            for f in file:
                                shutil.move(f, target_path)  # Move files
                            shutil.rmtree(temp_dir)  # Delete directories

                    # Finish the Weights & Biases run
                    wandb.finish()

                    print(
                        "============================================================")
                    print("Fold", i, " Finished.")  # Fold-completion log
                    print(
                        "============================================================")

                # Integrate K-fold validation results
                # Integrate K-fold fold results and create the confusion matrix
                # Run statistical evaluation on combined predictions from multiple folds
                indexs = [str(i) for i in range(5)]  # Fold numbers (0-4)
                indexs.append("Mean")  # Append the mean row
                df_list = [indexs]
                last_results = [task, item]  # Task and item information
                df_columns = [
                    "Fold",
                    "Accuracy",
                    "Balanced_Accuracy",
                    # Per-class accuracy columns
                    *[f"Class{c}_Accuracy" for c in situations["classes"]],
                    *OTHER_METRIC_KEYS,
                ]

                # Create the K-fold combined confusion matrix
                # Generate a confusion matrix and metrics from all-fold predictions
                makeConfMatrixOnExam(
                    result_class_fold.all_true_labels,  # All-fold combined true labels
                    result_class_fold.all_pred_labels,  # All-fold combined predicted labels
                    situations["classes"],              # Class definitions
                    np.mean(result_class_fold.accuracies),  # Mean accuracy
                    # Mean balanced accuracy
                    np.mean(result_class_fold.balanced_accuracies),
                    os.path.join(root_target_path,
                                 "Stratified_Seed" + str(seed)),  # Output path
                    situations["model"]["name"],        # Model name
                    item,                              # FIM item
                    task,                              # Action task
                )

                # Process result statistics
                result_class_fold.appendMean()  # Calculate and append mean values
                last_results.extend(
                    result_class_fold.returnMeanList())  # Append mean results to the list
                # Append all results to the DataFrame input list
                df_list.extend(result_class_fold.returnLists())
                all_results.append(last_results)  # Append to the all-results list

                # Save results as a CSV file
                df = pd.DataFrame(df_list)
                df = df.T  # Transpose the DataFrame
                df.columns = df_columns
                df.to_csv(os.path.join(root_target_path, "Stratified_Seed" + str(seed),
                                       "Accuracies.csv"), index=False)

                # Append seed-level results to the overall results
                result_class.append(
                    accuracy=result_class_fold.mean_accuracies,        # Mean accuracy
                    balanced_accuracy=result_class_fold.mean_balanced_accuracies,  # Mean balanced accuracy
                    each_class_accuracy=result_class_fold.mean_each_class_accuracies,  # Mean per-class accuracy
                    other_metrics=result_class_fold.mean_other_metrics,  # Mean F1/MCC/Precision/Recall/AUCROC
                    pred_labels=result_class_fold.all_pred_labels,     # All predicted labels
                    true_labels=result_class_fold.all_true_labels,     # All true labels
                )

                print("============================================================")
                print("seed", seed, " Finished.")  # Seed-processing completion log
                print("============================================================")

            # Create the all-seed combined confusion matrix
            # Run final evaluation by combining results across multiple seeds
            indexs = situations["seeds"].copy()  # Copy the seed list
            indexs.append("Mean")  # Append the mean row
            df_list = [indexs]
            last_results = [task, item]  # Task and item information
            df_columns = [
                "Seed",
                "Accuracy",
                "Balanced_Accuracy",
                # Per-class accuracy columns
                *[f"Class{c}_Accuracy" for c in situations["classes"]],
                *OTHER_METRIC_KEYS,
            ]

            # Create the all-seed combined confusion matrix
            # Generate a confusion matrix and metrics from all-seed predictions
            makeConfMatrixOnExam(
                result_class.all_true_labels,   # All-seed combined true labels
                result_class.all_pred_labels,   # All-seed combined predicted labels
                situations["classes"],          # Class definitions
                np.mean(result_class.accuracies),  # Mean accuracy
                np.mean(result_class.balanced_accuracies),  # Mean balanced accuracy
                root_target_path,               # Output directory
                situations["model"]["name"],    # Model name
                item,                          # FIM item
                task,                          # Action task
            )

            # Process all-seed result statistics
            result_class.appendMean()  # Calculate and append mean values
            last_results.extend(result_class.returnMeanList())  # Append mean results to the list
            df_list.extend(result_class.returnLists())  # Append all results to the DataFrame input list
            all_results.append(last_results)  # Append to the all-results list

            # Save results as a CSV file
            df = pd.DataFrame(df_list)
            df = df.T  # Transpose the DataFrame
            df.columns = df_columns
            df.to_csv(root_target_path + "/Accuracies.csv", index=False)

            # Merge into the overall results
            # Save all accuracies - merge action-to-FIM-item results into the overall results
            df["FromAction"] = task  # Add the input-action-task column
            df["ToFIM"] = item       # Add the target-FIM-item column
            df.reindex(columns=columns)  # Adjust column order
            df_all = pd.concat([df_all, df], axis=0)  # Concatenate into the overall-results DataFrame

        print("============================================================")
        print("action:", task, ", FIMtask:", item, " Finished.")  # Combination-completion log
        print("============================================================")

        # Final save for the overall results
        # Save the CSV file containing all action-task/FIM-item combination results
        df_all.to_csv(os.path.join(rootPath, "Accuracies.csv"), index=False)

        # Extract only mean results and save them separately
        # Extract only the mean rows from per-seed results
        df_result = df_all.copy()[df_all["Seed"] == "Mean"]
        df_result.drop(columns="Seed", inplace=True)  # Remove the Seed column
        df_result.to_csv(os.path.join(rootPath, "results.csv"), index=False)

    # experimentcompletenotification
    print(situations["device"], exam_name + "\nAll Finished.")


if __name__ == "__main__":
    """
    Main execution block.

    Entry point for running the seed-wise single-action experiment.
    """
    parser = argparse.ArgumentParser(description="Seed-wise single-action experiment control script")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config.yaml",
        help="Path to the config file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Root directory under which the experiment results are saved (default: results)",
    )
    args = parser.parse_args()
    main(configPath=args.config, outputDir=args.output)  # Run the main experiment process
