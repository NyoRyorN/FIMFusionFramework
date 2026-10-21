import os
import yaml
import shutil
import argparse
import numpy as np
import pandas as pd
import torch
import wandb
from src.utils import *
from src.utils.delete_after import DeleteAfter
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.training import Run as run
from src.data.dataset import makeSomeTasksDataset
from src.data.feeder.feeder import Feeder
from src.data.feeder.results import Results as results
from src.data.feeder.results import OTHER_METRIC_KEYS


def main(configPath: str = "config/configSomeTask.yaml", outputDir: str = "results"):
    """Main execution function for fusion experiments.

    Runs fusion-model training and evaluation for multiple FIM items
    and multiple seeds, then saves and records the results.

    Experiment flow:
    1. Load the config file
    2. Loop over each FIM item
    3. Loop over training and evaluation for each seed
    4. Evaluate with K-fold cross-validation or a single split
    5. Aggregate and save results

    Args:
        configPath: Path to the config file (default: config/configSomeTask.yaml)
        outputDir: Root directory under which the experiment results are saved (default: results)
    """
    # Initialize font settings
    setPltFont()

    # Load the config file
    situations = None
    try:
        with open(configPath, "r") as f:
            situations = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Error: Config file not found at config.yaml")
    except yaml.YAMLError as e:
        raise ValueError(f"Error: Failed to parse YAML file. Details: {e}")

    # Read experiment settings
    wandb_entity = situations["wandb"]["entity"]
    task_index = situations["combined_action_indices"]  # Target action indices
    tasks = Tasks()
    exam_tasks = [tasks[i] for i in task_index]  # Target action list

    # Build a combined action-name string
    act_idx = ""
    for task in exam_tasks:
        act_idx += task[:2] + "_"

    items = FIMitems()
    exam_items = [items[i]
                  for i in situations["fim_item_indices"]]  # Target FIM item
    exam_name, wb_project = decideExamName(
        situations, use_wb=True)  # Resolve the experiment name and project name

    # Define DataFrame columns for result recording
    columns = [
        "Seed",
        "ToFIM",
        "Accuracy",
        "Balanced_Accuracy",
        *[f"Class{c}_Accuracy" for c in situations["classes"]],
        *OTHER_METRIC_KEYS,
    ]

    # Initialize the DataFrame used to record all results
    df_all = pd.DataFrame([], columns=columns)

    # Create the root directory for saved results
    rootPath = os.path.join(outputDir, exam_name)

    if not os.path.exists(rootPath):
        os.makedirs(rootPath)
        # Copy the config file to preserve experiment reproducibility
        shutil.copy(configPath, os.path.join(rootPath, "situations.yaml"))
        os.chmod(os.path.join(rootPath, "situations.yaml"), 0o444)  # Set the copied config to read-only

    # List for recording all results
    all_results = []

    # Experiment loop for each FIM item
    for item in exam_items:
        print("============================================================")
        print("FIMitem:", item, " Start.")
        print("============================================================")
        # Get the FIM-item index as a two-digit string
        index = str(items.index(item)).zfill(2)
        each_exam_name = str(index) + "-" + item

        # Create the directory dedicated to this FIM item
        root_target_path = os.path.join(rootPath, each_exam_name)
        if not os.path.exists(root_target_path):
            os.makedirs(root_target_path)

        # Initialize the result-recording class
        result_class = results(situations["classes"])
        config_upload = True  # Config-file upload flag

        # Experiment loop for each seed
        for seed in situations["seeds"]:
            print("============================================================")
            print("seed", seed, " Start.")
            print("============================================================")

            # Set the seed according to the split-fixing mode
            decideSeed(
                seed=42 if situations["seed_fix_mode"].lower() == "split" else seed)

            # Create the directory dedicated to this seed
            target_path = os.path.join(
                root_target_path, "Stratified_Seed" + str(seed)
            )
            if not os.path.exists(target_path):
                os.makedirs(target_path)
                # Create the dataset
                makeSomeTasksDataset(
                    seed=42 if situations["seed_fix_mode"].lower(
                    ) == "split" else seed,
                    config=situations,
                    action_tasks=exam_tasks,
                    fim_item=item,
                    target_path=target_path,
                )

            # Initialize the per-fold result-recording class
            result_class_fold = results(situations["classes"])

            # Experiment loop for K-fold cross-validation or a single split
            for i in range(5):
                # Set the seed for model training
                decideSeed(
                    seed=42 if situations["seed_fix_mode"].lower() == "model" else seed)

                # Create a directory dedicated to this fold
                target_path = os.path.join(
                    root_target_path,
                    "Stratified_Seed" + str(seed),
                    "Fold" + str(i)
                )
                print(
                    "============================================================")
                print("Fold", i, " Start.")
                print(
                    "============================================================")

                # Create the Weights & Biases config
                wandb_config = decideWandbConfig(
                    situations, seed, item, None, i)
                # Set tags
                tags = exam_name.split("_")[1:]
                tags.append(wandb_config["fusion_type"])

                # Initialize the Weights & Biases run
                wb_run = wandb.init(
                    entity=wandb_entity,
                    project=wb_project,
                    group=item,
                    name=wandb_config["fusion_type"] + "_" + str(seed),
                    config=wandb_config,
                    job_type="Fold" + str(i),
                    tags=tags,
                    settings=wandb.Settings(
                        save_code=False,
                        init_timeout=500)
                ) if situations["wandb"]["use_wandb"] else None
                # Upload the config file only on the first run
                if config_upload and situations["wandb"]["use_wandb"]:
                    wb_run.save(
                        os.path.join(rootPath, "situations.yaml")
                    )
                    config_upload = False

                # Train the fusion model end-to-end
                print("============================================================")
                print("Training Start.")
                print("============================================================")

                # Prepare the DataLoader for fusion learning
                data_loader = dict()
                train_path = os.path.join(
                    target_path, "train", "data.npy")  # All-action integrated data
                train_dir_path = target_path + "/train/trainDir.csv"  # Data directory information
                train_label_path = target_path + "/train/target.npy"  # Label data

                # Create the dataset for fusion learning
                dataset = Feeder(
                    data_path=train_path,
                    label_path=train_label_path,
                    dir_path=train_dir_path,
                    use_transform=situations["preprocess"]["use_data_transform"],
                )

                # Calculate fusion-training class weights for imbalanced data
                t_label = np.load(train_label_path)
                label_dist = np.bincount(t_label)  # Calculate class distribution
                class_weights = 1 / label_dist  # Use inverse weights to compensate for class imbalance
                class_weights = torch.FloatTensor(class_weights).to(
                    situations["device"]
                )

                # Create the weighted sampler using sample-level weights
                data_weights = [class_weights[t_label[i]]
                                for i in range(len(t_label))]
                sampler = WeightedRandomSampler(
                    weights=data_weights,
                    num_samples=len(data_weights),
                    replacement=True,
                    generator=torch.Generator().manual_seed(
                        42 if situations["seed_fix_mode"].lower() == "model" else seed),
                )

                # Create the training DataLoader for fusion learning
                # Use weighted sampling when the weighted sampler is enabled
                if situations["trainings"]["is_weighted_random_sampler"]:
                    data_loader["train"] = DataLoader(
                        dataset=dataset,
                        batch_size=situations["trainings"]["batch_size"],
                        sampler=sampler,
                        drop_last=situations["trainings"]["is_drop_last"],
                    )
                else:
                    # Use normal shuffling when the weighted sampler is disabled
                    data_loader["train"] = DataLoader(
                        dataset=dataset,
                        batch_size=situations["trainings"]["batch_size"],
                        shuffle=True,
                        drop_last=situations["trainings"]["is_drop_last"],
                        generator=torch.Generator().manual_seed(
                            42 if situations["seed_fix_mode"].lower() == "model" else seed),
                    )

                # Create the test DataLoader
                test_path = target_path + "/test/data.npy"  # Test-data path
                test_dir_path = target_path + "/test/testDir.csv"  # Test directory information
                test_label_path = target_path + "/test/target.npy"  # Test labels
                data_loader["test"] = DataLoader(
                    dataset=Feeder(
                        data_path=test_path,
                        label_path=test_label_path,
                        dir_path=test_dir_path,
                        use_transform=False,  # Disable data transformation during testing
                    ),
                    batch_size=situations["trainings"]["batch_size"],
                    shuffle=False,  # Do not shuffle during testing
                )

                # Run fusion-model training and evaluation
                (
                    accuracy,
                    balanced_accuracy,
                    each_class_accuracy,
                    other_metrics,
                    pred_labels,
                    true_labels,
                ) = run(
                    config=situations,                 # Experiment config
                    trainloader=data_loader["train"],  # Data for fusion learning
                    testloader=data_loader["test"],    # Test data for evaluation
                    paths=target_path,                  # Model save path
                    wb_run=wb_run,                     # Weights & Biases run instance
                    seed=42 if situations["seed_fix_mode"].lower(
                    ) == "model" else seed,            # Seed value setting
                )

                # Save fold results
                # Append each fold's prediction accuracy and detailed results to the result list
                result_class_fold.append(
                    accuracy=accuracy,                    # Overall accuracy
                    balanced_accuracy=balanced_accuracy,  # Balanced accuracy
                    each_class_accuracy=each_class_accuracy,  # Per-class accuracy
                    other_metrics=other_metrics,         # F1/MCC/Precision/Recall/AUCROC
                    pred_labels=pred_labels,             # Predicted labels
                    true_labels=true_labels,             # True labels
                )

                # Delete model files and temporary directories after training
                # (key is absent in older configs; default to keeping files)
                if situations["trainings"].get("delete_model_after_training", False):
                    DeleteAfter(target_path)

                # Finish the Weights & Biases run
                if wb_run is not None:
                    wb_run.finish()

                print(
                    "============================================================")
                print("Fold", i, " Finished.")  # Fold-completion message
                print(
                    "============================================================")

            # Aggregate K-fold validation results
            # Create the confusion matrix and result summary
            # Integrate K-fold fold results and run final evaluation
            indexs = [str(i) for i in range(5)]  # Fold numbers (0-4)
            indexs.append("Mean")  # Append the mean row
            df_list = [indexs]
            last_results = [item]  # FIM-item information
            df_columns = [
                "Fold",
                "Accuracy",
                "Balanced_Accuracy",
                # Per-class accuracy columns
                *[f"Class{c}_Accuracy" for c in situations["classes"]],
                *OTHER_METRIC_KEYS,
            ]

            # Create the K-fold validation confusion matrix
            # Generate a confusion matrix and metrics from all-fold predictions
            makeConfMatrixOnExam(
                result_class_fold.all_true_labels,  # All-fold combined true labels
                result_class_fold.all_pred_labels,  # All-fold combined predicted labels
                situations["classes"],              # Class definitions
                np.mean(result_class_fold.accuracies),  # Mean accuracy
                np.mean(result_class_fold.balanced_accuracies),  # Mean balanced accuracy
                os.path.join(root_target_path,
                             "Stratified_Seed" + str(seed)),  # Output path
                situations["model"]["name"],        # Model name
                item,                              # FIM item
            )

            # Calculate result means and build the result list
            result_class_fold.appendMean()  # Append mean values
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

            # Append seed-level results to the FIM-item aggregate
            result_class.append(
                accuracy=result_class_fold.mean_accuracies,
                balanced_accuracy=result_class_fold.mean_balanced_accuracies,
                each_class_accuracy=result_class_fold.mean_each_class_accuracies,
                other_metrics=result_class_fold.mean_other_metrics,
                pred_labels=result_class_fold.all_pred_labels,
                true_labels=result_class_fold.all_true_labels,
            )
            print("============================================================")
            print("seed", seed, " Finished.")  # Seed-processing completion message
            print("============================================================")

        # Create the all-seed combined confusion matrix
        # Run final evaluation by combining results across multiple seeds
        indexs = situations["seeds"].copy()  # Copy the seed list
        indexs.append("Mean")  # Append the mean row
        df_list = [indexs]
        last_results = [item]  # FIM-item information
        df_columns = [
            "Seed",
            "Accuracy",
            "Balanced_Accuracy",
            *[f"Class{c}_Accuracy" for c in situations["classes"]],  # Per-class accuracy columns
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
            item)                          # FIM item

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
        # Save all accuracies - merge per-FIM-item results into the overall results
        df["ToFIM"] = item  # Add the FIM-item column
        df.reindex(columns=columns)  # Adjust column order
        df_all = pd.concat([df_all, df], axis=0)  # Concatenate into the overall-results DataFrame

    print("============================================================")
    print("FIMitem:", item, " Finished.")  # FIM-item processing completion message
    print("============================================================")

    # Final save for the overall results
    # Save the CSV file containing results merged across all FIM items
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

    Entry point for running the fusion experiment.
    """
    parser = argparse.ArgumentParser(description="Fusion experiment control script")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/configSomeTask.yaml",
        help="Path to the config file (default: config/configSomeTask.yaml)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Root directory under which the experiment results are saved (default: results)",
    )
    args = parser.parse_args()
    main(configPath=args.config, outputDir=args.output)  # Run the main experiment process
