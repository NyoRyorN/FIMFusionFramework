"""
result managementclass

evaluation metrics integrate.
multiple evaluation metrics, statistical processing.

Key features:
- each
- per-class accuracy individual
- balanced accuracy calculate
- /true labels save
- mean value/ calculate
"""

import numpy as np

# detailed metrics recorded alongside accuracy in the result CSVs;
# order defines the CSV column order
OTHER_METRIC_KEYS = ["F1Score", "MCC", "Precision", "Recall", "AUCROC"]


class Results:
    """
    /class

    eachevaluation metrics,
    statistical processing and analyze.

    Args:
        classes (list[str]): class label list

    :
        accuracies (list): overall accuracy list
        balanced_accuracies (list): balanced accuracy list
        each_class_accuracies (dict): per-class accuracy dictionary
        all_pred_labels (list): allpredicted labels
        all_true_labels (list): alltrue labels
    """

    def __init__(self, classes: list[str]):
        """result managementobject initialize

        eachevaluation metrics list initialize,
        per-class accuracy row.

        Args:
            classes (list[str]): targetclass list
        """
        # overall accuracy
        self.accuracies = []
        # balanced accuracy (class imbalance)
        self.balanced_accuracies = []
        # per-class accuracy dictionary
        self.each_class_accuracies = dict()
        for c in classes:
            self.each_class_accuracies[f"Class{c}"] = []
        # detailed metrics dictionary (F1Score / MCC / Precision / Recall / AUCROC)
        self.other_metrics = dict()
        for k in OTHER_METRIC_KEYS:
            self.other_metrics[k] = []
        # all/true labels
        self.all_pred_labels = []
        self.all_true_labels = []

    def append(self, accuracy, balanced_accuracy, each_class_accuracy, pred_labels, true_labels, other_metrics=None):
        """1 append

        or epoch
        eachevaluation metricslist append.

        Args:
            accuracy (float): overall accuracy
            balanced_accuracy (float): balanced accuracy
            each_class_accuracy (dict): per-class accuracydictionary
            pred_labels (list): predicted labelslist
            true_labels (list): true labelslist
            other_metrics (dict): detailed metrics dictionary
                (F1Score / MCC / Precision / Recall / AUCROC); optional
        """
        # eachevaluation metrics append
        self.accuracies.append(accuracy)
        self.balanced_accuracies.append(balanced_accuracy)
        # per-class accuracy individualappend
        for c in each_class_accuracy:
            self.each_class_accuracies[c].append(
                each_class_accuracy[c])
        # detailed metrics individualappend
        if other_metrics is not None:
            for k in other_metrics:
                self.other_metrics[k].append(other_metrics[k])
        # Translated note
        self.all_pred_labels.extend(pred_labels)
        self.all_true_labels.extend(true_labels)

    def appendMean(self):
        """mean value calculate/append

        evaluation metrics mean value calculate,
        eachlist end append.

        Processing details:
        - overall accuracy calculate
        - balanced accuracy calculate
        - per-class accuracy calculate
        - mean value to append
        """
        # overall accuracy calculate/append
        self.mean_accuracies = np.mean(self.accuracies)
        self.accuracies.append(self.mean_accuracies)

        # balanced accuracy calculate/append
        self.mean_balanced_accuracies = np.mean(self.balanced_accuracies)
        self.balanced_accuracies.append(self.mean_balanced_accuracies)

        # per-class accuracy calculate/append
        self.mean_each_class_accuracies = dict()
        for c in self.each_class_accuracies:
            self.mean_each_class_accuracies[c] = np.mean(
                self.each_class_accuracies[c])
            self.each_class_accuracies[c].append(
                self.mean_each_class_accuracies[c])

        # detailed metrics calculate/append
        self.mean_other_metrics = dict()
        for k in self.other_metrics:
            self.mean_other_metrics[k] = np.mean(self.other_metrics[k])
            self.other_metrics[k].append(self.mean_other_metrics[k])

    def returnMeanList(self):
        """mean valuelist get

        calculatemean value return.
        analyze/generate.

        Returns:
            list: mean value list
                [overall accuracy, balanced accuracy, class1, class2,...,
                 F1Score, MCC, Precision, Recall, AUCROC]

        List structure:
        - mean_accuracy: overall accuracy mean value
        - mean_balanced_accuracy: balanced accuracy mean value
        - mean_classX_accuracy: eachclass mean value
        """
        # basic accuracy metrics beginning
        return_list = [self.mean_accuracies, self.mean_balanced_accuracies]
        # per-class accuracy append
        for c in self.each_class_accuracies:
            return_list.append(self.mean_each_class_accuracies[c])
        # detailed metrics append
        for k in self.other_metrics:
            return_list.append(self.mean_other_metrics[k])
        return return_list

    def returnLists(self):
        """allevaluation metricslist get

        all evaluation metrics return.
        analyze/visualize.

        Returns:
            list: eachevaluation metrics list
                [list, balanced accuracylist, class1list, class2list,...,
                 F1Scorelist, MCClist, Precisionlist, Recalllist, AUCROClist]

        List structure:
        - accuracy: overall accuracy list
        - balanced_accuracy: balanced accuracy list
        - classX_accuracy: eachclass list
        """
        # basic accuracy metrics beginning
        return_list = [self.accuracies, self.balanced_accuracies]
        # per-class accuracy append
        for c in self.each_class_accuracies:
            return_list.append(self.each_class_accuracies[c])
        # detailed metrics append
        for k in self.other_metrics:
            return_list.append(self.other_metrics[k])
        return return_list
