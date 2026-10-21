import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# from src.utils.set_plt_font import setPltFont
from sklearn.metrics import accuracy_score


def makeConfMatrixOnExam(true_labels, pred_labels, classes, accuracy, balanced_accuracy,
                         paths, model_name, item, task=None, return_CBA=False):
    # setPltFont()

    ext = ["pdf", "svg", "png"]
    filename = ""
    if task is not None:
        filename += task+"to"
    filename += item
    filename += "_"+model_name if model_name != "" else ""
    plt.rcParams["font.size"] = 15

    confusion_matrix = np.zeros((len(classes), len(classes)))
    for t, p in zip(true_labels, pred_labels):
        confusion_matrix[t, p] += 1

    # Plot Confusion Matrix on Count
    conf_matrix = np.zeros((len(classes), len(classes)))
    for i in range(len(classes)):
        for j in range(len(classes)):
            conf_matrix[i][j] = confusion_matrix[len(classes)-1-i][j]
    sns.heatmap(conf_matrix.astype(int), cmap='Blues',
                annot=True, cbar=True, fmt="d")
    plt.title("accuracy: {:.3f}[%]".format(
        accuracy)+"\n"+"Balanced accuracy: {:.3f}[%]".format(balanced_accuracy))
    tick_marks = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5][:len(classes)])
    plt.xticks(tick_marks, classes, fontsize=15)
    plt.yticks(tick_marks, sorted(classes, reverse=True), fontsize=15)  # reverse order
    plt.ylabel('True FIM Score', labelpad=10, fontsize=20)
    plt.xlabel('Predicted FIM Score', labelpad=10, fontsize=20)
    plt.tight_layout()
    for e in ext:
        plt.savefig(os.path.join(paths, filename+'_countConfMtx.'+e))
    plt.show()
    plt.close()

    df = pd.DataFrame(confusion_matrix, index=sorted(
        classes, reverse=True), columns=classes)
    df.to_csv(os.path.join(paths, filename+'_countConfMtx.csv'))

    # Plot Confusion Matrix on Rate
    len_cm = len(confusion_matrix)
    for i in range(len_cm):
        sum_cm = np.sum(confusion_matrix[i])
        for j in range(len_cm):
            if sum_cm == 0:
                confusion_matrix[i][j] = 0
            else:
                confusion_matrix[i][j] = 100 * \
                    (confusion_matrix[i][j] / sum_cm)

    # Plot Confusion Matrix
    conf_matrix = np.zeros((len(classes), len(classes)))
    for i in range(len(classes)):
        for j in range(len(classes)):
            conf_matrix[i][j] = confusion_matrix[len(classes)-1-i][j]
    sns.heatmap(conf_matrix, cmap='Blues', annot=True,
                fmt=".2f", cbar=True, vmin=0, vmax=100)
    plt.title("accuracy: {:.3f}[%]".format(
        accuracy)+"\n"+"Balanced accuracy: {:.3f}[%]".format(balanced_accuracy))

    tick_marks = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5][:len(classes)])
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, sorted(classes, reverse=True), fontsize=20)  # reverse order
    plt.ylabel('True FIM Score', labelpad=10, fontsize=20)
    plt.xlabel('Predicted FIM Score', labelpad=10, fontsize=20)
    plt.tight_layout()
    for e in ext:
        plt.savefig(os.path.join(paths, filename+'_rateConfMtx.'+e))
    plt.show()
    plt.close()

    df = pd.DataFrame(confusion_matrix, index=sorted(
        classes, reverse=True), columns=classes)
    df.to_csv(os.path.join(paths, filename+'_rateConfMtx.csv'))

    if return_CBA:
        CBA_list = []
        for i in range(len_cm):
            CBA_list.append(confusion_matrix[i][i])
        return CBA_list
