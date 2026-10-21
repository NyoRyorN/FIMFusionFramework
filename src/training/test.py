from src.model.attention import GradCAM_SkeletonPreFusion
from src.utils import setPltFont
import sklearn.metrics as metrics
import torch
import os
import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import h5py


class Test():
    """ /evaluate runclass

    trained model test data for evaluate run,
    , confusion matrix, attention weights, GradCAM output.
    """

    def __init__(self,
                 model,
                 config,
                 paths,
                 model_pth,
                 wb_run=None,
                 seed=42,
                 ):
        """initializemethod

        Args:
            model: evaluatetarget
            config: config dictionary
            paths: result savingpath
            model_pth: trained modelfile path
            wb_run: Weights & Biasesrunobject
            seed:
        """
        # set load
        self.device = config["device"]
        self.model = model
        self.paths = paths
        self.classes = config["classes"]
        self.classes_num = len(config["classes"])
        self.model_name = config["model"]["name"]
        self.model_pth = model_pth
        self.fusion_mode = config["fusion"]["type"] if "fusion" in config else None
        self.wb_run = wb_run
        self.combine_action_idx = None if "combined_action_indices" not in config else config[
            "combined_action_indices"]
        self.is_output_weights = config["output_weights"] if "output_weights" in config else False
        self.seed = seed
        self.use_gradcam = config["model"]["cam_type"].lower(
        ) == "gradcam" if "cam_type" in config["model"] else False
        # set
        setPltFont()

    def output_weights(self, dir_list, weights, weight_type: str = "Att"):
        """(, GradCAM) HDF5file save

        Args:
            dir_list: data directory list
            weights: save
            weight_type: type("Att" or "CAM")
        """
        # Translated note
        weight_type_list = ["Att", "CAM"]
        if weight_type not in weight_type_list:
            raise ValueError(f"weight_type must be one of {weight_type_list}")
        if weights is None:
            return

        # HDF5file save
        with h5py.File(os.path.join(
                self.paths, f"weights_{weight_type}.h5"), 'w') as hf:
            for i in range(len(dir_list)):
                if self.fusion_mode is None:
                    # single model case
                    if isinstance(weights, dict):
                        g = hf.create_group(dir_list[i])
                        for key in weights.keys():
                            hf.create_dataset(
                                dir_list[i], data=weights[key][i], compression="gzip")
                    else:
                        hf.create_dataset(
                            dir_list[i], data=weights[i], compression="gzip")
                else:
                    # fusion model case
                    g = hf.create_group(dir_list[i])
                    for key in weights.keys():
                        idx = self.combine_action_idx[int(key)]
                        g.create_dataset(
                            str(idx), data=weights[key][i], compression="gzip")
            hf.close()

    def test(self, test_loader):
        """neural network run

        trained model test data evaluate,, confusion matrix,
        attention weights, GradCAM output/save.

        Args:
            test_loader: test DataLoader

        Returns:
            tuple: (accuracy, balanced_accuracy, each_class_accuracy,
                other_metrics, pred_label, true_label)
                where other_metrics is a dict with keys
                F1Score / MCC / Precision / Recall / AUCROC.
        """
        def reshape_transform(feats):  # for GradCAM
            """GradCAM featuresfunction"""
            assert feats.ndim == 4
            return feats

        # trained model load and set
        checkpoint = torch.load(self.model_pth, weights_only=True)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        # result savingvariable initialize
        pred_label = []
        true_label = []
        real_scores = []
        dir_list = []
        graph_weights = None

        correct = 0
        confusion_matrix = np.zeros((self.classes_num, self.classes_num))
        weights_att = None
        weights_cam = None
        all_outputs = None

        # GradCAMset("CAM" case)
        target_layers = []
        cam = None
        if self.use_gradcam:
            # fusion modelsingle model target layer set
            if hasattr(self.model, "model_list"):
                target_layers = [
                    self.model.model_list[i].st_gcn.st_gcn_networks[-1].tcn for i in range(len(self.model.model_list))]
            else:
                target_layers = [self.model.st_gcn.st_gcn_networks[-1].tcn]

        # GradCAM gradientscalculate enabled, otherwisedisabled
        with torch.set_grad_enabled(self.use_gradcam):
            # test data evaluate
            for batch_idx, (data, label, dir, score, mask) in enumerate(test_loader):
                data, label = data.float().to(self.device), label.detach().cpu().numpy()
                score = score.detach().cpu().numpy()
                dir = dir

                # run (fusion models receive the real availability mask;
                # no modality dropout at inference time)
                if getattr(self.model, "accepts_mask", False):
                    outputs = self.model(
                        data, mask=mask.float().to(self.device))
                else:
                    outputs = self.model(data)
                outputs = outputs.detach().cpu().numpy()

                # attention weights get("Att" case)
                if ("Att" in self.model_name or "AGLS" in self.model_name) \
                        and hasattr(self.model, "get_weights") and self.is_output_weights:
                    got_w = self.model.get_weights()
                    if got_w is None:
                        weights_att = None
                    elif weights_att is None:
                        # get format and save
                        if isinstance(got_w, tuple):
                            weights_att = dict({
                                "Spatial": got_w[0],
                                "Temporal": got_w[1],
                            })
                        elif isinstance(got_w, dict) and self.fusion_mode is not None:
                            weights_att = dict()
                            for key in got_w.keys():
                                weights_att[key] = got_w[key]
                        else:
                            weights_att = got_w

                    else:
                        # 2and later appendsave
                        if isinstance(got_w, tuple) and isinstance(weights_att, dict):
                            weights_att["Spatial"] = np.append(
                                weights_att["Spatial"], got_w[0], axis=0)
                            weights_att["Temporal"] = np.append(
                                weights_att["Temporal"], got_w[1], axis=0)
                        elif isinstance(got_w, dict) and self.fusion_mode is not None:
                            for key in got_w.keys():
                                if key not in weights_att.keys():
                                    weights_att[key] = got_w[key]
                                else:
                                    weights_att[key] = np.append(
                                        weights_att[key], got_w[key], axis=0)
                        else:
                            weights_att = np.append(
                                weights_att, self.model.get_weights(), axis=0)

                # predicted labels and true labels save
                true_label.extend(label)
                predict = np.argmax(outputs.data, 1)
                pred_label.extend(predict)
                correct += np.sum(predict == label)

                # GradCAM calculate
                if self.use_gradcam:
                    was = torch.backends.cudnn.enabled
                    torch.backends.cudnn.enabled = False

                    # GradCAMinstance create/deleteGPU
                    cam = GradCAM_SkeletonPreFusion(
                        model=self.model, target_layers=target_layers, reshape_transform=reshape_transform)
                    try:
                        cams = cam(data, targets=None,
                                   aug_smooth=False, eigen_smooth=False)  # shape(N, A, T, V)
                        if weights_cam is None:
                            # fusion mode CAM saveformat resolve
                            if self.fusion_mode is not None:
                                weights_cam = dict()
                                for key in range(cams.shape[1]):
                                    weights_cam[key] = cams[:, key]
                            else:
                                weights_cam = cams
                        else:
                            if self.fusion_mode is not None and isinstance(weights_cam, dict):
                                for key in range(cams.shape[1]):
                                    weights_cam[key] = np.append(
                                        weights_cam[key], cams[:, key], axis=0)
                            else:
                                weights_cam = np.append(
                                    weights_cam, cams, axis=0)
                    finally:
                        # GradCAM GPU
                        cam.activations_and_grads.release()
                        del cam

                    torch.backends.cudnn.enabled = was
                    data = data.detach().cpu().numpy()

                # output probabilities save
                all_outputs = outputs if all_outputs is None else np.append(
                    all_outputs, outputs, axis=0)

                # confusion matrix
                for t, p in zip(label, predict):
                    confusion_matrix[t, p] += 1

                # directory and save
                if dir is not None:
                    dir_list.extend(dir)
                if score is not None:
                    real_scores.extend(score)

            self.model.zero_grad()
            self.model = self.model.cpu()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # confusion matrix percentage format convert(eachclass and prediction results ratio)
        len_cm = len(confusion_matrix)
        for i in range(len_cm):
            sum_cm = np.sum(confusion_matrix[i])
            for j in range(len_cm):
                if sum_cm == 0:
                    confusion_matrix[i][j] = 0
                else:
                    confusion_matrix[i][j] = 100 * \
                        (confusion_matrix[i][j] / sum_cm)

        # calculate and result output
        accu = 100. * metrics.accuracy_score(true_label, pred_label)
        print('# Test Accuracy: {:.3f}[%]'.format(accu))
        accu_b = 100. * metrics.balanced_accuracy_score(true_label, pred_label)
        print('# Balanced Accuracy: {:.3f}[%]'.format(accu_b))
        for i in range(self.classes_num):
            a = np.where(np.array(true_label) == i)
            if len(a[0]) == 0:
                continue
            b_true = []
            b_pred = []
            for i in range(len(a[0])):
                b_true.append(true_label[a[0][i]])
                b_pred.append(pred_label[a[0][i]])

        # per-class accuracy and detailed metrics are always computed (independent
        # of wandb) so they can be recorded in the result CSVs
        accu_each = dict()
        for i in range(self.classes_num):
            # diagonal of the row-normalized confusion matrix = per-class recall [%]
            accu_each[f"Class{self.classes[i]}"] = confusion_matrix[i][i]
        # binary metrics treat class 0 as positive; multi-class uses weighted/ovr
        other_metrics = {
            "F1Score": metrics.f1_score(true_label, pred_label, average='weighted')
            if self.classes_num > 2 else metrics.f1_score(true_label, pred_label, average='binary', pos_label=0),
            "MCC": metrics.matthews_corrcoef(true_label, pred_label),
            "Precision": metrics.precision_score(true_label, pred_label, average='weighted')
            if self.classes_num > 2 else metrics.precision_score(true_label, pred_label, average='binary', pos_label=0),
            "Recall": metrics.recall_score(true_label, pred_label, average='weighted')
            if self.classes_num > 2 else metrics.recall_score(true_label, pred_label, average='binary', pos_label=0),
            "AUCROC": metrics.roc_auc_score(true_label, all_outputs, multi_class='ovr')
            if self.classes_num > 2 else metrics.roc_auc_score(1 - np.array(true_label), all_outputs[:, 0]),
        }
        if self.wb_run is not None:
            self.wb_run.log(
                {"Test Accuracy": accu,
                 "Balanced Accuracy": accu_b,
                 "F1 Score": other_metrics["F1Score"],
                 "MCC": other_metrics["MCC"],
                 "Precision": other_metrics["Precision"],
                 "Recall": other_metrics["Recall"],
                 "AUCROC": other_metrics["AUCROC"],
                 })
            for i in range(self.classes_num):
                self.wb_run.log(
                    {f"Accuracy of Class {self.classes[i]}": confusion_matrix[i][i]})

        # Plot Confusion Matrix
        conf_matrix = np.zeros((self.classes_num, self.classes_num))
        for i in range(self.classes_num):
            for j in range(self.classes_num):
                conf_matrix[i][j] = confusion_matrix[self.classes_num-1-i][j]
        # confusion matrix visualize and save
        sns.heatmap(conf_matrix, cmap='Blues', annot=True,
                    fmt=".2f", cbar=True, vmin=0, vmax=100)
        plt.title("accuracy: {:.3f}[%]".format(
            accu)+"\n"+"Balanced accuracy: {:.3f}[%]".format(accu_b))
        tick_marks = np.array(
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5][:self.classes_num])
        plt.xticks(tick_marks, self.classes, fontsize=15)
        plt.yticks(tick_marks, sorted(
            self.classes, reverse=True), fontsize=15)  # y-axis reverse order display
        plt.ylabel('True FIM Score', labelpad=10, fontsize=20)
        plt.xlabel('Predicted FIM Score', labelpad=10, fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.paths, 'confusion_matrix.png'))
        plt.show()
        # Weights & Biases confusion matrix record
        if self.wb_run is not None:
            self.wb_run.log({"confusion_matrix": wandb.Image(
                os.path.join(self.paths, 'confusion_matrix.png'))})

        # prediction results and CSV file save
        df = pd.DataFrame({
            "Directory": dir_list,
            "Predicted": list(map(int, pred_label)),
            "True": list(map(int, true_label)),
            "RealScore": real_scores
        })
        # class and confidence(probability)column append
        prob_cols = [f"Prob_{i}" for i in range(self.classes_num)]
        df_probs = pd.DataFrame(all_outputs, columns=prob_cols)

        # data frame
        df = pd.concat([df, df_probs], axis=1)

        # CSV file save
        df.to_csv(os.path.join(self.paths, "PredictTruelabel.csv"), index=False)
        plt.close()
        # Weights & Biases record
        if self.wb_run is not None:
            self.wb_run.log({"test_results": wandb.Table(dataframe=df)})

        # attention weights output
        if ("Att" in self.model_name or "AGLS" in self.model_name) and weights_att is not None and self.is_output_weights:
            self.output_weights(dir_list, weights_att, weight_type="Att")

        # GradCAM output
        if self.use_gradcam and weights_cam is not None and self.is_output_weights:
            self.output_weights(dir_list, weights_cam, weight_type="CAM")

        # return results
        return accu, accu_b, accu_each, other_metrics, pred_label, true_label
