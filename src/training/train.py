import torch
import os
# import torch.nn.functional as F
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union
import wandb
from src.model.mask_token import apply_modality_dropout


class Train():
    """neural network rowclass

    Runs the epoch loop for a single model or a fusion model,
    with learning-rate scheduling.
    """

    def __init__(self,
                 config,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: Optional[torch.nn.Module] = None,
                 wb_run: Optional[wandb.run] = None,
                 paths: Optional[str] = None,
                 model_pth: Optional[str] = None,
                 ) -> None:
        """Train class initialize

        Args:
            config: training settings config dictionary
            model: target neural network
            optimizer: optimization algorithm(Adam, SGD)
            criterion: loss function(CrossEntropyLoss)
            wb_run: Weights & Biases runinstance()
            paths: result saving path
            model_pth: model weights destination path
        """
        # set initialize
        self.situation = config
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = config["device"]
        self.model = model
        self.model_name = config["model"]["name"]
        self.paths = paths
        self.model_pth = model_pth

        # training parameters set
        train_attr = config["trainings"]
        self.batch_size = train_attr["batch_size"]
        # modality dropout probability; 0.0 disables (training only)
        self.modality_dropout_p = train_attr.get("modality_dropout_p", 0.0)

        self.optimizer_attr = train_attr
        self.num_epoch = train_attr["epoch"]

        # model structure set
        self.in_channels = len(config["attributes"]) * 3  # number of input channels(number of attributes×3coordinate dimensions)
        # ST-GCN output channel list (absent for baseline models)
        self.channels = config["model"].get(
            "st-gcn", {}).get("output_channel_list")

        # AGC-LSTM attention loss
        self.agc_lstm_attn_gamma = config["model"].get(
            "agc-lstm", {}).get("loss", {}).get("gamma", 0.1)

        # Weights & Biases runinstance
        self.wb_run = wb_run

    def set_scheduler(self, train_loader):
        """learning ratescheduler set

        setaccording tolearning ratescheduler initialize.

        Args:
            train_loader: DataLoader(OneCycleLR steps_per_epochcalculate)
        """
        if self.optimizer_attr["scheduler"] == "ReduceLROnPlateau":
            # on plateau learning rate reducescheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.optimizer_attr["reduce_lr_on_plateau"]["mode"],
                factor=self.optimizer_attr["reduce_lr_on_plateau"]["factor"],
                patience=self.optimizer_attr["reduce_lr_on_plateau"]["patience"])
        elif self.optimizer_attr["scheduler"] == "MultiStepLR":
            # specifyepoch learning rate stepwise reducescheduler
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.optimizer_attr["multi_step_lr"]["milestones"],
                gamma=self.optimizer_attr["multi_step_lr"]["gamma"])
        elif self.optimizer_attr["scheduler"] == "OneCycleLR":
            # 1 learning rate scheduler
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.optimizer_attr["lr"],
                steps_per_epoch=len(train_loader),
                epochs=self.num_epoch,
                pct_start=self.optimizer_attr["one_cycle_lr"]["pct_start"])
        elif self.optimizer_attr["scheduler"] == "None":
            # scheduler
            self.scheduler = None

    def train(self, train_loader: torch.utils.data.DataLoader):
        """neural network run

        single model or fusion model row,
        loss calculate, backpropagation, parameter update run.

        Args:
            train_loader: DataLoader
        """
        # training recordslist initialize
        ELLA = []  # Epoch, Loss, Learning rate, Accuracy record
        loss_list_train = []
        lr_list = []

        # training mode set
        self.model.train()

        checkpoint = None
        self.model.to(self.device)

        # learning ratescheduler set
        self.set_scheduler(train_loader)

        print("Training...")

        # recorddata frame initialize
        label_df = pd.DataFrame(columns=["batch_idx", "label", "dir"])
        # 1Epoch number of samples initialize
        total_samples = 0

        # epoch loop
        for epoch in range(self.num_epoch):
            total_samples = 0
            # epoch initialize statistics
            train_correct = 0  # number of correct predictions
            train_sum_loss = {
                "total": 0.0,
                "crossent": 0.0,
                "agc_lstm_attn": 0.0,
            }

            # batch loop
            for batch_idx, (data, label, dir, score, mask) in enumerate(train_loader):
                # Translated note
                data = data.float().to(self.device)
                label = label.to(self.device)

                # fusion models receive an effective mask that merges the
                # real missing actions with training-time modality dropout
                if getattr(self.model, "accepts_mask", False):
                    mask = mask.float().to(self.device)
                    eff_mask = apply_modality_dropout(
                        mask, self.modality_dropout_p)
                    outputs = self.model(data, mask=eff_mask)
                else:
                    outputs = self.model(data)
                outputs = torch.log(torch.clamp_min(outputs, 1e-10))

                # classification loss calculate
                loss = self.criterion(outputs, label)
                train_sum_loss["crossent"] += loss.item() * len(data)

                # AGC-LSTM attention loss append
                if hasattr(self.model, "get_agc_lstm_attn_loss") and "AGLS" in self.model_name:
                    agc_lstm_loss = self.agc_lstm_attn_gamma * self.model.get_agc_lstm_attn_loss()
                    loss += agc_lstm_loss
                    train_sum_loss["agc_lstm_attn"] += agc_lstm_loss.item() * \
                        len(data)

                # backpropagation and parameter update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                label = label.detach().cpu().numpy()

                # update statistics
                train_sum_loss["total"] += loss.item() * len(data)
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.detach().cpu().numpy()
                train_correct += np.sum(predicted == label)
                total_samples += len(data)

                # first epoch record
                if epoch == 0:
                    # batch information record(analyze)
                    batch_idx_list = np.array(
                        [batch_idx for i in range(len(label))])
                    temp_label_df = pd.DataFrame({
                        "batch_idx": batch_idx_list,
                        "label": label,
                        "dir": dir})
                    label_df = pd.concat([label_df, temp_label_df], axis=0)

                # OneCycleLR scheduler (per batch)
                if self.optimizer_attr["scheduler"] == "OneCycleLR":
                    self.scheduler.step()

            # epoch statisticscalculate and display
            temp_accuracy = (100. * train_correct /
                             total_samples) if self.criterion is not None else 0
            train_mean_loss = train_sum_loss["total"] / \
                total_samples if total_samples > 0 else 0
            print('# Epoch: {:03} | Loss: {:.4f} | Accuracy: {:.4f} | LR: {}'.format(
                epoch, train_mean_loss, temp_accuracy, self.optimizer.param_groups[0]['lr']))
            # training history record
            ELLA.append([epoch, train_mean_loss,
                        self.optimizer.param_groups[0]['lr'], temp_accuracy])

            # Weights & Biases to result recording
            if self.wb_run is not None:
                idx = ""
                # per action case create
                if "checkpoint_" in self.model_pth:
                    idx = "/Action"
                    idx += self.model_pth.split(
                        "checkpoint_")[-1].split(".")[0]
                # for logging dictionarycreate
                log_dict = {
                    "TrainLoss/All"+idx: train_mean_loss,
                    "TrainAccuracy"+idx: (100. * train_correct / total_samples) if total_samples > 0 else 0,
                    "LearningRate"+idx: self.optimizer.param_groups[0]['lr']
                }
                # agc_lstm_attn_loss case record
                if train_sum_loss["agc_lstm_attn"] > 0.0:
                    log_dict["TrainLoss/AGC_LSTM_Attn"+idx] = train_sum_loss["agc_lstm_attn"] / total_samples \
                        if total_samples > 0 else 0
                    log_dict["TrainLoss/CrossEntropy"+idx] = train_sum_loss["crossent"] / \
                        total_samples if total_samples > 0 else 0
                # Weights & Biases log
                self.wb_run.log(log_dict)

            # training history record
            loss_list_train.append(train_mean_loss)
            lr_list.append(self.optimizer.param_groups[0]['lr'])

            # scheduler (per epoch)
            if self.optimizer_attr["scheduler"] == "ReduceLROnPlateau":
                self.scheduler.step(train_sum_loss["total"])
            elif self.optimizer_attr["scheduler"] == "MultiStepLR":
                self.scheduler.step()

            # epoch, checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }

        # model weights save
        torch.save(checkpoint["model_state_dict"], self.model_pth)

        # training history CSV file save
        columns = ["Epoch", "TrainLoss", "LearningRate", "TrainAccuracy"]
        df = pd.DataFrame(ELLA, columns=columns)
        if self.criterion is None:
            # loss function unspecified case accuracy column delete
            df = df.drop(columns=["TrainAccuracy"])

        # CSV filesave(epoch)
        df.to_csv(os.path.join(self.paths, "ELLA_ep" +
                  str(checkpoint["epoch"])+".csv"), index=False)
