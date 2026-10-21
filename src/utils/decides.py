import datetime
import random
import numpy as np
import torch


def decideSeed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)                # numpy generate initialize
    torch.manual_seed(seed)             # all generate initialize
    torch.cuda.manual_seed(seed)        # current GPU for generate initialize
    torch.cuda.manual_seed_all(seed)    # all GPU generate initialize
    # True and cuDNN, False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def decideExamName(situations, use_wb=False):
    dt_now = datetime.datetime.now()
    dateForName = dt_now.strftime('%Y-%m-%d_%H-%M')+"_start"

    exam_name = "Ex_" + situations["model"]["name"]
    layer = "_L" + \
        str(len(situations["model"]["st-gcn"]["output_channel_list"])
            ) if situations["model"]["name"].count("STGC") > 0 else ""

    exam_name += layer
    wb_name = exam_name

    if situations["seed_fix_mode"].lower() == "split":
        exam_name += "_SplitFix"
        wb_name += "_SplitFix"
    elif situations["seed_fix_mode"].lower() == "model":
        exam_name += "_ModelSeedFix"
        wb_name += "_ModelSeedFix"

    if "fusion" in situations:
        # fusion.type already names the full strategy (e.g. "late_probability")
        exam_name += "_" + situations["fusion"]["type"]
        wb_name += "_" + situations["fusion"]["type"]

        # Use condition labels so results can be compared without decoding
        # implementation-level configuration values.
        missing_repr = situations["fusion"].get("missing_representation", "zero")
        if not situations["preprocess"].get("is_fill_actionless", True):
            exam_name += "_CompleteCase"
            wb_name += "_CompleteCase"
        elif situations["preprocess"].get("complement_actionless", False):
            if situations["preprocess"].get("complement_map"):
                map_name = situations["preprocess"].get("complement_map_name")
                if not map_name:
                    raise ValueError(
                        "preprocess.complement_map_name is required when "
                        "preprocess.complement_map is set")
                exam_name += "_CmplAct-" + map_name
                wb_name += "_CmplAct-" + map_name
            else:
                exam_name += "_CmplAct"
                wb_name += "_CmplAct"
        else:
            representation_suffix = (
                "_Masking" if missing_repr == "pad_mask" else "_ZeroPad"
            )
            exam_name += representation_suffix
            wb_name += representation_suffix

            if situations["trainings"].get("modality_dropout_p", 0.0) > 0:
                exam_name += "_MD"
                wb_name += "_MD"

    if "Att" in situations["model"]["name"]:
        exam_name += "_"+situations["model"]["attention"]["mode"]
    exam_name += "_" + dateForName
    wb_name += "_" + dateForName

    if use_wb:
        return exam_name, wb_name
    else:
        return exam_name, None


def decideWandbConfig(situations, seed, item, task=None, fold_idx=0) -> dict:
    config = {}
    # ST-GCN / attention blocks are absent for baseline models
    stgcn_channels = situations["model"].get(
        "st-gcn", {}).get("output_channel_list")
    attention_mode = situations["model"].get("attention", {}).get("mode")

    if task is None and "fusion" in situations:
        fusion_attr = situations["fusion"]
        config = {
            "seed": seed,
            "fim_item": item,
            "batch_size": situations["trainings"]["batch_size"],
            "fusion_type": situations["fusion"]["type"],
            "epoch": situations["trainings"]["epoch"],
            "optimizer": situations["trainings"]["optimizer"],
            "scheduler": situations["trainings"]["scheduler"],
            "lr": situations["trainings"]["lr"],
            "center_norm": situations["preprocess"]["norm_centering_mode"],
            "center_point": situations["preprocess"]["norm_center_point"] if situations["preprocess"]["norm_centering_mode"] != "None" else None,
            "moving_frame": situations["preprocess"]["use_moving_frame"],
            "frame_length": situations["preprocess"]["frame_length"],
            "fill_short": situations["preprocess"]["is_fill_short"],
            "stgcn_layer": len(stgcn_channels) if stgcn_channels else None,
            "attention_type": attention_mode,
            "ST-GCN_channels": stgcn_channels,
            "combined_action_indices": situations["combined_action_indices"],
            "is_fill_actionless": situations["preprocess"]["is_fill_actionless"],
            "complement_actionless": situations["preprocess"]["complement_actionless"],
            "complement_map_name": situations["preprocess"].get("complement_map_name"),
            "is_drop_last": situations["trainings"]["is_drop_last"],
            "missing_representation": fusion_attr.get("missing_representation", "zero"),
            "modality_dropout_p": situations["trainings"].get("modality_dropout_p", 0.0),
        }
        config["fold_idx"] = fold_idx
        if "agc-lstm" in situations["model"] and "AGLS" in situations["model"]["name"]:
            config["agc_lstm"] = situations["model"]["agc-lstm"]
        elif "LS" in situations["model"]["name"] and "AG" not in situations["model"]["name"]:
            config["temporal_model"] = "BiLSTM" if situations["model"]["rnn"]["bidirectional"] else "LSTM"
        # elif "GRU" in situations["model"]["name"] and "AG" not in situations["model"]["name"]:
        #     config["temporal_model"] = "BiGRU" if situations["model"]["rnn"]["bidirectional"] else "GRU"
        # elif "TTF" in situations["model"]["name"]:
        #     config["temporal_model"] = "Transformer"
        if "Att" not in situations["model"]["name"]:
            del config["attention_type"]

    else:
        config = {
            "seed": seed,
            "fim_item": item,
            "action_task": task,
            "batch_size": situations["trainings"]["batch_size"],
            "epoch": situations["trainings"]["epoch"],
            "optim": situations["trainings"]["optimizer"],
            "scheduler": situations["trainings"]["scheduler"],
            "lr": situations["trainings"]["lr"],
            "center_norm": situations["preprocess"]["norm_centering_mode"],
            "center_point": situations["preprocess"]["norm_center_point"] if situations["preprocess"]["norm_centering_mode"] != "None" else None,
            "moving_frame": situations["preprocess"]["use_moving_frame"],
            "frame_length": situations["preprocess"]["frame_length"],
            "fill_short": situations["preprocess"]["is_fill_short"],
            "ST-GCN_channels": stgcn_channels,
        }
        config["fold_idx"] = fold_idx
        if "agc-lstm" in situations["model"] and "AGLS" in situations["model"]["name"]:
            config["agc_lstm"] = situations["model"]["agc-lstm"]
        elif "LS" in situations["model"]["name"] and "AG" not in situations["model"]["name"]:
            config["temporal_model"] = "BiLSTM" if situations["model"]["rnn"]["bidirectional"] else "LSTM"
        # elif "GRU" in situations["model"]["name"] and "AG" not in situations["model"]["name"]:
        #     config["temporal_model"] = "BiGRU" if situations["model"]["rnn"]["bidirectional"] else "GRU"
        # elif "TTF" in situations["model"]["name"]:
        #     config["temporal_model"] = "Transformer"
        if "Att" in situations["model"]["name"]:
            config["attention_type"] = attention_mode

    return config
