from src.training.test import Test
from src.training.train import Train
from src.model.build_model import build_encoder, is_baseline
from src.model import LateFusion, MiddleFusion, EarlyFusion
from src.model.params.separate_channels import separate_channels
import torch
import os


# fusion.type names the whole strategy: "<family>_<variant>". Keeping the
# family and its variant in one value makes combinations that do not exist
# (e.g. late fusion with a middle-fusion insertion point) unrepresentable.
FUSION_TYPES = (
    "early_chn_stack",
    "early_exp_graph",
    "middle_shallow",
    "middle_deep",
    "late_feature",
    "late_probability",
)


def _build_model(config):
    """Instantiate the encoder or fusion model selected by config.

    Args:
        config: experiment config dictionary. Presence of the top-level
            "fusion" key selects a fusion model (dispatched by
            config["fusion"]["type"]); otherwise a single encoder is built.

    Returns:
        nn.Module ready for training.

    Raises:
        ValueError: for a fusion.type outside FUSION_TYPES, or when a
            baseline model is combined with middle/early fusion (baselines
            lack the ST-GCN channel list / early-fusion graph support).
    """
    if "fusion" not in config:
        return build_encoder(config)

    fusion_type = config["fusion"]["type"]
    if fusion_type not in FUSION_TYPES:
        raise ValueError("Invalid fusion type: {}. Must be one of {}".format(
            fusion_type, ", ".join(FUSION_TYPES)))
    # each model derives its own variant from fusion.type; only the family
    # is needed to pick the class
    family = fusion_type.split("_", 1)[0]

    if family == "late":
        # Late Fusion: each branch predicts independently, fused at the end
        return LateFusion(config=config)
    elif family == "middle":
        # Middle Fusion: fuse intermediate feature maps across actions
        if is_baseline(config):
            raise ValueError(
                "Middle fusion is not supported for baseline models "
                "(LITEMV / ConvTran); use late fusion")
        gcn_channels, fusion_channels = separate_channels(
            config["model"]["st-gcn"]["output_channel_list"],
            fusion_type.split("_", 1)[1]
        )
        return MiddleFusion(
            config=config, gcn_channels=gcn_channels, fusion_channels=fusion_channels)
    # early
    if is_baseline(config):
        raise ValueError("Baselines do not support early fusion")
    return EarlyFusion(config=config)


def _build_optimizer(model, train_attr):
    """Instantiate the optimizer selected by train_attr["optimizer"]."""
    if train_attr["optimizer"] == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=train_attr["lr"],
            weight_decay=train_attr["weight_decay"]
        )
    elif train_attr["optimizer"] == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=train_attr["lr"],
            momentum=train_attr["momentum"],
            weight_decay=train_attr["weight_decay"]
        )
    elif train_attr["optimizer"] == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=train_attr["lr"],
            weight_decay=train_attr["weight_decay"]
        )
    return None


def Run(config,
        trainloader,
        testloader,
        paths=None,
        wb_run=None,
        seed=42,
        ):
    """Train and evaluate rowfunction

    Builds the model selected by config (single encoder or fusion model),
    trains it end-to-end, and evaluates it on the test set. Shared by the
    single-model driver (exam_each_seed.py) and the fusion driver
    (exam_fusion.py).

    Args:
        config: config dictionary(model, training parameters)
        trainloader: training DataLoader
        testloader: test DataLoader
        paths: save path
        wb_run: Weights & Biasesrunobject
        seed: random seed(used by Test for logging)

    Returns:
        tuple: (accuracy, balanced_accuracy, each_class_accuracy, other_metrics, pred_labels, true_labels)
    """
    criterion = torch.nn.NLLLoss()
    model = _build_model(config)
    optimizer = _build_optimizer(model, config["trainings"])

    # model save path set
    model_pth = os.path.join(paths, "checkpoint.pth")

    # runobject initialize and run
    train = Train(
        config=config,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        wb_run=wb_run,
        paths=paths,
        model_pth=model_pth,
    )
    train.train(train_loader=trainloader)

    # runobject initialize and evaluaterun
    test = Test(
        config=config,
        model=model,
        paths=paths,
        model_pth=model_pth,
        wb_run=wb_run,
        seed=seed,
    )

    # evaluate run
    accuracy, balanced_accuracy, each_class_accuracy, other_metrics, pred_labels, true_labels = test.test(
        test_loader=testloader)

    print("Done.")

    # evaluatereturn results
    return accuracy, balanced_accuracy, each_class_accuracy, other_metrics, pred_labels, true_labels
