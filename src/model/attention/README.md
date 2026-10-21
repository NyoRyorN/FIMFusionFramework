# src/model/attention — Attention subnet and CAM/GradCAM fusion

The attention stage of the proposed encoder, plus the class-activation-map visualization
used to interpret fusion predictions. Two attention modes are supported and selected by the
model variant: **ST-sequential** (spatial then temporal) and **ST-joint** (joint
spatial-temporal). The primitive attention blocks live in `components/`.

**Source:** attention subnet is part of the proposed method (this project); the CAM/GradCAM
visualization follows Grad-CAM (Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization," ICCV 2017).

## Files

| File | Role |
|---|---|
| `attention_part.py` | `Attention_part` / `Attention_part_fusion`: the attention head over the GCN+RNN features; supports weight freezing and attention-weight export for interpretability. |
| `cam_fusion.py` | Pre-fusion Grad-CAM / Class Activation Mapping over the spatial (T, V) dimensions for multi-action fusion; produces the weights visualized in `reports/skeleton_plots/`. |

## Subfolders

| Subfolder | Role |
|---|---|
| [`components/`](components/) | Attention building blocks: base attention and ST-joint attention. |
