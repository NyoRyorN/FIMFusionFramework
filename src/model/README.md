# src/model — Network definitions

All neural-network definitions: the proposed **STGCLSATT** encoder, the three fusion models
that combine multiple actions, the encoder factory, and ported baselines. The encoder is a
GCN → RNN → attention stack; each stage lives in its own subfolder (`gcn/`, `rnn/`,
`attention/`) so it can be swapped for the model ablations in `config/model_ablation/`.

## Files

| File | Role | Source |
|---|---|---|
| `stgclsatt.py` | `STGCLSATT` — the proposed spatial-temporal graph + LSTM + attention encoder; also parameterizes the in-house module variants (STGCN / STGCNAtt / STGCLSTM / …). | Proposed method (this project, TNSRE 2026); builds on ST-GCN and AGC-LSTM (see `gcn/`, `rnn/`). |
| `early_fusion.py` | Early fusion: combines multiple actions at the input (channel-stack or graph-expansion) before a single encoder; includes adjacency-matrix visualization whose element spread is a population-style matrix diagnostic. | Proposed method (this project). |
| `middle_fusion.py` | Middle fusion: per-action encoders fused at an intermediate feature stage. | Proposed method (this project). |
| `late_fusion.py` | Late fusion: per-action encoders averaged at the feature/probability level. | Proposed method (this project). |
| `build_model.py` | Encoder factory: resolves `config["model"]["name"]` to an encoder class and flags baselines (`is_baseline`). | — |
| `mask_token.py` | Shared helpers for missing-action masks in fusion models: zero padding, pad_mask (present-only forward), and modality dropout. | Proposed method (this project). |

## Subfolders

| Subfolder | Role |
|---|---|
| [`gcn/`](gcn/) | Spatial-temporal graph convolution subnet (ST-GCN). |
| [`rnn/`](rnn/) | Recurrent subnets: LSTM and AGC-LSTM. |
| [`attention/`](attention/) | Attention subnet and CAM/GradCAM fusion visualization. |
| [`baselines/`](baselines/) | PyTorch ports of two external baseline models. |
| [`params/`](params/) | Model parameter helpers (channel separation for fusion). |

## Baselines and source papers

Baselines are selected by `model.name` and configured in `config/model_ablation/`.

| Model | Source |
|---|---|
| LITEMV | DeepRehabPile (Ismail-Fawaz et al., arXiv:2409.02869) |
| ConvTran | DeepRehabPile (Foumani et al., DMKD 2024) |
