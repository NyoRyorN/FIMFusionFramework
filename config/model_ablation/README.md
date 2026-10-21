# model_ablation — Model / module ablation and baselines

Configs in this folder compare **which encoder model** is used, holding the rest
of the pipeline fixed. Every key except the `model` section is identical across
the configs, so they run as one ablation batch. All configs are run with
`experiments/exam_fusion.py`.

The folder holds both the proposed-model module variants (selected by
`model.name` and the attention `mode`) and two baseline models.

## Proposed-model module variants

| Config | `model.name` | Attention mode |
|---|---|---|
| `stgclsatt.yaml` | STGCLSAtt (proposed) | ST-sequential |
| `stam.yaml` | STGCNAtt | ST-sequential |
| `stgcn.yaml` | STGCN | ST-sequential |
| `stgcn_agcbilstm.yaml` | STGCAGLS | ST-sequential |
| `stgcn_bilstm.yaml` | STGCLSTM | ST-joint |
| `stgcn_bilstm_stjoint.yaml` | STGCLSAtt | ST-joint |
| `stgcn_stjoint.yaml` | STGCNAtt | ST-joint |

## Baselines

Ported to PyTorch under `src/model/baselines/`. All use
`fusion.type: late_probability` (early and middle fusion are rejected for
baselines); GradCAM is not supported for baselines.

| Config | `model.name` | Source |
|---|---|---|
| `litemv.yaml` | LITEMV | DeepRehabPile (Ismail-Fawaz et al., arXiv:2409.02869) |
| `convtran.yaml` | ConvTran | DeepRehabPile (Foumani et al., DMKD 2024) |

## Run

```bash
source path.sh

# Proposed-model module variants
python experiments/exam_fusion.py -c config/model_ablation/stgclsatt.yaml
python experiments/exam_fusion.py -c config/model_ablation/stam.yaml
python experiments/exam_fusion.py -c config/model_ablation/stgcn.yaml
python experiments/exam_fusion.py -c config/model_ablation/stgcn_agcbilstm.yaml
python experiments/exam_fusion.py -c config/model_ablation/stgcn_bilstm.yaml
python experiments/exam_fusion.py -c config/model_ablation/stgcn_bilstm_stjoint.yaml
python experiments/exam_fusion.py -c config/model_ablation/stgcn_stjoint.yaml

# Baselines
python experiments/exam_fusion.py -c config/model_ablation/litemv.yaml
python experiments/exam_fusion.py -c config/model_ablation/convtran.yaml
```
