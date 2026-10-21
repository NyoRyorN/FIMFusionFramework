# fusion_strategy_ablations — Fusion-strategy ablation

Configs in this folder compare **how per-action features are fused** into a
single FIM-item prediction. The model is `STGCN` in all of them; the configs
differ only in `fusion.type`, which names the whole strategy:

- `early_chn_stack` — input level, channel stack
- `early_exp_graph` — input level, graph expansion
- `middle_shallow` — intermediate features, shallow insertion point
- `middle_deep` — intermediate features, deep insertion point
- `late_feature` — pre-FC features
- `late_probability` — per-action output probabilities

All fusion models train end-to-end in a single pass (no per-action
pretraining stage). Most configs run with `experiments/exam_fusion.py`. The
exception is `single_action.yaml`, the single-action baseline, which runs with
`experiments/exam_each_seed.py`.

## Comparison conditions

| Config | Condition | Script |
|---|---|---|
| `single_action.yaml` | Single-action baseline (no fusion) | `exam_each_seed.py` |
| `early_data.yaml` | Early fusion, channel stack (`chn_stack`) | `exam_fusion.py` |
| `early_graph.yaml` | Early fusion, graph expansion (`exp_graph`) | `exam_fusion.py` |
| `middle_shallow.yaml` | Middle fusion, shallow layer | `exam_fusion.py` |
| `middle_deep.yaml` | Middle fusion, deep layer | `exam_fusion.py` |
| `late_feature.yaml` | Late feature-level fusion | `exam_fusion.py` |
| `late_probability.yaml` | Late probability-level fusion | `exam_fusion.py` |

## Run

```bash
source path.sh

# Single-action baseline
python experiments/exam_each_seed.py -c config/fusion_strategy_ablations/single_action.yaml

# Early fusion
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/early_data.yaml
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/early_graph.yaml

# Middle fusion
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/middle_shallow.yaml
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/middle_deep.yaml

# Late fusion
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/late_feature.yaml
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/late_probability.yaml
```
