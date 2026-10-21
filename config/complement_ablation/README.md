# complement_ablation — Missing-action handling ablation

Configs in this folder compare **how subjects with missing actions are handled**
during late fusion. The base config is
`config/action_selection/stand_actions.yaml`; every key except the
complementation keys is identical across the configs. All configs are run
with `experiments/exam_fusion.py`.

## Comparison conditions

| Config | Key changed from base | Behavior |
|---|---|---|
| `complete_case.yaml` | `preprocess.is_fill_actionless: false` | Subjects missing any action are excluded (complete-case, `_CompleteCase` suffix) |
| `zero_fill.yaml` | none (baseline) | Missing actions stay all-zero slices (`_ZeroPad` suffix) |
| `zero_fill_modality_dropout.yaml` | `zero_fill.yaml` + `trainings.modality_dropout_p: 0.15` | Same as `zero_fill.yaml`, but additional present actions are randomly dropped during training (modality dropout) at the real dataset's average missing-action rate, 0.15 (`_ZeroPad_MD` suffix) |
| `pad_mask.yaml` | `fusion.missing_representation: pad_mask` | Each branch is forwarded only for present samples (missing slots never reach the encoder, no BatchNorm contamination), then aggregated with a masked mean (`_Masking` suffix) |
| `pad_mask_modality_dropout.yaml` | `pad_mask.yaml` + `trainings.modality_dropout_p: 0.15` | Same as `pad_mask.yaml`, with the same 0.15 modality dropout rate applied during training (`_Masking_MD` suffix) |

`pad_mask` aggregates the present actions with a masked mean.
`trainings.modality_dropout_p` (see `src/utils/decides.py` for the
`_MD` naming suffix) randomly marks additional present actions as
pseudo-missing during training only; it is 0.0 (disabled) in every other config
in this folder.

## Run

```bash
source path.sh
python experiments/exam_fusion.py -c config/complement_ablation/complete_case.yaml
python experiments/exam_fusion.py -c config/complement_ablation/zero_fill.yaml
python experiments/exam_fusion.py -c config/complement_ablation/zero_fill_modality_dropout.yaml
python experiments/exam_fusion.py -c config/complement_ablation/pad_mask.yaml
python experiments/exam_fusion.py -c config/complement_ablation/pad_mask_modality_dropout.yaml
```
