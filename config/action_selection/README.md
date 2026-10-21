# action_selection — Action-subset ablation

Configs in this folder compare **which subset of actions is fused** for FIM-item
classification. Every config is identical except for `combined_action_indices`
(a 0-based subset of the 10 actions). All configs are run with
`experiments/exam_fusion.py`.

Action index → name (from `src/utils/names.py`):

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Name | Sit | SitULE | SitFR | SitDRR | SitDLR | Std | StdFR | StdDRR | StdDLR | Squat |

`Sit` = static sitting, `SitULE` = sitting upper-limb elevation, `FR` = forward
reach, `DRR`/`DLR` = downward right/left reach, `Std` = static standing.

## Comparison conditions

| Config | `combined_action_indices` | Condition |
|---|---|---|
| `all.yaml` | `[0,1,2,3,4,5,6,7,8,9]` | All 10 actions |
| `sit_actions.yaml` | `[0,1,2,3,4]` | Sitting actions only |
| `stand_actions.yaml` | `[1,5,6,7,8,9]` | Standing set (+ SitULE); base for `complement_ablation/` |
| `moving_actions.yaml` | `[1,2,3,4,6,7,8,9]` | Dynamic actions only (drops static Sit=0, Std=5) |
| `std_actions_without_std.yaml` | `[1,6,7,8,9]` | Standing set minus static Std (5) |
| `std_actions_without_stdfr.yaml` | `[1,5,7,8,9]` | Standing set minus StdFR (6) |
| `std_actions_without_stddr.yaml` | `[1,5,6,9]` | Standing set minus StdDRR + StdDLR (7, 8) |
| `std_actions_without_squat.yaml` | `[1,5,6,7,8]` | Standing set minus Squat (9) |

## Run

```bash
source path.sh
python experiments/exam_fusion.py -c config/action_selection/all.yaml
python experiments/exam_fusion.py -c config/action_selection/sit_actions.yaml
python experiments/exam_fusion.py -c config/action_selection/stand_actions.yaml
python experiments/exam_fusion.py -c config/action_selection/moving_actions.yaml
python experiments/exam_fusion.py -c config/action_selection/std_actions_without_std.yaml
python experiments/exam_fusion.py -c config/action_selection/std_actions_without_stdfr.yaml
python experiments/exam_fusion.py -c config/action_selection/std_actions_without_stddr.yaml
python experiments/exam_fusion.py -c config/action_selection/std_actions_without_squat.yaml
```
