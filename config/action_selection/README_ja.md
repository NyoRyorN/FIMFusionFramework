# action_selection — 動作サブセットのアブレーション

このフォルダの config は、FIM 項目分類において **どの動作サブセットを統合する
か** を比較する。各 config は `combined_action_indices`（10 動作から選ぶ 0 始まり
のサブセット）だけが異なり、それ以外の設定は同一。いずれも
`experiments/exam_fusion.py` で実行する。

動作インデックス → 名称（`src/utils/names.py` より）:

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 名称 | Sit | SitULE | SitFR | SitDRR | SitDLR | Std | StdFR | StdDRR | StdDLR | Squat |

`Sit` = 静止座位、`SitULE` = 座位上肢挙上、`FR` = 前方リーチ、`DRR`/`DLR` = 側方
リーチ（右/左）、`Std` = 静止立位。

## 比較条件

| Config | `combined_action_indices` | 条件 |
|---|---|---|
| `all.yaml` | `[0,1,2,3,4,5,6,7,8,9]` | 全 10 動作 |
| `sit_actions.yaml` | `[0,1,2,3,4]` | 座位動作のみ |
| `stand_actions.yaml` | `[1,5,6,7,8,9]` | 立位セット（+ SitULE）。`complement_ablation/` の基準 config |
| `moving_actions.yaml` | `[1,2,3,4,6,7,8,9]` | 動的動作のみ（静止 Sit=0・Std=5 を除外） |
| `std_actions_without_std.yaml` | `[1,6,7,8,9]` | 立位セットから静止 Std (5) を除外 |
| `std_actions_without_stdfr.yaml` | `[1,5,7,8,9]` | 立位セットから StdFR (6) を除外 |
| `std_actions_without_stddr.yaml` | `[1,5,6,9]` | 立位セットから StdDRR + StdDLR (7, 8) を除外 |
| `std_actions_without_squat.yaml` | `[1,5,6,7,8]` | 立位セットから Squat (9) を除外 |

## 実行方法

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
