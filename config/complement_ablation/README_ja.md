# complement_ablation — 欠損動作の補完アブレーション

このフォルダの config は、Late Fusion 時に **動作が欠損した被験者をどう扱うか**
を比較する。基準 config は `config/action_selection/stand_actions.yaml` で、各
config は補完関連キー以外はすべて同一。いずれも `experiments/exam_fusion.py` で実行
する。

## 比較条件

| Config | 基準からの変更キー | 挙動 |
|---|---|---|
| `complete_case.yaml` | `preprocess.is_fill_actionless: false` | いずれかの動作が欠損した被験者を除外（complete-case、接尾辞 `_CompleteCase`） |
| `zero_fill.yaml` | なし（ベースライン） | 欠損動作をゼロ埋めのスライスのまま扱う（接尾辞 `_ZeroPad`） |
| `zero_fill_modality_dropout.yaml` | `zero_fill.yaml` + `trainings.modality_dropout_p: 0.15` | `zero_fill.yaml` と同様だが、学習時に実データセットの平均欠損率と同じ 0.15 の確率で存在アクションを追加でランダムに欠損扱いにする（modality dropout、接尾辞 `_ZeroPad_MD`） |
| `pad_mask.yaml` | `fusion.missing_representation: pad_mask` | 各ブランチを存在サンプルのみで順伝搬（欠損スロットはエンコーダに到達せず、BatchNorm を汚染しない）し、マスク平均で集約（接尾辞 `_Masking`） |
| `pad_mask_modality_dropout.yaml` | `pad_mask.yaml` + `trainings.modality_dropout_p: 0.15` | `pad_mask.yaml` と同様だが、学習時に同じ 0.15 の modality dropout を適用（接尾辞 `_Masking_MD`） |

`pad_mask` は存在アクションをマスク平均で集約する。
`trainings.modality_dropout_p`（接尾辞 `_MD` の命名は
`src/utils/decides.py` を参照）は、学習時のみ存在アクションを追加で疑似欠損
としてマークする。このフォルダの他の config ではすべて 0.0（無効）になっている。

## 実行方法

```bash
source path.sh
python experiments/exam_fusion.py -c config/complement_ablation/complete_case.yaml
python experiments/exam_fusion.py -c config/complement_ablation/zero_fill.yaml
python experiments/exam_fusion.py -c config/complement_ablation/zero_fill_modality_dropout.yaml
python experiments/exam_fusion.py -c config/complement_ablation/pad_mask.yaml
python experiments/exam_fusion.py -c config/complement_ablation/pad_mask_modality_dropout.yaml
```
