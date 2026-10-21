# fusion_strategy_ablations — 統合戦略のアブレーション

このフォルダの config は、動作ごとの特徴量を 1 つの FIM 項目予測へ **どのように
統合するか** を比較する。モデルはいずれも `STGCN` で、統合戦略全体を表す
`fusion.type` のみが異なる:

- `early_chn_stack` — 入力レベル、チャネルスタック
- `early_exp_graph` — 入力レベル、グラフ拡張
- `middle_shallow` — 中間特徴量、shallow 挿入位置
- `middle_deep` — 中間特徴量、deep 挿入位置
- `late_feature` — FC 前の特徴量
- `late_probability` — 動作ごとの確率出力

すべての fusion モデルは単一段階の end-to-end 学習を行う(動作ごとの事前学習
ステージはない)。`single_action.yaml`(単一動作ベースライン)以外は
`experiments/exam_fusion.py` で実行する。`single_action.yaml` のみ
`experiments/exam_each_seed.py` で実行する。

## 比較条件

| Config | 条件 | スクリプト |
|---|---|---|
| `single_action.yaml` | 単一動作ベースライン(統合なし) | `exam_each_seed.py` |
| `early_data.yaml` | Early Fusion、チャネルスタック(`chn_stack`) | `exam_fusion.py` |
| `early_graph.yaml` | Early Fusion、グラフ拡張(`exp_graph`) | `exam_fusion.py` |
| `middle_shallow.yaml` | Middle Fusion、shallow 層 | `exam_fusion.py` |
| `middle_deep.yaml` | Middle Fusion、deep 層 | `exam_fusion.py` |
| `late_feature.yaml` | Late Fusion(特徴量レベル) | `exam_fusion.py` |
| `late_probability.yaml` | Late Fusion(確率レベル) | `exam_fusion.py` |

## 実行方法

```bash
source path.sh

# 単一動作ベースライン
python experiments/exam_each_seed.py -c config/fusion_strategy_ablations/single_action.yaml

# Early Fusion
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/early_data.yaml
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/early_graph.yaml

# Middle Fusion
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/middle_shallow.yaml
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/middle_deep.yaml

# Late Fusion
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/late_feature.yaml
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/late_probability.yaml
```
