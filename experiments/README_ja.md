# experiments — 実験エントリスクリプト

実験を実行する 2 つのトップレベルスクリプト。それぞれ `config/` の YAML config を読み込み、
データフィーダを構築し、`src/training/` の学習/評価ループを駆動する。出力は `results/` 以下
に書き出される（`--output/-o` でディレクトリを上書き可能）。

## ファイル

| ファイル | 役割 |
|---|---|
| `exam_each_seed.py` | 単一アクション・複数 seed 実験。1 つのアクションから FIM 項目を予測し、`action_task_indices` × `fim_item_indices` の組み合わせで K-fold / holdout 検証と統計的分析を行う。`src/training/Run` を使用。 |
| `exam_fusion.py` | 複数アクション fusion 実験。`combined_action_indices` のアクションを Early / Middle / Late Fusion（またはベースライン）で統合する。`src/training/Run` を使用。 |

## 実行方法

```bash
source path.sh
python experiments/exam_each_seed.py -c config/fusion_strategy_ablations/single_action.yaml
python experiments/exam_fusion.py   -c config/fusion_strategy_ablations/early_data.yaml
# -o/--output <dir> で結果ディレクトリを上書き
```

`config/` 以下の各 config フォルダ（それぞれ README を持つ）に、利用可能な条件と実行コマンド
が一覧されている。
