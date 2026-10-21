# src/data/dataset — HDF5 データセット生成

前処理済みの `ConciseData` から、feeder が読み込む `train.h5` / `test.h5` を生成する。
単一アクションデータセットと複数アクション（fusion）データセットの両方に対応し、
被験者の全サンプルを 1 つの fold 内に収める StratifiedGroupKFold 分割を行う。

## ファイル

| ファイル | 役割 |
|---|---|
| `make_single_dataset.py` | 1 つの（アクションタスク, FIM 項目）ペア用データセットを生成。被験者単位の StratifiedGroupKFold 交差検証分割を適用。 |
| `make_some_tasks_dataset.py` | 複数アクションタスクを統合した融合データセットを生成。統合前にアクションごとのサンプルを整列・補完する。 |

## サブフォルダ

| サブフォルダ | 役割 |
|---|---|
| [`utils/`](utils/) | データセット生成ヘルパ。動作開始フレーム検出と DataFrame→NumPy 変換。 |
