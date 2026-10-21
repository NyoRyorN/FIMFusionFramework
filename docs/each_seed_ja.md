# exam_each_seed.py: 単一動作 FIM 予測実験

`experiments/exam_each_seed.py` は、1 つのリハビリ動作から 1 つの FIM 項目を予測する単一動作ベースライン実験です。設定ファイルで指定された `action_task_indices` と `fim_item_indices` の全組み合わせに対して、複数 seed の 5-fold 交差検証を実行し、Fold 単位、seed 単位、動作と FIM 項目の組み合わせ単位、実験全体単位の結果を `results/` に保存します。

## 実行方法

現在の checkout では、単一動作実験用の設定は `config/fusion_strategy_ablations/single_action.yaml` にあります。スクリプトには `config/config.yaml` というデフォルト値がありますが、同名ファイルは標準では配置されていないため、通常は `-c` で設定ファイルを明示します。

```bash
python experiments/exam_each_seed.py -c config/fusion_strategy_ablations/single_action.yaml
```

デフォルトの `results/` 以外に保存したい場合は `--output` / `-o` で出力先ディレクトリを指定できます。

```bash
python experiments/exam_each_seed.py -c config/fusion_strategy_ablations/single_action.yaml -o /path/to/output
```

`wandb.use_wandb: true` の場合は、各 Fold が Weights & Biases run として記録されます。`wandb.entity` には利用する entity または team 名を指定します。

## 主な設定項目

| Key | 内容 |
| --- | --- |
| `model.name` | 使用するモデル名。`STGCN`, `STGCLSAtt`, `STGCAGLS` など。ベースライン: `LITEMV`, `ConvTran`, `VanTran` (DeepRehabPile)、`SgcnLstm` (STGCN-rehab)、`DDGCN` (DD-GCN)。 |
| `action_task_indices` | 入力として使う単一動作の index。`src/utils/names.py` の `Tasks()` に対応します。 |
| `fim_item_indices` | 予測対象 FIM 項目の index。`src/utils/names.py` の `FIMitems()` に対応します。 |
| `seeds` | データ分割とモデル学習を繰り返す seed のリスト。 |
| `classes` | 分類対象にする FIM スコア。コード内部の教師ラベルは 0-based に変換されます。 |
| `data_list_path` | 使用する被験者ディレクトリ一覧 CSV。空文字の場合は対象データディレクトリを自動走査します。 |
| `seed_fix_mode` | `split` は分割 seed を 42 に固定、`model` はモデル学習 seed を 42 に固定、`change` は指定 seed を分割と学習の両方に使います。 |
| `preprocess.*` | 正規化、フレーム長、動的フレーム、短系列補完などの前処理設定。 |
| `trainings.*` | epoch、optimizer、scheduler、batch size、weighted sampler などの学習設定。 |
| `output_weights` | Attention または GradCAM 重みを出力するかどうか。モデル側が重み取得に対応している場合のみ有効です。 |

## 実験フロー

1. YAML 設定を読み込み、対象動作、対象 FIM 項目、seed、class を決定します。
2. `decideExamName()` により `results/<experiment_name>/` を作成し、使用した設定を `situations.yaml` としてコピーします。
3. 各 `action -> FIM item` の組み合わせごとに `makeSingleDataset()` を実行し、被験者 ID を group とした `StratifiedGroupKFold(n_splits=5)` の train/test split を作成します。
4. 各 Fold で STGCLSATT 系モデルを学習し、同じ Fold の test split で評価します。
5. Fold 結果を seed 単位に集約し、さらに seed 結果を動作と FIM 項目の組み合わせ単位、最後に実験全体へ集約します。

## 内部処理と担当ファイル/関数

### 1. エントリーポイントと設定読み込み

| 処理 | ファイル / 関数 | 詳細 |
| --- | --- | --- |
| CLI 引数の解釈 | `experiments/exam_each_seed.py` の `__main__` | `argparse` で `--config` / `-c` と `--output` / `-o` を受け取り、`main(configPath=..., outputDir=...)` に渡します。 |
| 実験全体の制御 | `experiments/exam_each_seed.py` の `main()` | YAML 読み込み、対象動作・対象 FIM 項目・seed・Fold のループ、学習、評価、集約、ファイル保存までを統括します。 |
| フォント設定 | `src/utils/set_plt_font.py` の `setPltFont()` | matplotlib で日本語や図中テキストを描画するためのフォント設定を初期化します。 |
| 対象名の解決 | `src/utils/names.py` の `Tasks()` / `FIMitems()` | config の `action_task_indices` と `fim_item_indices` を、実際の動作名と FIM 項目名へ変換します。 |
| 実験名の生成 | `src/utils/decides.py` の `decideExamName()` | モデル名、ST-GCN 層数、seed 固定モード、Attention 設定、開始時刻から `results/<experiment_name>/` の名前を作ります。 |
| seed 固定 | `src/utils/decides.py` の `decideSeed()` | Python、NumPy、PyTorch、CUDA の乱数 seed と cuDNN の deterministic 設定を揃えます。`seed_fix_mode` に応じて分割用 seed とモデル学習用 seed が変わります。 |
| W&B config 作成 | `src/utils/decides.py` の `decideWandbConfig()` | seed、Fold、対象動作、対象 FIM、前処理、モデル、学習設定を W&B に記録する dict に整形します。 |

### 2. 単一動作データセット作成

| 処理 | ファイル / 関数 | 詳細 |
| --- | --- | --- |
| データセット作成の入口 | `src/data/dataset/make_single_dataset.py` の `makeSingleDataset()` | 1 動作と 1 FIM 項目の組み合わせについて、対象データの探索、FIM ラベル読み込み、NumPy 化、5-fold split、split ファイル保存を行います。 |
| データ格納先の選択 | `makeSingleDataset()` | 入力ルートは常に `data/ConciseData` です。中心化は読み込み時（`norm_centering_mode: First`）に適用されます。 |
| 被験者一覧の決定 | `makeSingleDataset()` | `data_list_path` が存在する場合は CSV の `Directory` 列を使い、空または未指定の場合は対象データディレクトリを走査します。 |
| FIM score 取得 | `makeSingleDataset()` | 各被験者ディレクトリの `FIM.csv` を読み、対象 FIM 項目の `Score` を取得します。0 以下または NaN の score は除外します。 |
| class label 変換 | `makeSingleDataset()` | FIM score を `classes` の数に合わせて 0-based class index へ変換します。例えば `classes: [6,7]` では score 6 と 7 が内部 label 0 と 1 に対応します。 |
| 骨格 CSV の NumPy 化 | `src/data/dataset/utils/dataframe_to_numpy.py` の `dataframeToNumpy()` | `p_x.csv`, `p_y.csv`, `p_z.csv` など、`attributes` で指定した position / velocity / acceleration の各軸 CSV を読み、モデル入力用配列へ変換します。 |
| 動作開始フレーム検出 | `src/data/dataset/utils/detect_movement_onset.py` の `detectMovementOnset()` | `use_moving_frame: true` の場合に速度系列から切り出し開始フレームを推定し、`frame_length` 分の窓を作ります。 |
| 短系列補完 | `dataframeToNumpy()` | 入力系列が `frame_length` より短い場合、`is_fill_short: true` ならゼロパディングし、`false` ならそのサンプルを無効扱いにします。 |
| 座標中心化 | `dataframeToNumpy()` | `norm_centering_mode: First` の場合は先頭フレームの `norm_center_point` を基準に position を中心化します。 |
| Fold 分割 | `makeSingleDataset()` | `StratifiedGroupKFold(n_splits=5)` で、class 分布を保ちながら被験者 ID 単位の group split を作ります。group は `Directory` の `_` より前の文字列です。 |
| split 出力 | `makeSingleDataset()` | 各 Fold に `train/data.npy`, `train/target.npy`, `train/trainDir.csv`, `train/train_distribution.csv`, `test/data.npy`, `test/target.npy`, `test/testDir.csv`, `test/test_distribution.csv` を保存します。 |

### 3. DataLoader とサンプリング

| 処理 | ファイル / 関数 | 詳細 |
| --- | --- | --- |
| Dataset 初期化 | `src/data/feeder/feeder.py` の `Feeder.__init__()` | `.npy` の骨格データと label、`trainDir.csv` / `testDir.csv` の `Directory` と `RealScore` を読み込みます。NaN が含まれる場合は停止します。 |
| サンプル取得 | `src/data/feeder/feeder.py` の `Feeder.__getitem__()` | DataLoader から呼ばれ、`data, label, dir, real_score` を返します。 |
| データ拡張 | `src/data/feeder/tools.py` の `random_move()` | `preprocess.use_data_transform: true` の訓練時のみ、50% の確率でランダム変換を適用します。評価時は常に無効です。 |
| class 不均衡対応 | `experiments/exam_each_seed.py` の `main()` | `train/target.npy` から class count を求め、`trainings.is_weighted_random_sampler: true` の場合は `WeightedRandomSampler` で少数 class を補正します。 |
| DataLoader 作成 | `experiments/exam_each_seed.py` の `main()` | train は weighted sampler または shuffle、test は `shuffle=False` で DataLoader を作成します。 |

### 4. モデル学習

| 処理 | ファイル / 関数 | 詳細 |
| --- | --- | --- |
| 学習・評価の入口 | `src/training/run.py` の `Run()` | `STGCLSATT` モデル、loss、optimizer、`Train`、`Test` を組み立て、1 Fold 分の学習と評価を実行します。 |
| モデル構築 | `src/model/build_model.py` の `build_encoder()` | `model.name` から `STGCLSATT` (`src/model/stgclsatt.py`) または DeepRehabPile ベースライン (`src/model/baselines/`) を解決して構築します。 |
| optimizer 作成 | `src/training/run.py` の `Run()` | `trainings.optimizer` に応じて Adam / SGD / AdamW を作成し、`lr`, `momentum`, `weight_decay` を反映します。 |
| 学習 loop | `src/training/train.py` の `Train.train()` | epoch ごとに forward、NLLLoss、必要に応じて AGC-LSTM attention regularization、backprop、optimizer step、scheduler step を実行します。 |
| scheduler | `src/training/train.py` の `Train.set_scheduler()` | `OneCycleLR`, `ReduceLROnPlateau`, `MultiStepLR`, `None` を config に応じて設定します。 |
| 学習ログ保存 | `src/training/train.py` の `Train.train()` | epoch ごとの `Epoch`, `TrainLoss`, `LearningRate`, `TrainAccuracy` を `ELLA_ep<N>.csv` に保存します。 |
| checkpoint 保存 | `src/training/train.py` の `Train.train()` | 最終 epoch の model `state_dict` を `checkpoint.pth` に保存します。 |

### 5. 評価と Fold 出力

| 処理 | ファイル / 関数 | 詳細 |
| --- | --- | --- |
| 評価入口 | `src/training/test.py` の `Test.test()` | `checkpoint.pth` を読み込み、test DataLoader に対して推論します。 |
| 指標計算 | `Test.test()` | `Accuracy`, `Balanced Accuracy`, class 別正解率、F1、MCC、Precision、Recall、AUCROC などを計算します。W&B 有効時は一部指標を log します。 |
| Fold 混同行列 | `Test.test()` | Fold 単位の rate 混同行列を `confusion_matrix.png` として保存します。 |
| 推論 CSV | `Test.test()` | `Directory`, `Predicted`, `True`, `RealScore`, `Prob_*` を `PredictTruelabel.csv` に保存します。 |
| Attention / CAM 出力 | `Test.output_weights()` | `output_weights: true` かつモデルが重みを返す場合に、`weights_Att.h5` または `weights_CAM.h5` を保存します。 |
| 結果保持 | `src/data/feeder/results.py` の `Results.append()` | Fold の accuracy、balanced accuracy、class 別精度、予測 label、正解 label をメモリ上に蓄積します。 |

### 6. 集約と後処理

| 処理 | ファイル / 関数 | 詳細 |
| --- | --- | --- |
| Fold 平均 | `src/data/feeder/results.py` の `Results.appendMean()` | 5 Fold の平均値を計算し、`Stratified_Seed<seed>/Accuracies.csv` の `Mean` 行として出力できる形にします。 |
| seed 単位混同行列 | `src/utils/make_confusion_matrix.py` の `makeConfMatrixOnExam()` | 5 Fold 全体の `true_labels` / `pred_labels` から count / rate 混同行列を CSV と `png/pdf/svg` で保存します。 |
| seed 結果 CSV | `experiments/exam_each_seed.py` の `main()` | `Stratified_Seed<seed>/Accuracies.csv` を作成し、Fold 0-4 と `Mean` の指標を保存します。 |
| 組み合わせ単位集約 | `experiments/exam_each_seed.py` の `main()` と `Results` | 各 seed の Fold 平均をさらに集約し、`<Action>_to_<FIMItem>/Accuracies.csv` と組み合わせ単位の混同行列を保存します。 |
| 実験全体集約 | `experiments/exam_each_seed.py` の `main()` | 全組み合わせの結果を `results/<experiment_name>/Accuracies.csv` に結合し、`Seed == "Mean"` の行だけを `results.csv` に保存します。 |
| Fold 後クリーンアップ | `experiments/exam_each_seed.py` の `main()` | 学習後に `*_*.pth` を削除し、`train/` と `test/` 内の CSV を Fold 直下へ移動して一時ディレクトリを削除します。 |

## 出力フォルダ構造

`<experiment_name>` は、モデル名、ST-GCN 層数、seed 固定モード、Attention 種別、開始時刻から自動生成されます。

```text
results/<experiment_name>/
├── situations.yaml
├── Accuracies.csv
├── results.csv
└── <Action>_to_<ItemIndex>-<FIMItem>/
    ├── Accuracies.csv
    ├── <Action>to<FIMItem>_<Model>_countConfMtx.csv
    ├── <Action>to<FIMItem>_<Model>_countConfMtx.pdf
    ├── <Action>to<FIMItem>_<Model>_countConfMtx.svg
    ├── <Action>to<FIMItem>_<Model>_countConfMtx.png
    ├── <Action>to<FIMItem>_<Model>_rateConfMtx.csv
    ├── <Action>to<FIMItem>_<Model>_rateConfMtx.pdf
    ├── <Action>to<FIMItem>_<Model>_rateConfMtx.svg
    ├── <Action>to<FIMItem>_<Model>_rateConfMtx.png
    └── Stratified_Seed<seed>/
        ├── Accuracies.csv
        ├── <Action>to<FIMItem>_<Model>_countConfMtx.*
        ├── <Action>to<FIMItem>_<Model>_rateConfMtx.*
        └── Fold<fold>/
            ├── checkpoint.pth
            ├── ELLA_ep<N>.csv
            ├── confusion_matrix.png
            ├── PredictTruelabel.csv
            ├── trainDir.csv
            ├── train_distribution.csv
            ├── testDir.csv
            ├── test_distribution.csv
            ├── weights_Att.h5        # optional
            └── weights_CAM.h5        # optional
```

`exam_each_seed.py` は Fold 評価後に `train/` と `test/` ディレクトリ内の CSV を Fold 直下へ移動し、split 用の `.npy` ファイルを含む一時ディレクトリを削除します。そのため、完走後の Fold 直下には上記の CSV と評価出力が残ります。

## 出力ファイルの内容

### `situations.yaml`

実行時に読み込んだ YAML 設定ファイルのコピーです。`results/<experiment_name>/` 作成時に保存され、読み取り専用に変更されます。モデル、前処理、対象動作、対象 FIM 項目、seed、class、W&B 設定を後から確認するための再現性情報です。

### `results/<experiment_name>/Accuracies.csv`

全ての `action -> FIM item` 組み合わせをまとめた seed 別結果です。

| Column | 内容 |
| --- | --- |
| `Seed` | 実行 seed。各組み合わせの末尾に `Mean` 行が入ります。 |
| `FromAction` | 入力に使った動作名。 |
| `ToFIM` | 予測対象の FIM 項目名。 |
| `Accuracy` | 5 Fold 平均の通常精度。単位は percent です。 |
| `Balanced_Accuracy` | 5 Fold 平均の balanced accuracy。クラス不均衡を考慮した指標です。 |
| `Class<score>_Accuracy` | 設定した `classes` ごとのクラス別正解率。列名の `<score>` は FIM score 値です。 |

### `results/<experiment_name>/results.csv`

ルート `Accuracies.csv` から `Seed == "Mean"` の行だけを抜き出し、`Seed` 列を落とした要約です。各 `FromAction` と `ToFIM` の平均性能を比較するためのファイルです。

### `<Action>_to_<ItemIndex>-<FIMItem>/Accuracies.csv`

1 つの動作と 1 つの FIM 項目の組み合わせについて、seed ごとの集約結果を保存します。`Seed` 列には各 seed と `Mean` が入り、指標列はルート `Accuracies.csv` と同じです。ここでの各 seed 行は、その seed の 5 Fold 結果を平均した値です。

### `Stratified_Seed<seed>/Accuracies.csv`

1 つの seed 内の 5 Fold 評価結果です。

| Column | 内容 |
| --- | --- |
| `Fold` | `0` から `4` の Fold 番号と、末尾の `Mean` 行。 |
| `Accuracy` | 各 Fold の通常精度。 |
| `Balanced_Accuracy` | 各 Fold の balanced accuracy。 |
| `Class<score>_Accuracy` | 各 Fold のクラス別正解率。 |

### `Fold<fold>/PredictTruelabel.csv`

Fold の test split に対するサンプル単位の推論結果です。

| Column | 内容 |
| --- | --- |
| `Directory` | 被験者または試行を識別するディレクトリ名。 |
| `Predicted` | モデルが予測した 0-based class index。 |
| `True` | 正解の 0-based class index。 |
| `RealScore` | 元の FIM score。`classes` への丸め込み前の実スコアです。 |
| `Prob_0`, `Prob_1`, ... | 各 class index に対するモデル出力確率。列数は `len(classes)` です。 |

このファイルは cross-subject 集計や ablation plot 用 report script の入力になります。report script で Fold を結合する場合は、`PredictTruelabel_all_folds.csv` が `Stratified_Seed<seed>/` 直下に作成されます。

### `Fold<fold>/ELLA_ep<N>.csv`

学習ログです。`<N>` は最終 epoch index です。

| Column | 内容 |
| --- | --- |
| `Epoch` | epoch 番号。 |
| `TrainLoss` | epoch 平均の学習損失。 |
| `LearningRate` | epoch 終了時の learning rate。 |
| `TrainAccuracy` | epoch 内の学習データ正解率。 |

### `checkpoint.pth`

Fold で学習されたモデルの `state_dict` です。`exam_each_seed.py` の通常実行では最終単一動作モデルとして Fold 直下に残ります。

### `trainDir.csv` / `testDir.csv`

train/test split に含まれるサンプルのメタ情報です。

| Column | 内容 |
| --- | --- |
| `Directory` | データ元ディレクトリ名。 |
| `FrameStart` | `use_moving_frame` により切り出した窓の開始フレーム。 |
| `RealScore` | 元の FIM score。 |

### `train_distribution.csv` / `test_distribution.csv`

train/test split のクラス分布です。index に `classes` の値と `Total` が入り、値として各 class の件数と総数を保持します。分割の偏り確認に使います。

### `confusion_matrix.png`

Fold 単位の rate 混同行列です。横軸が予測 FIM score、縦軸が正解 FIM score です。タイトルに Fold の `Accuracy` と `Balanced Accuracy` が表示されます。

### `*_countConfMtx.*`

Fold を seed 単位にまとめた混同行列、または seed を組み合わせ単位にまとめた混同行列です。`.csv` は実数 count、`.png/.pdf/.svg` は count heatmap です。

### `*_rateConfMtx.*`

`*_countConfMtx.*` と同じ対象に対して、正解 class ごとに 100% へ正規化した混同行列です。`.csv` は割合、画像ファイルは rate heatmap です。

### `weights_Att.h5` / `weights_CAM.h5`

`output_weights: true` かつモデルが Attention または GradCAM 重みを返す場合に出力されます。HDF5 内のキーは test sample の `Directory` に対応し、値はモデル固有の時空間重み配列です。重みの次元は使用モデルと Attention/CAM 実装に依存します。
