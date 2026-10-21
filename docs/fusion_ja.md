# exam_fusion.py: 複数動作 Fusion FIM 予測実験

`experiments/exam_fusion.py` は、複数のリハビリ動作を統合して FIM 項目を予測する fusion 実験です。設定ファイルの `combined_action_indices` で指定した複数動作を 1 つの入力集合として扱い、`fusion.type` に応じて Early Fusion、Middle Fusion、Late Fusion を実行します。すべての fusion モデルは単一段階の end-to-end 学習を行い、動作ごとの出力は動作軸方向の平均で統合します。

## 実行方法

現在の checkout では、fusion 実験用の設定は `config/fusion_strategy_ablations/`, `config/model_ablation/`, `config/action_selection/`, `config/complement_ablation/` にあります。スクリプトには `config/configSomeTask.yaml` というデフォルト値がありますが、同名ファイルは標準では配置されていないため、通常は `-c` で設定ファイルを明示します。

```bash
# Early Fusion
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/early_data.yaml

# Middle Fusion
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/middle_shallow.yaml

# Late Fusion
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/late_feature.yaml
```

デフォルトの `results/` 以外に保存したい場合は `--output` / `-o` で出力先ディレクトリを指定できます。

```bash
python experiments/exam_fusion.py -c config/fusion_strategy_ablations/early_data.yaml -o /path/to/output
```

`wandb.use_wandb: true` の場合は、各 FIM 項目、seed、Fold の実行が Weights & Biases に記録されます。run の config には fusion type、fusion mode、Fold index、対象 FIM 項目、seed、前処理、学習設定が含まれます。

## 主な設定項目

| Key | 内容 |
| --- | --- |
| `fusion.type` | 統合戦略。系統と変種の両方を 1 つの値で表します。`early_chn_stack` / `early_exp_graph`（入力レベル。チャネルスタックまたはグラフ拡張）、`middle_shallow` / `middle_deep`（中間特徴量。挿入位置）、`late_feature` / `late_probability`（FC 前の特徴量、または動作ごとの確率出力）。 |
| `fusion.missing_representation` | 欠損動作の表現方法。`zero`（ゼロパディング、既定）または `pad_mask`（Late Fusion 限定。各ブランチを存在サンプルのみで順伝搬し欠損スロットをエンコーダに到達させない）。詳細は後述の「欠損動作の表現と Modality Dropout」を参照。 |
| `trainings.modality_dropout_p` | Modality dropout の確率(既定 `0.0` = 無効)。学習時のみ、存在する動作をこの確率で追加的に欠損扱いにします。各サンプルで最低 1 動作は必ず残ります。 |
| `combined_action_indices` | 統合対象の動作 index。`src/utils/names.py` の `Tasks()` に対応します。 |
| `fim_item_indices` | 予測対象 FIM 項目の index。 |
| `trainings.delete_model_after_training` | `true` の場合、Fold 後に一部の一時モデルや中間ディレクトリを削除し、split CSV を Fold 直下へ移動します。 |
| `output_weights` | Attention または GradCAM 重みを出力するかどうか。 |

## 欠損動作の表現と Modality Dropout

欠損動作は `makeSomeTasksDataset()` により全ゼロの `(C, T, V)` スライスとして保存されます。`Feeder` はこの全ゼロスライスからサンプルごと・動作ごとの存在マスク(1 = 存在、0 = 欠損)を導出し、各サンプルの 5 番目の要素として返します。`Train`/`Test` の loop はマスクを fusion model(`accepts_mask = True`)にのみ渡すため、単一動作の `STGCLSATT` 系の処理は変わりません。

`fusion.missing_representation` で 2 手法を選択できます(ヘルパは `src/model/mask_token.py`):

| 手法 | 挙動 |
| --- | --- |
| `zero` | ベースライン。欠損スロットは全ゼロのまま。Modality dropout による疑似欠損スロットもゼロ化されます。 |
| `pad_mask` | Late Fusion 限定。各動作ブランチを存在サンプルのみで順伝搬します(`f[present] = model_list[a](x[present, a])`)。欠損スロットはエンコーダに一切到達しないため BatchNorm 統計を汚染せず、勾配も流れません。集約はマスク平均を使用します。 |

`trainings.modality_dropout_p > 0` で modality dropout が有効になります。学習時のみ、存在する動作をランダムに疑似欠損化し、実マスクと統合した effective mask を選択中の表現手法に渡します。各サンプルで最低 1 動作は必ず残り、推論時は実マスクのみ使用します。実験名は表現方法に応じて `_ZeroPad` または `_Masking` とし、modality dropout 有効時は `_MD` を追加します。Complete-case 前処理は `_CompleteCase` になります。

既知の制限: 動作別 BatchNorm の running statistics にはゼロスライスの影響が混入します(ゼロパディングベースライン。`pad_mask` は例外です。上記参照)。また GradCAM はマスクなしでモデルを再実行するため、CAM は全動作存在の前提で計算されます。表現手法の比較実験では `output_weights: false` を推奨します。

## 実験フロー

1. YAML 設定を読み込み、統合対象動作、対象 FIM 項目、seed、fusion 設定を決定します。
2. `results/<experiment_name>/` を作成し、設定ファイルを `situations.yaml` として保存します。
3. 各 FIM 項目ごとに `makeSomeTasksDataset()` を実行し、複数動作をまとめた train/test split を 5 Fold 作成します。
4. 複数動作をまとめた split で fusion model を end-to-end 学習し、test split で評価します。
5. Fold 結果を seed 単位に集約し、さらに seed 結果を FIM 項目単位、実験全体単位に集約します。

## 内部処理と担当ファイル/関数

### 1. エントリーポイントと設定読み込み

| 処理 | ファイル / 関数 | 詳細 |
| --- | --- | --- |
| CLI 引数の解釈 | `experiments/exam_fusion.py` の `__main__` | `argparse` で `--config` / `-c` と `--output` / `-o` を受け取り、`main(configPath=..., outputDir=...)` に渡します。 |
| 実験全体の制御 | `experiments/exam_fusion.py` の `main()` | YAML 読み込み、FIM 項目、seed、Fold のループ、fusion model 学習、評価、集約、保存を統括します。 |
| フォント設定 | `src/utils/set_plt_font.py` の `setPltFont()` | 図の出力前に matplotlib 用フォントを初期化します。 |
| 動作名と FIM 名の解決 | `src/utils/names.py` の `Tasks()` / `FIMitems()` | `combined_action_indices` と `fim_item_indices` を実際の動作名と FIM 項目名へ変換します。 |
| 実験名の生成 | `src/utils/decides.py` の `decideExamName()` | モデル、ST-GCN 層数、seed 固定モード、fusion type/mode、Attention 設定、開始時刻から `results/<experiment_name>/` を決めます。 |
| seed 固定 | `src/utils/decides.py` の `decideSeed()` | split 用 seed と model 用 seed を `seed_fix_mode` に従って設定し、Python、NumPy、PyTorch、CUDA の乱数を固定します。 |
| W&B config 作成 | `src/utils/decides.py` の `decideWandbConfig()` | seed、Fold、対象 FIM、fusion type/mode、combined action、前処理、モデル、学習設定を W&B 用 dict にまとめます。 |
| モデル振り分け | `src/training/run.py` の `_build_model()` | `config["fusion"]["type"]` で選ばれたモデル(Early/Middle/Late Fusion)を構築します。Middle Fusion の場合は ST-GCN channel を独立部と fusion 後部にも分割します。 |
| Middle Fusion channel 分割 | `src/model/params/separate_channels.py` の `separate_channels()` | `fusion.type` の挿入位置（`middle_shallow` / `middle_deep`）に応じて、各動作で個別に処理する ST-GCN 層と、融合後に処理する ST-GCN 層の channel list を分離します。 |

### 2. 複数動作データセット作成

| 処理 | ファイル / 関数 | 詳細 |
| --- | --- | --- |
| データセット作成の入口 | `src/data/dataset/make_some_tasks_dataset.py` の `makeSomeTasksDataset()` | 1 つの FIM 項目について、`combined_action_indices` の複数動作をまとめて読み込み、5-fold split を作成します。 |
| データ格納先の選択 | `makeSomeTasksDataset()` | 入力ルートは常に `data/ConciseData` です。中心化は読み込み時(`norm_centering_mode: First`)に適用されます。 |
| 被験者一覧の決定 | `makeSomeTasksDataset()` | `data_list_path` が存在する場合は CSV の `Directory` 列を使い、空または未指定の場合は対象データディレクトリを走査します。 |
| FIM score 取得 | `makeSomeTasksDataset()` | 各被験者の `FIM.csv` から対象 FIM 項目の `Score` を読み、無効 score を除外します。 |
| class label 変換 | `makeSomeTasksDataset()` | 元の FIM score を `classes` に合わせて 0-based class index に変換します。 |
| 動作欠損処理 | `makeSomeTasksDataset()` | 指定動作が存在しない場合、`is_fill_actionless` と `complement_actionless` に従って、`preprocess.complement_map`(タスクごとの優先順位付き候補リスト。未指定時は従来の座位⇔立位の対応関係がデフォルト)でフォールバック動作を補完するか、ゼロ配列で埋めるか、サンプルを除外します。 |
| 骨格 CSV の NumPy 化 | `src/data/dataset/utils/dataframe_to_numpy.py` の `dataframeToNumpy()` | 各動作ディレクトリから position / velocity / acceleration の CSV を読み、`frame_length` に揃えた配列へ変換します。 |
| 動作開始フレーム検出 | `src/data/dataset/utils/detect_movement_onset.py` の `detectMovementOnset()` | `use_moving_frame: true` の場合、速度系列から切り出し開始フレームを推定します。 |
| 複数動作の結合 | `makeSomeTasksDataset()` | 各被験者について、動作ごとの NumPy 配列を action 軸に積み上げ、fusion model 用の multi-action 配列を作ります。 |
| Fold 分割 | `makeSomeTasksDataset()` | `StratifiedGroupKFold(n_splits=5)` で class 分布を保ちつつ被験者 ID 単位の group split を作ります。 |
| split 出力 | `makeSomeTasksDataset()` | 各 Fold に `train/data.npy`, `train/target.npy`, `train/trainDir.csv`, `train/train_distribution.csv`, `test/data.npy`, `test/target.npy`, `test/testDir.csv`, `test/test_distribution.csv` を保存します。 |

### 3. DataLoader とサンプリング

| 処理 | ファイル / 関数 | 詳細 |
| --- | --- | --- |
| Dataset 初期化 | `src/data/feeder/feeder.py` の `Feeder.__init__()` | `.npy` と `.csv` から骨格データ、label、`Directory`、`RealScore` を読み、全ゼロスライスから動作存在マスクを導出します。 |
| サンプル取得 | `src/data/feeder/feeder.py` の `Feeder.__getitem__()` | `data, label, dir, real_score, mask` を返します。fusion では `data` が複数動作を含む配列になり、`mask` が動作の存在を示します。 |
| データ拡張 | `src/data/feeder/tools.py` の `random_move()` | 訓練時に `preprocess.use_data_transform: true` なら、50% の確率でランダム変換を適用します。 |
| class 不均衡対応 | `experiments/exam_fusion.py` の `main()` | `target.npy` の class count から sample weight を作り、`trainings.is_weighted_random_sampler: true` の場合は `WeightedRandomSampler` を使います。 |
| DataLoader 作成 | `experiments/exam_fusion.py` の `main()` | 複数動作をまとめた split から fusion train/test loader を作成します。 |

### 4. Fusion モデル学習

| 処理 | ファイル / 関数 | 詳細 |
| --- | --- | --- |
| Fusion 学習入口 | `src/training/run.py` の `Run()` | fusion type に応じてモデルを作成し、end-to-end で学習と評価を実行します。単一モデル用ドライバ(`exam_each_seed.py`)とも共通です。 |
| Late Fusion model | `src/model/late_fusion.py` の `LateFusion` | 各動作の出力（`late_probability`）、または FC 前特徴量（`late_feature`）を平均で融合します。 |
| Middle Fusion model | `src/model/middle_fusion.py` の `MiddleFusion` | 各動作を前段 ST-GCN で処理したあと、中間特徴を融合し、後段 ST-GCN と分類 head へ流します。 |
| Early Fusion model | `src/model/early_fusion.py` の `EarlyFusion` | 入力レベルで複数動作を統合します。channel stack（`early_chn_stack`）または graph expansion（`early_exp_graph`）を使います。 |
| Fusion 学習 loop | `src/training/train.py` の `Train.train()` | `trainings` の設定を用いて epoch ごとの forward、loss、backprop、optimizer step、scheduler step を実行します。 |
| 欠損マスク処理 | `src/training/train.py` の `Train.train()` と `src/model/mask_token.py` | 存在マスクに `apply_modality_dropout()` を適用し、effective mask を fusion model の `forward(x, mask)` に渡します。 |
| Fusion 学習ログ | `src/training/train.py` の `Train.train()` | Fold 直下に `ELLA_ep<N>.csv` を保存します。 |
| Fusion checkpoint | `src/training/train.py` の `Train.train()` | Fold 直下に最終 fusion model の `checkpoint.pth` を保存します。 |

### 5. 評価と Fold 出力

| 処理 | ファイル / 関数 | 詳細 |
| --- | --- | --- |
| 評価入口 | `src/training/test.py` の `Test.test()` | `checkpoint.pth` を読み込み、test DataLoader に対して推論します。fusion model には実際の存在マスクを渡します(推論時に modality dropout は適用されません)。 |
| 指標計算 | `Test.test()` | `Accuracy`, `Balanced Accuracy`, class 別精度、F1、MCC、Precision、Recall、AUCROC などを計算します。 |
| Fold 混同行列 | `Test.test()` | Fold 単位の rate 混同行列を `confusion_matrix.png` として保存します。 |
| 推論 CSV | `Test.test()` | `Directory`, `Predicted`, `True`, `RealScore`, `Prob_*` を `PredictTruelabel.csv` に保存します。 |
| Attention / CAM 出力 | `Test.output_weights()` | `output_weights: true` かつモデルが重みを返す場合、`weights_Att.h5` または `weights_CAM.h5` を保存します。fusion model では `Directory` ごとの group に action index 別 dataset を作ります。 |
| 結果保持 | `src/data/feeder/results.py` の `Results.append()` | Fold の accuracy、balanced accuracy、class 別精度、予測 label、正解 label を蓄積します。 |

### 6. 集約と後処理

| 処理 | ファイル / 関数 | 詳細 |
| --- | --- | --- |
| Fold 平均 | `src/data/feeder/results.py` の `Results.appendMean()` | 5 Fold の平均値を計算し、`Stratified_Seed<seed>/Accuracies.csv` の `Mean` 行に使います。 |
| seed 単位混同行列 | `src/utils/make_confusion_matrix.py` の `makeConfMatrixOnExam()` | 5 Fold 全体の予測と正解から count / rate 混同行列を CSV と `png/pdf/svg` で保存します。 |
| seed 結果 CSV | `experiments/exam_fusion.py` の `main()` | `Stratified_Seed<seed>/Accuracies.csv` に Fold 0-4 と `Mean` の指標を保存します。 |
| FIM 項目単位集約 | `experiments/exam_fusion.py` の `main()` と `Results` | 各 seed の Fold 平均を集約し、`<ItemIndex>-<FIMItem>/Accuracies.csv` と FIM 項目単位の混同行列を保存します。 |
| 実験全体集約 | `experiments/exam_fusion.py` の `main()` | 全 FIM 項目の結果を `results/<experiment_name>/Accuracies.csv` に結合し、`Seed == "Mean"` の行だけを `results.csv` に保存します。 |
| 任意の削除処理 | `experiments/exam_fusion.py` の `main()` | `trainings.delete_model_after_training: true` の場合に、一時ディレクトリを削除し、split CSV を Fold 直下へ移動します。 |

## 出力フォルダ構造

`<experiment_name>` はモデル名、ST-GCN 層数、seed 固定モード、fusion 設定、Attention 種別、開始時刻から自動生成されます。

```text
results/<experiment_name>/
├── situations.yaml
├── Accuracies.csv
├── results.csv
└── <ItemIndex>-<FIMItem>/
    ├── Accuracies.csv
    ├── <FIMItem>_<Model>_countConfMtx.csv
    ├── <FIMItem>_<Model>_countConfMtx.pdf
    ├── <FIMItem>_<Model>_countConfMtx.svg
    ├── <FIMItem>_<Model>_countConfMtx.png
    ├── <FIMItem>_<Model>_rateConfMtx.csv
    ├── <FIMItem>_<Model>_rateConfMtx.pdf
    ├── <FIMItem>_<Model>_rateConfMtx.svg
    ├── <FIMItem>_<Model>_rateConfMtx.png
    └── Stratified_Seed<seed>/
        ├── Accuracies.csv
        ├── <FIMItem>_<Model>_countConfMtx.*
        ├── <FIMItem>_<Model>_rateConfMtx.*
        └── Fold<fold>/
            ├── train/
            │   ├── data.npy
            │   ├── target.npy
            │   ├── trainDir.csv
            │   └── train_distribution.csv
            ├── test/
            │   ├── data.npy
            │   ├── target.npy
            │   ├── testDir.csv
            │   └── test_distribution.csv
            ├── checkpoint.pth
            ├── ELLA_ep<N>.csv
            ├── confusion_matrix.png
            ├── PredictTruelabel.csv
            ├── weights_Att.h5        # optional
            └── weights_CAM.h5        # optional
```

`<ActionIndex>` は `combined_action_indices` の値に 1 を足して 2 桁ゼロ埋めした文字列です。例: index `0` は `01` です。`trainDir.csv`/`testDir.csv` の `use_<ActionName>` 列や `weights_Att.h5`/`weights_CAM.h5` の動作別 group に登場します。

`trainings.delete_model_after_training: true` の場合は、Fold 終了後に `train/`, `test/` などの中間ディレクトリが削除されます。その際、`trainDir.csv`, `testDir.csv`, `train_distribution.csv`, `test_distribution.csv` は Fold 直下へ移動されます。

## 出力ファイルの内容

### `situations.yaml`

実行時に読み込んだ YAML 設定のコピーです。fusion type、fusion mode、対象動作、対象 FIM 項目、seed、前処理、学習設定、W&B 設定などを保存します。

### `results/<experiment_name>/Accuracies.csv`

全 FIM 項目の seed 別集約結果です。

| Column | 内容 |
| --- | --- |
| `Seed` | 実行 seed。各 FIM 項目の末尾に `Mean` 行が入ります。 |
| `ToFIM` | 予測対象 FIM 項目名。 |
| `Accuracy` | 各 seed の 5 Fold 平均精度。単位は percent です。 |
| `Balanced_Accuracy` | 各 seed の 5 Fold 平均 balanced accuracy。 |
| `Class<score>_Accuracy` | 設定した `classes` ごとのクラス別正解率。 |

### `results/<experiment_name>/results.csv`

ルート `Accuracies.csv` から `Seed == "Mean"` の行だけを抜き出し、`Seed` 列を削除した要約です。fusion 条件間の FIM 項目別平均性能を比較するためのファイルです。

### `<ItemIndex>-<FIMItem>/Accuracies.csv`

1 つの FIM 項目について、seed ごとの集約結果を保存します。各 seed 行はその seed の 5 Fold 結果の平均で、末尾に seed 間平均の `Mean` 行が入ります。

### `Stratified_Seed<seed>/Accuracies.csv`

1 つの seed 内の Fold 別結果です。

| Column | 内容 |
| --- | --- |
| `Fold` | `0` から `4` の Fold 番号と、末尾の `Mean` 行。 |
| `Accuracy` | Fold ごとの通常精度。 |
| `Balanced_Accuracy` | Fold ごとの balanced accuracy。 |
| `Class<score>_Accuracy` | Fold ごとのクラス別正解率。 |

### `Fold<fold>/train/data.npy`

fusion model の train split です。サンプル、動作、チャンネル、時間、関節の順に複数動作データを保持します。実際の shape は `attributes`, `frame_length`, skeleton layout、`combined_action_indices` に依存します。

### `target.npy`

train/test split の 0-based class label 配列です。元の FIM score は `RealScore` として `trainDir.csv` / `testDir.csv` に残ります。

### `trainDir.csv` / `testDir.csv`

split に含まれるサンプルのメタ情報です。

| Column | 内容 |
| --- | --- |
| `Directory` | データ元ディレクトリ名。 |
| `<ActionName>` | `use_moving_frame: true` の場合に出力される、各動作の窓開始フレーム。 |
| `RealScore` | 元の FIM score。 |
| `use_<ActionName>` | 実際に使用した動作名。欠損補完時は補完元や `None` が記録されます。 |

### `train_distribution.csv` / `test_distribution.csv`

train/test split の class count です。index に `classes` と `Total` が入り、分割後のクラス分布を確認できます。

### `Fold<fold>/ELLA_ep<N>.csv`

fusion model の学習ログです。`<N>` は最終 epoch index です。

| Column | 内容 |
| --- | --- |
| `Epoch` | epoch 番号。 |
| `TrainLoss` | epoch 平均の学習損失。 |
| `LearningRate` | epoch 終了時の learning rate。 |
| `TrainAccuracy` | epoch 内の学習データ正解率。 |

### `checkpoint.pth`

Fold の最終 fusion model の `state_dict` です。評価時にはこの重みが読み込まれます。

### `PredictTruelabel.csv`

Fold の test split に対するサンプル単位の推論結果です。

| Column | 内容 |
| --- | --- |
| `Directory` | 被験者または試行を識別するディレクトリ名。 |
| `Predicted` | 予測した 0-based class index。 |
| `True` | 正解の 0-based class index。 |
| `RealScore` | 元の FIM score。 |
| `Prob_0`, `Prob_1`, ... | 各 class index のモデル出力確率。 |

このファイルは cross-subject 集計や ablation plot の入力です。Fold 結合 report を実行すると、`PredictTruelabel_all_folds.csv` が `Stratified_Seed<seed>/` 直下に作成されます。

### `confusion_matrix.png`

Fold 単位の rate 混同行列です。横軸が予測 FIM score、縦軸が正解 FIM score で、タイトルに `Accuracy` と `Balanced Accuracy` が表示されます。

### `*_countConfMtx.*`

Fold を seed 単位にまとめた混同行列、または seed を FIM 項目単位にまとめた混同行列です。`.csv` は count、`.png/.pdf/.svg` は count heatmap です。

### `*_rateConfMtx.*`

`*_countConfMtx.*` と同じ対象に対して、正解 class ごとに 100% 正規化した混同行列です。`.csv` は割合、画像ファイルは rate heatmap です。

### `weights_Att.h5` / `weights_CAM.h5`

`output_weights: true` かつモデルが Attention または GradCAM 重みを返す場合に出力されます。fusion model では HDF5 内に `Directory` ごとの group が作られ、その下に統合対象動作 index に対応する dataset が保存されます。値はモデル固有の時空間重み配列です。
