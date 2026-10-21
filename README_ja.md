# Multi-Action Fusion for Skeleton-Based FIM Score Prediction

[English version](README.md)

本リポジトリは、複数のリハビリ動作から FIM
（Functional Independence Measure）項目スコアを予測するための実験コードです。
単一動作ベースラインと、複数動作を統合する Early / Middle / Late Fusion 実験を実行できます。

対象論文:

> **Skeleton-Based Multi-Action Fusion Framework for Binary Classification of FIM Score from Simplified Rehabilitation Actions**
>
> Jun Masaki, Ariaki Higashi, Naoko Shinagawa, Kazuhiko Hirata, Yuichi Kurita, and Akira Furui
>
> *IEEE Transactions on Neural Systems and Rehabilitation Engineering(TNSRE), 2026*
>
> DOI: 10.1109/TNSRE.2026.3744185

## 概要

中心となるモデルは ST-GCN、RNN/AGC-LSTM、Spatio-Temporal Attention を組み合わせた
STGCLSATT 系の骨格系列分類モデルです。実験では、1 つの動作から FIM 項目を予測する
単一動作実験と、複数動作を 1 つの予測に統合する fusion 実験を扱います。

| 実験 | エントリーポイント | 内容 |
| --- | --- | --- |
| 単一動作実験 | `experiments/exam_each_seed.py` | `action_task_indices` と `fim_item_indices` の組み合わせごとに、複数 seed の 5-fold 評価を行います。 |
| Fusion 実験 | `experiments/exam_fusion.py` | `combined_action_indices` で指定した複数動作を Early / Middle / Late Fusion で統合します。 |
| Cross-subject 集計 | `reports/calculates/calc_cross_subject_variables.py` | Fold ごとの `PredictTruelabel.csv` を結合し、被験者単位の一致率を集計します。 |
| Ablation 可視化 | `reports/ablation_boxplot/ablation_cross_subject_variables.py` | cross-subject 集計結果から、条件別の violin + box plot を作成します。 |
| CAM weight 可視化 | `reports/skeleton_plots/` | 保存済み Attention / GradCAM 重みを骨格動画、fusion combined view、GradCAM 分布 plot として可視化します。 |

実験スクリプトの詳細は以下も参照してください。

- [単一動作実験: 日本語](docs/each_seed_ja.md)
- [単一動作実験: 英語](docs/each_seed.md)
- [Fusion 実験: 日本語](docs/fusion_ja.md)
- [Fusion 実験: 英語](docs/fusion.md)
- [CAM weight 可視化: 日本語](docs/cam_weights_ja.md)
- [CAM weight 可視化: 英語](docs/cam_weights.md)
- [Experiment docs index](docs/README.md)

## 環境構築

この checkout では、動作確認時に以下の環境スクリプトを読み込む想定です。

```bash
source ./path.sh
```

`path.sh` は `PYTHONPATH` にリポジトリルートを追加し、
プロジェクト用の仮想環境を有効化します。
このファイルは環境依存のため gitignore 対象です。新規 clone 時は
`path_example.sh` を `path.sh` にコピーし、プレースホルダを埋めてください。

新しく依存関係を入れる場合は、`pyproject.toml` または `requirements.txt` を使います。

```bash
# uv を使う場合
uv sync

# pip を使う場合
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

主な前提:

- Python >= 3.11
- PyTorch 2.8.0 系
- CUDA 対応環境を使う場合は、`pyproject.toml` の PyTorch index 設定と利用 GPU 環境を確認してください。

## ディレクトリ構成

```text
TNSRE2026FIMFusion/
├── config/                          # 実験設定 YAML（各フォルダに比較条件と実行コマンドをまとめた README.md / README_ja.md を配置）
│   ├── action_selection/            # 動作サブセットの ablation 設定
│   ├── complement_ablation/         # 欠損動作の補完手法 ablation の設定
│   ├── fusion_strategy_ablations/   # 単一動作・fusion 戦略の設定
│   └── model_ablation/              # モデル ablation の設定 (モジュール構成違い＋ベースライン、モデル以外の条件は共通)
├── data/                            # 入力データと前処理後データ
│   ├── OriginalData/                # 元データ
│   ├── ComplementedData/            # 時間補完後データ
│   ├── FilteredData/                # フィルタ・特徴量計算後データ
│   ├── ConciseData/                 # 簡略関節セットのデータ
│   └── Info/                        # データリストなどのメタ情報
├── docs/                            # 実験ごとの詳細 README
├── experiments/                     # 実験エントリーポイント（exam_each_seed.py, exam_fusion.py）
├── reports/                         # 結果集計・可視化スクリプト
│   ├── calculates/                  # cross-subject 変数集計
│   ├── ablation_boxplot/            # ablation box/violin plot
│   ├── check_fim_dist/                  # FIM 分布確認
│   └── skeleton_plots/               # 骨格・Attention/GradCAM 可視化
├── results/                         # 実験結果出力先
├── src/                             # experiments/ が import するコアライブラリ
│   ├── data/                        # 骨格データパイプライン
│   │   ├── preprocessing/           # 生データ前処理（order/, utils/）
│   │   ├── dataset/                 # HDF5 データセット生成（utils/）
│   │   └── feeder/                  # PyTorch Dataset/DataLoader + 結果管理
│   ├── model/                       # ネットワーク定義
│   │   ├── gcn/                     # ST-GCN サブネット（components/）
│   │   ├── rnn/                     # LSTM / AGC-LSTM サブネット
│   │   ├── attention/              # attention サブネット + CAM/GradCAM（components/）
│   │   ├── baselines/               # 移植ベースライン（LITEMV/ConvTran）
│   │   └── params/                  # モデルパラメータヘルパ
│   ├── training/                    # Train / Test / Run（experiments/ が使用）
│   ├── plot/                        # 評価指標計算と図生成
│   └── utils/                       # 名称・フォント・混同行列・クリーンアップ
├── path_example.sh                  # path.sh のテンプレート（gitignore 対象、環境依存）
├── pyproject.toml                   # uv 用プロジェクト定義
└── requirements.txt                 # pip 用依存関係リスト
```

`src/`・`experiments/`・`reports/`・`config/` 以下の各フォルダは、ファイルとサブフォルダを
説明する `README.md` / `README_ja.md` を備えている。

### フォルダの役割と依存関係

リポジトリ全体は、生データから図表までの 1 本のパイプラインを構成する。

```text
data/OriginalData
   └─ src/data/preprocessing ──► data/ConciseData
                                     └─ src/data/dataset ──► train.h5 / test.h5
                                                                 └─ src/data/feeder
                                                                        ▼
config/*.yaml ──► experiments/ ──► src/training ◄── src/model
                                        │
                                        ▼
                                    results/ ──► reports/ ──► 図表・サマリ
```

- **`src/data/`** — データパイプライン。`preprocessing/` が `data/OriginalData/` を読み
  込み、骨格点削減済みの `data/ConciseData/` を出力する（ファイル整理は `order/`、数値
  ヘルパは `utils/`）。`dataset/` が `ConciseData` から被験者を考慮した K-fold 分割付きの
  `train.h5` / `test.h5` を生成し、`feeder/` がそれをデータ拡張付き PyTorch
  `Dataset`/`DataLoader` として包む。学習中の結果集計クラスもここに置く。ラベル/関節の
  定義は `src/utils/names.py` を参照する。
- **`src/model/`** — ネットワーク定義。データパイプラインには依存しない。提案エンコーダ
  `stgclsatt.py` は 3 つのサブネット `gcn/` → `rnn/` → `attention/` を合成し、
  Early/Middle/Late Fusion モデルが 1 つ以上のエンコーダを包む。欠損アクション表現は
  `mask_token.py` を共有する。`build_model.py` は `src/training` の両エントリクラスが
  `model.name` を解決するために呼ぶファクトリ（`baselines/` も含む）。
- **`src/training/`** — feeder と model をつなぐ層。`Run` が `build_model.py` 経由で
  モデル(単一エンコーダ、または `fusion.type` で分岐する Early/Middle/Late Fusion)を
  構築し、`Train`/`Test` ループを駆動する。`Test` は `src/plot/calc_indicator.py` で指標を、
  `src/utils/` で混同行列を、`src/model/attention/cam_fusion.py` で GradCAM 重みを計算
  する。
- **`experiments/`** — 実験を起動する唯一のスクリプト群。`config/` の YAML を読み込んで
  feeder を構築し、`src/training` を呼び出して `results/` 以下にすべてを書き出す。
  `exam_each_seed.py` と `exam_fusion.py` はいずれも `Run` を使う。
- **`reports/`** — 事後処理専用。`results/` の出力ファイルにのみ依存し、学習コードには
  依存しない。`calculates/` が fold ごとの `PredictTruelabel.csv` を被験者ごとの一致率に
  集計し、それを `ablation_boxplot/` が比較図に変換する。`skeleton_plots/` は `Test` が
  保存した Attention/GradCAM 重みを描画する。
- **`config/` と `docs/`** — config は実験バッチごとのモデル/fusion/アクション選択を
  定義し、docs は上で参照した実験ごとの詳細 README を収める。

## データ前処理

元の骨格データは `data/OriginalData/` 配下に被験者・計測単位のディレクトリとして配置します。

```text
data/OriginalData/
├── NEDO001-0001_...
├── NEDO001-0002_...
└── ...
```

前処理は現在の `src/data/preprocessing/` 配下のスクリプトを使います。
`orderliness_data.py` はファイル移動や削除を含むため、必要に応じてバックアップを取ってから実行してください。

```bash
# データ整理
python src/data/preprocessing/orderliness_data.py

# 一括パイプライン: 時間補完 → フィルタリング・速度・角度・骨長計算 → 簡略関節セットの作成
python src/data/preprocessing/make_concise_data.py
```

`make_concise_data.py` は `data/OriginalData/` から `data/ConciseData/` までの
前処理を 1 コマンドで実行します。時間補完・フィルタリング・速度/角度/骨長計算は
すべてメモリ上で処理し、中間ディレクトリはディスクに出力しません。
中心化はデータセット生成時（`norm_centering_mode: First`）に適用されるため、
前処理では非中心化データのみを生成します。

実行時に全被験者×動作の状態を分類し、FIM Itemごとの動作欠損分布を
`data/Distributions/` に出力します。骨格品質不良は既存の `Unused/` の証拠だけで判定し、`make_concise_data.py`
自体は生データCSVを `Unused/` に移動しません。詳細は
[docs/missing_motion_ja.md](docs/missing_motion_ja.md) を参照してください。

## 実験実行

現在の checkout には `experiments/exam_each_seed.py` のデフォルト
`config/config.yaml` と、`experiments/exam_fusion.py` のデフォルト
`config/configSomeTask.yaml` は標準配置されていません。
通常は `-c` / `--config` で実在する設定ファイルを明示して実行します。

`config/` 配下の各 ablation フォルダには、比較条件と実行コマンドを網羅した
`README.md` / `README_ja.md` があります:
[`action_selection/`](config/action_selection/README_ja.md)、
[`complement_ablation/`](config/complement_ablation/README_ja.md)、
[`fusion_strategy_ablations/`](config/fusion_strategy_ablations/README_ja.md)、
[`model_ablation/`](config/model_ablation/README_ja.md)。

### 単一動作実験

```bash
python experiments/exam_each_seed.py -c config/fusion_strategy_ablations/single_action.yaml
```

### Fusion 実験

```bash
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

### Model / action selection ablation

モデルの ablation は `config/model_ablation/` にあります。提案モデルの
モジュール構成違い (`stgclsatt.yaml`, `stgcn.yaml` など) とベースライン
(`litemv.yaml`, `convtran.yaml`) を同じ
フォルダに置き、1 つのアブレーションバッチとして実行できます。`model`
セクション以外の設定は全 config で共通です。
動作選択の ablation は `config/action_selection/` にあります。
いずれも `exam_fusion.py -c <config>` で実行します。

### 欠損動作の補完手法 ablation

`config/complement_ablation/` は `config/action_selection/stand_actions.yaml`
を基準に、欠損動作の扱いだけを変えた config を比較します（補完関連キー
以外の設定はすべて共通）。`exam_fusion.py -c <config>` で実行します。完全な
比較表（complete-case・zero padding・padding mask・modality dropout 版を含む）
は [`complement_ablation/README_ja.md`](config/complement_ablation/README_ja.md)
を参照してください。

| Config | 基準からの変更点 | 挙動 |
| --- | --- | --- |
| `complete_case.yaml` | `preprocess.is_fill_actionless: false` | 欠損動作のある被験者を除外（complete-case、`_CompleteCase` サフィックス） |
| `zero_fill.yaml` | なし（ベースライン） | 欠損動作を全ゼロスライスのまま使用（`_ZeroPad` サフィックス） |
| `zero_fill_modality_dropout.yaml` | `zero_fill.yaml` + `trainings.modality_dropout_p: 0.15` | `zero_fill.yaml` に加え、実データセットの平均欠損率と同じ 0.15 で学習時 modality dropout を適用（`_ZeroPad_MD` サフィックス） |
| `pad_mask.yaml` | `fusion.missing_representation: pad_mask` | 各ブランチを存在サンプルのみで順伝搬し、マスク平均で集約（`_Masking` サフィックス） |
| `pad_mask_modality_dropout.yaml` | `pad_mask.yaml` + `trainings.modality_dropout_p: 0.15` | `pad_mask.yaml` に加え、同じ 0.15 の modality dropout を適用（`_Masking_MD` サフィックス） |

### ベースラインモデル

2 ベースラインの PyTorch 移植が `src/model/baselines/` にあります。いずれも
[DeepRehabPile](https://github.com/MSD-IRIMAS/DeepRehabPile) 由来で、
`LITEMV` (Ismail-Fawaz et al., arXiv:2409.02869) と `ConvTran`
(Foumani et al., DMKD 2024、公式実装準拠) です。
`model.name` で選択し、プロジェクト標準の学習プロトコルで学習します
(モデルのハイパーパラメータは各公式実装のデフォルトを踏襲)。

```bash
# Fusion 実験 (fusion.type: late_* のみ。early / middle は非対応)
python experiments/exam_fusion.py -c config/model_ablation/litemv.yaml
python experiments/exam_fusion.py -c config/model_ablation/convtran.yaml
```

GradCAM 出力 (`model.cam_type: gradcam`) には対応していません。

## 主な設定項目

| Key | 内容 |
| --- | --- |
| `model.name` | 使用するモデル名。`STGCN`, `STGCLSAtt`, `STGCAGLS` など。ベースライン: `LITEMV`, `ConvTran`。 |
| `fusion.type` | 統合戦略。`early_chn_stack`, `early_exp_graph`, `middle_shallow`, `middle_deep`, `late_feature`, `late_probability` のいずれか。単一動作設定では使いません。 |
| `fusion.missing_representation` | 欠損動作の表現方法。`zero`（既定）、`exclude`（masked mean）、`token`（学習可能マスクトークン）。詳細は `docs/fusion_ja.md` を参照。 |
| `fusion.mask_token_granularity` | `token` 時のトークン粒度。`shared`（既定）または `per_action`。 |
| `trainings.modality_dropout_p` | 学習時 modality dropout の確率（既定 `0.0` = 無効）。 |
| `action_task_indices` | 単一動作実験で使う動作 index。`src/utils/names.py` の `Tasks()` に対応します。 |
| `combined_action_indices` | Fusion 実験で統合する動作 index。 |
| `fim_item_indices` | 予測対象の FIM 項目 index。 |
| `classes` | 分類対象の FIM スコア。内部 label は 0-based に変換されます。 |
| `seed_fix_mode` | `split`, `model`, `change`。データ分割 seed とモデル学習 seed の固定方法を決めます。 |
| `preprocess.*` | フレーム長、中心化、動的フレーム、短系列補完、動作欠損補完など。 |
| `trainings.*` | epoch、optimizer、scheduler、batch size、weighted sampler、後処理削除設定など。 |
| `wandb.use_wandb` | Weights & Biases に記録するかどうか。 |
| `output_weights` | Attention / Grad-CAM 重みを保存するかどうか。 |

## 結果出力

実験結果は `results/<experiment_name>/` に保存されます。

単一動作実験の代表的な構造:

```text
results/<experiment_name>/
├── situations.yaml
├── Accuracies.csv
├── results.csv
└── <Action>_to_<ItemIndex>-<FIMItem>/
    ├── Accuracies.csv
    └── Stratified_Seed<seed>/
        ├── Accuracies.csv
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

Fusion 実験では、FIM 項目ディレクトリの下に `Stratified_Seed<seed>/Fold<fold>/`
が作られます。

主な CSV:

| ファイル | 内容 |
| --- | --- |
| `situations.yaml` | 実行時 config のコピー。 |
| `Accuracies.csv` | seed / Fold / FIM 項目ごとの集約指標。`Accuracy`・`Balanced_Accuracy`・クラス別の `Class<c>_Accuracy` [%]・`F1Score` / `MCC` / `Precision` / `Recall` / `AUCROC`(0-1)を含みます。すべて `wandb.use_wandb` の設定に関係なく算出されます。 |
| `results.csv` | `Seed == "Mean"` の行を抜き出した比較用サマリ。 |
| `PredictTruelabel.csv` | Fold のサンプル単位推論結果。`Directory`, `Predicted`, `True`, `RealScore`, `Prob_*` を含みます。 |
| `ELLA_ep<N>.csv` | epoch ごとの学習ログ。 |
| `trainDir.csv`, `testDir.csv` | split に含まれる `Directory`, `FrameStart`, `RealScore` などのメタ情報。 |

### 過去の実験結果への指標バックフィル

クラス別正答率・詳細指標が wandb と独立に記録されるようになる前の実験ディレクトリでは、
`Class<c>_Accuracy` 列が空で `F1Score` / `MCC` / `Precision` / `Recall` / `AUCROC` 列が
存在しません。Fold ごとの `PredictTruelabel.csv` から再計算して埋めるには以下を実行します。

```bash
python -m src.backfill_metrics results/<experiment_name>            # 欠損列を補完
python -m src.backfill_metrics results/<experiment_name> --dry-run  # 比較のみ(書き込みなし)
```

既に値が入っている列は `--force` を付けない限りスキップされます。集約は実験スクリプトと
同じ「Fold 平均 → seed 平均」方式です。

## Cross-subject 集計

Fold ごとの `PredictTruelabel.csv` を被験者単位の一致率へ変換します。

```bash
# 1 つ以上の実験 root を明示
python reports/calculates/calc_cross_subject_variables.py results/<experiment_name>

# 親ディレクトリ配下の実験 root を自動探索
python reports/calculates/calc_cross_subject_variables.py --scan results/<parent_dir>
```

主な出力:

| 出力 | 保存先 |
| --- | --- |
| `PredictTruelabel_all_folds.csv` | 各 `Stratified_Seed<seed>/` 直下 |
| `CrossSubjectVariables.csv` | 各 FIM 項目または `<Action>_to_<Item>` ディレクトリ直下 |
| `CrossSubjectVariables_all_actions.csv` | 動作別ディレクトリを含む item ディレクトリ直下 |
| `CrossSubjectVariables_action_to_item/<Item>.csv` | 実験 root 直下 |
| `CrossSubjectVariables_all_items.csv` | 実験 root 直下 |

## Ablation 可視化

cross-subject 集計済みの複数条件を比較する場合は、各条件ディレクトリを 1 つの
ablation root の下に配置して実行します。`order.csv` がある場合は、その順序で条件を描画します。

```bash
python reports/ablation_boxplot/ablation_cross_subject_variables.py results/<ablation_root>
```

出力は `results/<ablation_root>/AblationPlotsCrossSubject/` に保存されます。
現在の出力契約では、通常の item 条件と SingleAction の `MeanAction` / `MaxAction`
を同じファイル群に統合します。

```text
AblationPlotsCrossSubject/
├── AverageMatchRate_AllItems.csv
├── AverageMatchRate_AllItems_Stats.csv
├── AverageMatchRate_AllItems.png
├── AverageMatchRate_AllItems.pdf
├── AverageMatchRate_AllItems.svg
├── MatchRate_<Item>.csv
├── MatchRate_<Item>_Stats.csv
├── MatchRate_<Item>.png
├── MatchRate_<Item>.pdf
└── MatchRate_<Item>.svg
```

`MaxAction` は `Summary.csv` の `Max_Balanced_Acc_Action` を優先し、必要に応じて
`results.csv` / `Accuracies.csv` の balanced accuracy 系列から選択します。
対応している列名は `BalancedAccuracy`, `Balanced_Accuracy`, `Balanced Accuracy` です。

## Weights & Biases

`wandb.use_wandb: true` にすると、Fold 単位の学習・評価が Weights & Biases に記録されます。
`wandb.entity` に entity または team 名を設定してください。

```yaml
wandb:
  use_wandb: true
  entity: "your-team-name"
```

## Citation

このコードを研究で利用する場合は、以下を引用してください。

```bibtex
@article{author2026multifusion,
  title     = {Skeleton-Based Multi-Action Fusion Framework for Binary Classification of FIM Score from Simplified Rehabilitation Actions},
  author    = {Jun Masaki, Ariaki Higashi, Naoko Shinagawa, Kazuhiko Hirata, Yuichi Kurita, and Akira Furui},
  journal   = {IEEE Transactions on Neural Systems and Rehabilitation Engineering(TNSRE)},
  year      = {2026},
  volume    = {},
  number    = {},
  pages     = {},
  doi       = {10.1109/TNSRE.2026.3744185},
  memo      = {IEEE Trans. Neural Syst. Rehabil. Eng. (Abbreviation of journal name)}
}
```

## License

本プロジェクトは研究目的で提供されています。ライセンス情報については著者にお問い合わせください。
