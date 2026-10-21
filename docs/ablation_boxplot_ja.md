# アブレーションボックスプロットパイプライン

このドキュメントでは `reports/ablation_boxplot/` 内のアブレーション比較プロットスクリプトと、
バッチラッパー `reports/ablation_process.py` について説明します。

## 概要

これらのスクリプトは複数のアブレーション条件をボックスプロット・棒グラフ・バイオリンプロットで
並べて比較します。各アブレーション条件は、後処理パイプラインによって処理済みの実験結果ディレクトリに対応します。

| スクリプト | エントリ関数 | 入力 | 出力ディレクトリ |
| --- | --- | --- | --- |
| `ablation_sum_plots.py` | `ablationSumPlots()` | `Accuracies.csv`, `results.csv` | `<root>/AblationPlots/` |
| `ablation_plots_each_class.py` | `ablationPlotsEachClass()` | `Accuracies.csv`, `Summary.csv` | `<root>/AblationPlotsEachClass/` |
| `ablation_cross_subject_variables.py` | `ablationCrossSubjectVariables()` | `CrossSubjectVariables_all_items.csv`, `CrossSubjectVariables_action_to_item/` | `<root>/AblationPlotsCrossSubject/` |
| `ablation_process.py` *（ラッパー）* | — | — | — |

参照：

- [実験後処理: 英語版](results_process.md)
- [実験後処理: 日本語版](results_process_ja.md)

---

## 前提条件

### 後処理パイプライン

各条件ディレクトリには `src/results_process.py`（または `src/results_process_batch.py`）の出力が必要です：

- `Accuracies.csv` — シード・フォールドごとの集計メトリクス
- `Summary.csv` — FIMアイテムごとの統計（`Max_Balanced_Acc_Action` を含む）
- `results.csv` — `Accuracies.csv` から抽出したサマリ行

### 被験者横断集計（`ablation_cross_subject_variables.py` のみ）

`ablation_cross_subject_variables.py` を実行する前に、全実験に対して
`reports/calculates/calc_cross_subject_variables.py` を実行してください。
これにより以下のファイルが生成されます：

- `CrossSubjectVariables_all_items.csv` — 全FIMアイテムにわたる被験者ごとの一致率
- `CrossSubjectVariables_action_to_item/<Item>.csv` — 動作ごとの被験者別一致率（シングルアクション実験のみ）

---

## ディレクトリ構成と `order.csv`

**アブレーションルートディレクトリ**を1つ作成し、比較対象の実験結果ディレクトリをその中に移動します。
各条件ディレクトリ名は `order.csv` のプレフィックスで始まり、その後ろに `_` で続く名前にする必要があります：

```text
<ablation_root>/
├── order.csv                              ← 手動で作成; 1行1プレフィックス
├── <prefix_A>_<rest_of_name>/             ← 条件 A
│   ├── Accuracies.csv
│   ├── Summary.csv
│   ├── results.csv
│   └── CrossSubjectVariables_all_items.csv     ← ablationCrossSubjectVariables を使う場合
│   └── CrossSubjectVariables_action_to_item/   ← ablationCrossSubjectVariables を使う場合
├── <prefix_B>_<rest_of_name>/             ← 条件 B
│   └── ...
└── <prefix_C>_<rest_of_name>/             ← 条件 C
    └── ...
```

`order.csv` はヘッダなしの1列CSVです。各行がプロットの表示順（左から右）のプレフィックスになります：

```csv
prefix_A
prefix_B
prefix_C
```

スクリプトは `glob("<ablation_root>/<prefix>*")` で条件ディレクトリを照合するため、
プレフィックスはディレクトリ名の先頭として一意である必要があります。
プレフィックスの後に `_` を区切り文字として使用するのがプロジェクトの規約です。

> **注意** — ディレクトリ名に `"AblationPlots"` を含むものは `src/results_process_batch.py` で
> 自動的にスキップされるため、アブレーションルートを `results_process_batch.py` 処理対象の
> 親ディレクトリ内に置いても処理ループは発生しません。

---

## ワークフロー

後処理を行う**前**に対象の実験結果ディレクトリをアブレーションルートに移動し、
`src/results_process_batch.py` でまとめて後処理します。

```
1. <ablation_root>/ を作成
2. 対象の実験ディレクトリを移動し、<prefix>_<rest> の名前にする
3. order.csv を作成
4. python -m src.results_process_batch
   # rootDir: <ablation_root>
   # (AblationPlots* ディレクトリは自動スキップされる)
5. python reports/calculates/calc_cross_subject_variables.py --scan <ablation_root>
   # ablation_cross_subject_variables.py を使う場合のみ必要
6. python reports/ablation_process.py
   # rootDir: <ablation_root>
   # AblationPlots/ と AblationPlotsEachClass/ を生成
7. python reports/ablation_boxplot/ablation_cross_subject_variables.py <ablation_root>
   # AblationPlotsCrossSubject/ を生成
```

---

## 各スクリプトの処理詳細

### `ablation_sum_plots.py` — Balanced-Accuracy プロット

**実行方法:**

```bash
python reports/ablation_process.py      # 推奨（ablationPlotsEachClass も一緒に実行）
# または単独実行
python reports/ablation_boxplot/ablation_sum_plots.py
```

アブレーションルートディレクトリの入力を求めます。

**処理手順:**

1. `order.csv` を読み込んで条件プレフィックスとプロット順を取得します。
2. `FIMitems()[8:-1]` の各FIMアイテムについて、各条件を処理します：
   - 条件ディレクトリから `Accuracies.csv` を読み込みます。
   - **シングルアクション実験**（`FromAction` 列が存在する場合）：
     - **MeanAction** — シードごとに全動作の `Balanced_Accuracy` を平均します。
       プロットラベル: `<prefix>_MeanAction`。
     - **MaxAction** — `results.csv` を読み込み、そのアイテムで `Balanced_Accuracy` が
       最大の動作を特定し、`Accuracies.csv` をその動作でフィルタします。
       プロットラベル: `<prefix>_MaxAction`。
   - **Fusion 実験**（`FromAction` 列なし）：`Balanced_Accuracy` をそのまま使用します。
     プロットラベル: `<prefix>`。
3. アイテムごとに以下を生成します：
   - x = 条件、y = Balanced Accuracy のボックスプロット（各ボックスの上に平均値を赤字で表示）。
   - シードごとの平均 ± 標準偏差の棒グラフ（平均値を各棒の上に表示）。
4. 全アイテム処理後、(Seed, 条件) ごとにアイテム方向で Balanced Accuracy を平均し、
   全アイテム横断のボックスプロットと棒グラフを生成します。

**出力ファイル**（`<root>/AblationPlots/` 以下）：

| ファイル | 内容 |
| --- | --- |
| `result_all_<item>.csv` | ボックスプロット用の生のシード別 Balanced-Accuracy データ |
| `Balanced_Accuracy_<item>.{png,pdf,svg}` | アイテム別ボックスプロット |
| `Balanced_Accuracy_<item>_BarPlot.{png,pdf,svg}` | アイテム別棒グラフ（平均 ± 標準偏差） |
| `Balanced_Accuracy_<item>_Stats.csv` | 条件ごとの記述統計（平均・標準偏差） |
| `result_all_CrossActions.csv` | 全アイテム平均のシード別データ |
| `Balanced_Accuracy_CrossActions.{png,pdf,svg}` | 全アイテム横断ボックスプロット |
| `Balanced_Accuracy_CrossActions_BarPlot.{png,pdf,svg}` | 全アイテム横断棒グラフ（平均 ± 標準偏差） |
| `Balanced_Accuracy_CrossActions_Stats.csv` | 全アイテム横断の記述統計 |

---

### `ablation_plots_each_class.py` — クラス別精度プロット

**実行方法:**

```bash
python reports/ablation_process.py      # 推奨（ablationSumPlots も一緒に実行）
# または単独実行
python reports/ablation_boxplot/ablation_plots_each_class.py
```

アブレーションルートディレクトリの入力を求めます。

**処理手順:**

1. `order.csv` を読み込んで条件プレフィックスとプロット順を取得します。
2. `FIMitems()[8:-1]` の各FIMアイテムについて、各条件を処理します：
   - 条件ディレクトリから `Accuracies.csv` を読み込みます。
   - **シングルアクション実験**（`FromAction` 列が存在する場合）：
     - **MaxAction** — `Summary.csv` を読み込み `Max_Balanced_Acc_Action` を参照し、
       `Accuracies.csv` をその動作でフィルタして Class 系列列を抽出します。
       プロットラベル: `<prefix>_MaxAction`。
     - **MeanAction** — シードごとに全動作のクラス別精度列を平均します。
       プロットラベル: `<prefix>_MeanAction`。
   - **Fusion 実験**（`FromAction` 列なし）：クラス別精度列をそのまま抽出します。
     プロットラベル: `<prefix>`。
   - `"Class"` を含む列名がクラス別精度列として扱われ、ロング形式に変換して x 軸の
     `situation` グループとして使用されます。
3. x = クラスラベル、hue = 条件の分割ボックスプロットを描画します。
   凡例はプロット右外に配置されます。

**出力ファイル**（`<root>/AblationPlotsEachClass/` 以下）：

| ファイル | 内容 |
| --- | --- |
| `result_class-wise_<item>.csv` | ロング形式のシード別クラス精度データ |
| `Class-wise_Accuracy_<item>.{png,pdf,svg}` | グループ別ボックスプロット（x = クラス、hue = 条件） |

---

### `ablation_cross_subject_variables.py` — 被験者別一致率プロット

**実行方法:**

```bash
python reports/ablation_boxplot/ablation_cross_subject_variables.py <ablation_root>
# または対話的
python reports/ablation_boxplot/ablation_cross_subject_variables.py
```

**処理手順:**

1. アブレーションルートに `order.csv` があればそれを読み込みます。なければ
   アルファベット順で条件ディレクトリを自動探索します（`AblationPlots*` は除外）。
2. 各条件から以下を読み込みます：
   - `CrossSubjectVariables_all_items.csv` — FIMアイテムごとの被験者別一致率。
   - `CrossSubjectVariables_action_to_item/<Item>.csv` — 動作別の被験者別一致率
     （シングルアクション実験のみ存在）。
3. **全体平均プロット**（`AverageMatchRate_AllItems`）を生成します：
   - 被験者ごとに全FIMアイテム列の一致率を平均します。
   - シングルアクション条件では `<prefix>_MeanAction`（全動作の `AverageMatchRate` の
     被験者別値）と `<prefix>_MaxAction`（アイテムごとの最良動作の一致率をアイテム平均）
     のバリアントも追加します。
   - 最良動作は `find_best_action()` で決定します：`Summary.csv` を最初に確認し、
     次に `results.csv`・`Accuracies.csv` の順にフォールバックします。
4. 全条件のCSVに存在するFIMアイテム列ごとに**アイテム別プロット**（`MatchRate_<Item>`）を生成します。
   - シングルアクション条件では `CrossSubjectVariables_action_to_item/<Item>.csv` から
     `MeanAction` と `MaxAction` のバリアントも追加します。

各プロットの1データ点は被験者1人の一致率（シード集計値ではない）を表します。
バイオリン + ボックスオーバーレイで描画し、平均値を赤字で注記します。

**出力ファイル**（`<root>/AblationPlotsCrossSubject/` 以下）：

| ファイル | 内容 |
| --- | --- |
| `AverageMatchRate_AllItems.csv` | 全体平均プロット用の被験者別データ |
| `AverageMatchRate_AllItems_Stats.csv` | 条件ごとの記述統計 |
| `AverageMatchRate_AllItems.{png,pdf,svg}` | 全FIMアイテム横断のバイオリン + ボックスプロット |
| `MatchRate_<Item>.csv` | アイテム別プロット用の被験者別データ |
| `MatchRate_<Item>_Stats.csv` | アイテム別の条件ごとの記述統計 |
| `MatchRate_<Item>.{png,pdf,svg}` | アイテム別バイオリン + ボックスプロット |

---

### `ablation_process.py` — ラッパー

`ablationSumPlots` と `ablationPlotsEachClass` を同じルートディレクトリに対して順番に実行します。
`ablationCrossSubjectVariables` は含みません。

```bash
python reports/ablation_process.py
# rootDir を入力: <ablation_root>
```

---

## 出力ディレクトリ一覧

| 出力ディレクトリ | 生成スクリプト | 内容 |
| --- | --- | --- |
| `<root>/AblationPlots/` | `ablation_sum_plots.py` | Balanced-Accuracy のアイテム別・全アイテム横断ボックスプロット + 棒グラフ |
| `<root>/AblationPlotsEachClass/` | `ablation_plots_each_class.py` | クラス別精度のアイテム別ボックスプロット |
| `<root>/AblationPlotsCrossSubject/` | `ablation_cross_subject_variables.py` | 被験者別一致率のバイオリン + ボックスプロット |

全プロットファイルは `.png`・`.pdf`・`.svg` の3形式で保存されます。

---

## 注意事項

- **FIMアイテムの対象範囲** — `ablationSumPlots` と `ablationPlotsEachClass` は
  `FIMitems()[8:-1]`（定義済みサブセット）のみを処理します。`ablationCrossSubjectVariables`
  はCSVに存在するアイテム列を全て対象とします。
- **プレフィックス照合** — 条件ディレクトリは `glob("<prefix>*")` で照合されます。
  同じプレフィックスで始まるディレクトリが複数ある場合はソート後の最初の一致が使用されます。
  プレフィックスは一意になるよう設定してください。
- **カラーパレット** — 最大15色の基本色が使用可能です。条件が15を超える場合は先頭から色が再利用されます。
- **MaxAction の参照元はスクリプトによって異なります** — `ablationSumPlots` は `results.csv`、
  `ablationPlotsEachClass` は `Summary.csv["Max_Balanced_Acc_Action"]`、
  `ablationCrossSubjectVariables` は `Summary.csv` → `results.csv` → `Accuracies.csv`
  の順でフォールバックします。
