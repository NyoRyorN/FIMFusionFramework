# 実験後処理パイプライン

本ドキュメントでは、実験後処理スクリプト
`src/results_process.py` および `src/results_process_batch.py` について説明します。

## 概要

学習完了後、実験ルートディレクトリには生の CSV ファイルとチェックポイントファイルが残ります。
後処理パイプラインは以下の 4 ステップで構成されます。

1. **DeleteAfter** — `.pth` チェックポイントファイルを削除してディスクを解放する
2. **SummaryOutPuts** — Seed/Fold ごとの結果をサマリ CSV に集約する
3. **all_calc_indicator** *(mode C のみ)* — 生の予測 CSV から追加の分類指標を計算する
4. **BoxPlotsEachClass** — FIM 項目・クラスごとの精度をボックスプロット・バープロットで可視化する

---

## スクリプト

### `src/results_process.py` — 単一ディレクトリ処理

1 つの実験結果ディレクトリに対してパイプライン全体を実行します。

**対話的実行:**

```bash
python -m src.results_process
```

入力プロンプト:

| プロンプト | 説明 |
|---|---|
| `rootPath` | 実験結果ディレクトリのパス |
| `mode` | `N`（Normal: 3 クラス、スコア 1–5 vs 6 vs 7）、`A`（Assistance: 2 クラス、1–5 vs 6–7）、`C`（Complete independence: 2 クラス、1–6 vs 7）。学習時のクラス設定に合わせて選択する。 |
| `flag_deleteAfter` | `y` で `.pth` ファイルを削除、`n` でスキップ |
| `flag_summaryOutputs` | `y` で `SummaryOutPuts` を実行（mode C では `all_calc_indicator` も実行） |
| `flag_detailPlots` | `y` で `BoxPlotsEachClass` を実行 |
| Y 軸範囲 | `y` で Y 軸を 0–100 固定、`n` でデータに合わせて自動スケール |

**スクリプトからの呼び出し:**

```python
from src.results_process import ResultsProcess

ResultsProcess(
    rootPath="/path/to/results/experiment_name",
    mode="C",
    flag_deleteAfter="y",
    flag_summaryOutputs="y",
    flag_detailPlots="y",
    range_detailPlots="y",   # "y" = 0-100 固定、それ以外 = 自動スケール
)
```

---

### `src/results_process_batch.py` — 複数ディレクトリの一括処理

指定したルートディレクトリ直下のすべてのサブディレクトリに対して
`ResultsProcess` を順に適用します。複数の実験結果を 1 つの親ディレクトリに
まとめてから比較したい場合に便利です。

**実行:**

```bash
python -m src.results_process_batch
```

入力プロンプト:

| プロンプト | 説明 |
|---|---|
| `rootDir` | 実験サブディレクトリを含む親ディレクトリのパス |
| `mode` | `N` / `A` / `C` — 学習時のクラス設定に合わせて選択する |
| `flag_deleteAfter` | `y` / `n` |
| `flag_summaryOutputs` | `y` / `n` |
| `flag_detailPlots` | `y` / `n` |

> **注意:** ディレクトリ名に `"AblationPlots"` を含むディレクトリは自動的にスキップされます。
> プロットの Y 軸はデフォルトで自動スケールになります。

**一括処理の典型的なワークフロー:**

実験完了後、結果ディレクトリを自作の親ディレクトリへ移動することで
`results_process_batch.py` による一括処理が可能になります。

```
my_experiments/
├── experiment_A/      ← 処理対象
├── experiment_B/      ← 処理対象
└── AblationPlots/     ← 自動スキップ
```

```bash
# 実験結果を親ディレクトリへ移動
mv results/experiment_A /path/to/my_experiments/
mv results/experiment_B /path/to/my_experiments/

# 一括処理を実行
python -m src.results_process_batch
# rootDir を入力: /path/to/my_experiments
```

---

## 各処理ステップの詳細

### Step 1 — DeleteAfter (`src/utils/delete_after.py`)

`os.walk` で `rootPath` 以下を再帰的に走査し、`.pth` 拡張子を持つファイルを
すべて削除します。削除したファイルのパスは標準出力に出力されます。`.pth` ファイルが
1 件も見つからなかった場合もその旨を出力して正常終了します。学習後のモデル
チェックポイントは大きなディスク容量を占有するため、不要であれば削除を推奨します。

---

### Step 2 — SummaryOutPuts (`src/plot/summary_outputs.py`)

`rootPath` 直下の `Accuracies.csv` を読み込み、`Seed == "Mean"` の行を除外した
上で（Seed・Fold 単位の生データのみを対象として）統計量を計算します。

#### 単一動作実験（`FromAction` 列あり）

**クロスアイテムバープロット**（`PlotCrossItemPlots`）:

1. (Action, Seed) ごとに全 FIM 項目の `Balanced_Accuracy` を平均する
   （= Seed 単位のクロスアイテム平均）→ `CrossItem_Balanced_Accuracy_Data.csv` として保存。
2. その Seed 単位の平均値をアクションごとにまとめ、Seed 方向の平均と標準偏差を計算する
   → `CrossItem_Balanced_Accuracy_Stats.csv` として保存。
3. バープロット（アクション別の平均 ± 標準偏差、全体平均を赤い点線で表示）を
   `plotTasks()` が定義する標準アクション順で描画する
   → `CrossItem_Balanced_Accuracy_BarPlot_MeanStd.{png,pdf,svg}` として保存。

**FIM 項目ごとの統計**（`Summary.csv` に出力）:

各 FIM 項目に対して以下の列が算出されます。

| 列名 | 算出方法 |
|---|---|
| `Accuracy_Mean` | 当該項目の全行（全 Seed × 全アクション）の平均 |
| `Accuracy_Std` | Seed ごとにアクション方向で平均してから、その Seed 平均間の標準偏差を計算（`CalcAllStd`） |
| `Acc_Max_Action` | 当該項目で Accuracy 平均が最大のアクション |
| `Acc_Max_Accuracy` | そのアクションの Accuracy 平均値 |
| `Acc_Max_Std` | そのアクションの Accuracy の Seed 間標準偏差 |
| `Balanced_Accuracy_Mean` | Accuracy と同じ方法で Balanced Accuracy を算出 |
| `Balanced_Accuracy_Std` | 同上 |
| `Max_Balanced_Acc_Action` | Balanced Accuracy 平均が最大のアクション |
| `Max_Balanced_Acc_Accuracy` | そのアクションの Balanced Accuracy 平均値 |
| `Max_Balanced_Acc_Std` | そのアクションの Balanced Accuracy の Seed 間標準偏差 |

最終行（`FIMItem = "All"`）は全項目を横断した統計です。"Max" 列については、
各項目の Best action を使った (項目, Best action) の組み合わせのスコアを
Seed ごとに項目方向で平均してから、Seed 間の平均・標準偏差を計算します
（`CalcAllMaxActionStd`）。

#### 複数動作 Fusion 実験（`FromAction` 列なし）

FIM 項目ごとに、`Accuracies.csv` 内の全数値列（`Seed`・`ToFIM` 以外）の
平均と標準偏差を算出します。"All" 行の標準偏差は、単一動作実験と同様に
Seed ごとの平均を先に計算してから Seed 間の標準偏差を求める方式（`CalcAllStd`）
を使います。結果は `Summary.csv` として保存されます。

---

### Step 3 — all_calc_indicator (`src/plot/calc_indicator.py`) — mode C のみ

mode `C`（Complete independence、2 クラス: スコア 1–6 vs 7）のときのみ実行されます。
各 Fold の `PredictTruelabel.csv` から直接分類指標を計算し、複数の集約レベルで
CSV を出力します。

**ディレクトリ探索パターン:**

```
rootPath/
└── <NN>-<Item>                          → action = "All"、item = <Item>
    または <NN>-<Action>_to_<NN>-<Item>  → action = <Action>、item = <Item>
    └── Stratified_Seed<seed>/
        └── Fold<fold>/
            └── PredictTruelabel.csv     ← 読み込み元
```

**Fold ごとに計算される指標**（`calc_indicator`）:

| 指標 | 説明 |
|---|---|
| `Accuracy` | 全体の分類正解率 |
| `BalancedAccuracy` | バランス精度（クラスごとの再現率の平均） |
| `AccClass6` | FIM スコア ≤ 6（クラス 1–6）のサンプルのみの正解率 |
| `AccClass7` | FIM スコア > 6（クラス 7）のサンプルのみの正解率 |
| `F1Score` | 2 値 F1 スコア（正例クラス = 0、すなわちクラス 1–6） |
| `MCC` | マシューズ相関係数 |
| `Precision` | 2 値適合率（正例クラス = 0） |
| `AUCROC` | ROC 曲線下面積（`Prob_0` 列＝クラス 1–6 の確率を使用） |

**集約パイプライン:**

1. 全 Fold の生指標を `Indicator_Lists.csv` として保存。
2. `aggregate_indicators_each_fold`: (Action, FIMItem, Seed) でグループ化し、
   Fold 方向の平均を計算 → `Aggregated_Indicator_Lists.csv` として保存。
3. `summary_indicator` がアクション数に応じて 2 通りの集約を行います。

   **アクションが 1 つ（`action = "All"` のみ）の場合**:
   - FIMItem ごとに Seed 方向の平均と標準偏差を計算。
   - `MeanOfItems` 行: Seed ごとに全項目を平均してから、その Seed 平均間の
     平均・標準偏差を計算。
   - `Summary_Indicator_Lists.csv` に保存。`Each_Summary_Indicator_Lists.csv` は生成されない。

   **アクションが複数の場合**:
   - (Action, FIMItem) でグループ化 → Seed 方向の平均・標準偏差
     → `Each_Summary_Indicator_Lists.csv`。
   - 各 FIM 項目で `BalancedAccuracy_Mean` が最大のアクションを Best action として選択。
   - `BestOfActions` 行: Seed ごとに、各項目の Best action スコアを項目方向で平均し、
     その Seed 平均間の平均・標準偏差を計算。
   - `MeanOfActions` 行: 各 (FIMItem, Seed) でアクション方向の平均を先に計算してから
     Seed 方向で集約。`MeanOfItems` 行も同じ方法で計算。
   - 全行を統合して `Summary_Indicator_Lists.csv` に保存。

---

### Step 4 — BoxPlotsEachClass (`src/plot/box_plots_each_class.py`)

`Accuracies.csv` を読み込み、モードに応じてクラス列をリネームしてから、
`rootPath/results/` に可視化結果を保存します。

**モードとクラスラベルの対応:**

| モード | 分類設定 | クラス列のリネーム先 | ボックスの色 |
|---|---|---|---|
| `N`（Normal） | 3 クラス: スコア 1–5 vs 6 vs 7 | `Class:1-5`、`Class:6`、`Class:7` | lightcoral、Yellow、Lime |
| `A`（Assistance） | 2 クラス: スコア 1–5 vs 6–7 | `Class:1-5`、`Class:6-7` | lightcoral、Yellow、Lime |
| `C`（Complete independence） | 2 クラス: スコア 1–6 vs 7 | `Class:1-6`、`Class:7` | Yellow、Lime |

#### 単一動作実験（`FromAction` 列あり）

FIM 項目ごとに以下の 3 種類のプロットを生成します。

1. **Balanced Accuracy ボックスプロット**（`{item}.{png,pdf,svg}`）:
   x 軸 = アクション（`plotTasks()` の標準順で並び替え）、y 軸 = `Balanced_Accuracy`。
   平均値を赤い `+` マーカー、中央値を黒い点線で表示し、ひげ・外れ値も描画します。

2. **Balanced Accuracy バープロット**（`{item}_BarPlot_MeanStd.{png,pdf,svg}`）:
   アクションごとに Seed 方向の平均 ± 標準偏差を計算してバーとエラーバーで表示。
   各バーの上に平均値（小数点以下 2 桁）を注記し、全体平均を赤い点線で表示します。
   Y 軸はデータ範囲に合わせて自動スケールされます。

3. **クラスワイズ正解率ボックスプロット**（`{item}_ClassWiseAccuracies.{png,pdf,svg}`）:
   クラス列をロング形式に変換し、x 軸 = アクション、hue = クラスの
   グループドボックスプロットを描画します。Y 軸は `range_detailPlots == "y"` の
   場合は 0–100 固定、それ以外はデータの最小・最大値を 5 点刻みに丸めた範囲で自動スケールします。

#### 複数動作 Fusion 実験（`FromAction` 列なし）

全 FIM 項目を対象に以下の 2 種類のプロットを生成します。

1. **Balanced Accuracy ボックスプロット**（`EachItemBalancedAccuracies.{png,pdf,svg}`）:
   x 軸 = FIM 項目（`ToFIM`）、y 軸 = `Balanced_Accuracy`。スタイルは単一動作実験と同じです。

2. **クラスワイズ正解率ボックスプロット**（`All_ClassWiseAccuracies.{png,pdf,svg}`）:
   x 軸 = FIM 項目、hue = クラス。クラスリネームと Y 軸ロジックは単一動作実験と同じです。

すべてのプロットは `.png`・`.pdf`・`.svg` の 3 形式で保存されます。

---

## 出力ファイル一覧

すべての出力は実験結果ディレクトリ（`rootPath`）からの相対パスで保存されます。

### 常に生成（対応フラグが `y` のとき）

| ファイル | 保存先 | 内容 |
|---|---|---|
| *（`.pth` ファイル削除）* | `rootPath/**/*.pth` | モデルチェックポイントの削除 |
| `Summary.csv` | `rootPath/` | 項目別の精度統計（平均・標準偏差・Best action） |

### `FromAction` がある場合のみ生成（単一動作実験）

| ファイル | 保存先 | 内容 |
|---|---|---|
| `CrossItem_Balanced_Accuracy_Data.csv` | `rootPath/results/` | Seed・アクションごとのクロスアイテム平均 Balanced Accuracy |
| `CrossItem_Balanced_Accuracy_Stats.csv` | `rootPath/results/` | アクション別のクロスアイテム Balanced Accuracy（平均 ± 標準偏差） |
| `CrossItem_Balanced_Accuracy_BarPlot_MeanStd.{png,pdf,svg}` | `rootPath/results/` | クロスアイテム Balanced Accuracy のバープロット |
| `{item}.{png,pdf,svg}` | `rootPath/results/` | FIM 項目ごとの Balanced Accuracy アクション別ボックスプロット |
| `{item}_BarPlot_MeanStd.{png,pdf,svg}` | `rootPath/results/` | FIM 項目ごとの Balanced Accuracy アクション別バープロット（平均 ± 標準偏差） |
| `{item}_ClassWiseAccuracies.{png,pdf,svg}` | `rootPath/results/` | FIM 項目ごとのクラスワイズ正解率アクション別ボックスプロット |

### 複数動作 Fusion 実験のみ生成（`FromAction` なし）

| ファイル | 保存先 | 内容 |
|---|---|---|
| `EachItemBalancedAccuracies.{png,pdf,svg}` | `rootPath/results/` | 全 FIM 項目の Balanced Accuracy ボックスプロット |
| `All_ClassWiseAccuracies.{png,pdf,svg}` | `rootPath/results/` | 全 FIM 項目のクラスワイズ正解率ボックスプロット |

### mode C のみ生成（`all_calc_indicator`）

| ファイル | 保存先 | 内容 |
|---|---|---|
| `Indicator_Lists.csv` | `rootPath/` | Fold ごとの生指標（Action・FIMItem・Seed・Fold・各指標） |
| `Aggregated_Indicator_Lists.csv` | `rootPath/` | Fold 集約済み指標（Fold 方向の平均、Seed 単位） |
| `Summary_Indicator_Lists.csv` | `rootPath/` | Seed 集約サマリ（平均 ± 標準偏差）、MeanOfActions / BestOfActions 行を含む |
| `Each_Summary_Indicator_Lists.csv` | `rootPath/` | アクション × FIM 項目ごとの指標サマリ（Seed 方向の平均 ± 標準偏差） |

### `Summary_Indicator_Lists.csv` の行タイプ

| `BestAction` 値 | 説明 |
|---|---|
| `MeanOfActions` | 全アクションで平均した後に Seed 方向で集約した指標 |
| `<action_name>` | FIM 項目ごとの Best action（Balanced Accuracy 平均が最大のアクション） |
| `BestOfActions` | 各項目の Best action を使ったクロスアイテム平均 |
