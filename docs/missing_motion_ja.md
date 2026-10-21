# 動作欠損分布の集計

`src/data/preprocessing/make_concise_data.py` は、前処理の実行中に各被験者×各動作（11タスク）の状態を分類し、FIM Itemごと・FIMスコア（1-7）ごとの動作欠損分布を `data/Distributions/` に出力します。骨格品質不良は、既に `Unused/` にある生データCSVだけで判定します。このスクリプト自体は生データCSVを `Unused/` に移動しません。

## 実行方法

```bash
source path.sh
python3 src/data/preprocessing/make_concise_data.py
```

## 動作状態の分類

| Status | Reason | 条件 |
| --- | --- | --- |
| `Present` | - | 変換・保存に成功 |
| `MissingSkeletonQuality` | `Unused` | 生データが `Unused/` に移動済み |
| `MissingNoVideo` | `NoPattern` | 時間パターン照合に失敗（`ManyShortageTimeFiles.txt` にも記録） |
| `MissingNoVideo` | - | 生データが存在せず品質側の証拠もない（動画がそもそも存在しない） |

## 出力ファイル（`data/Distributions/`）

| ファイル | 列 | 内容 |
| --- | --- | --- |
| `MotionStatus.csv` | `Directory,Task,Status,Reason` | 被験者×動作の状態明細（FIMデータを持つ被験者のみ） |
| `MissingMotionCounts.csv` | `Item,Task,Score,Subjects,Present,MissingSkeletonQuality,MissingNoVideo` | FIM Item（13項目）× 動作 × スコア(1–7) ごとの件数。各行 `Present + MissingSkeletonQuality + MissingNoVideo = Subjects` |
| `MissingMotionSummary.csv` | `Item,Score,Subjects,Present,MissingSkeletonQuality,MissingNoVideo` | 動作を合算したサマリ。`Subjects` は当該スコアの被験者数のため、3状態列の合計は `Subjects × 11` |

加えて、各被験者の状態明細を被験者フォルダ内にも出力します。

| ファイル | 列 | 内容 |
| --- | --- | --- |
| `data/ConciseData/<被験者>/MissingMotion.csv` | `Task,Status,Reason` | その被験者の11動作の状態（動作順にソート）。FIMデータを持たない被験者には出力されません |

## 集計対象

- FIMデータを持たない被験者（`non_fim.csv` 参照）は集計対象外です。
- 各Itemについて、スコアが NaN・0以下・1–7範囲外の被験者はそのItemの集計からのみ除外されます（`makeSomeTasksDataset` と同じ有効判定）。`Sum` は集計対象外です。
- セッション全体が除外された被験者（`data/OriginalData/Unused/` 直下）はディレクトリ列挙に含まれず、集計対象外です。
- 特定のFIM Itemだけを見たい場合は `MissingMotionCounts.csv` を `Item` 列でフィルタしてください（全Item分を常時出力する設計です）。

## 任意のスコアクラスでの表示

`reports/check_fim_dist/missing_motion_class_view.py` は、指定した1つのFIM Itemについて
`MissingMotionCounts.csv` のスコア(1–7)行を任意のクラス（例: 1-6 vs 7 の2クラス）に
統合して表示するスクリプトです。動作ごとの件数と欠損率をクラスごとに出力します。
出力全体（動作ごとの行と `AnyMissing` 行）は `--any-tasks` で指定した対象動作に
限定されます（1始まりの動作番号をカンマ区切り・`-` で範囲指定。デフォルトは
全11動作。歩行が対象外の場合は `1-10` のように指定）。さらに `MotionStatus.csv` と
`all_fim.csv`（パスは `--status-csv` / `--fim-csv` で変更可能）を結合し、対象動作の
うち少なくとも1動作が欠損している被験者数をクラスごとに数えた `AnyMissing`
擬似タスク行を追加します。
骨格品質による欠損が1つでもある被験者は `MissingSkeletonQuality`、それ以外で
録画欠損がある被験者は `MissingNoVideo` に1回だけ計上されるため、3つのステータス列の
合計は `Subjects` に一致します。この行の `MissingRate(%)` は「1動作以上欠損している
被験者の割合」を表します。

```bash
python reports/check_fim_dist/missing_motion_class_view.py Eating 1-6,7
python reports/check_fim_dist/missing_motion_class_view.py 0 1-6,7      # 同じItemをFIMitems()のindexで指定
python reports/check_fim_dist/missing_motion_class_view.py Bathing 1-3,4-5,6-7 -o out.csv
python reports/check_fim_dist/missing_motion_class_view.py Eating 1-6,7 --any-tasks 1-10  # exclude 11-Walking
```

`item` 引数はItem名・0始まりの `FIMitems()` index のどちらでも指定できます（対応表は
`--help` 参照）。クラス指定はカンマでクラスを区切り、クラス内では `-` が連続スコア範囲、
`+` が複数の範囲・値の結合を表します（例: `1-3+5,4,6-7`）。
