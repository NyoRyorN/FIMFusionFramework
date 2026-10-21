# reports/calculates — 被験者横断変数の集計

fold ごとの予測を被験者ごとの一致率に集計する。アブレーション box/violin プロットの入力
となる。

## ファイル

| ファイル | 役割 |
|---|---|
| `calc_cross_subject_variables.py` | fold 単位の `PredictTruelabel.csv`（構成 `<root>/<Item>/<Stratified_SeedXXX>/FoldX/`）を連結し、被験者ごとの一致率を `CrossSubjectVariables` CSV にまとめる。 |
