# src/data/preprocessing — 生データ前処理パイプライン

生の NEDO 計測データを、データセット生成の入力となる骨格点削減済み `ConciseData` に変換
する。2 つのエントリを順に実行する。まず `orderliness_data.py` が生ファイルツリーを整理し、
次に `make_concise_data.py` が数値パイプライン（時間補完 → フィルタ + 特徴量計算 → 骨格点
削減出力）を実行する。再利用可能な数値ヘルパは `utils/` に、単発のファイル/ディレクトリ
整理スクリプトは `order/` に置く。

## ファイル

| ファイル | 役割 |
|---|---|
| `orderliness_data.py` | 生データツリーを整理。不要ディレクトリ/ファイルの移動・削除、被験者ごとのファイル集約、重複除去、ファイル名標準化、非 FIM ファイルの分離。 |
| `make_concise_data.py` | `data/OriginalData/` から直接 `data/ConciseData/` を生成。時間補完 → rolling median/mean フィルタ + 速度計算 → SimpleJoints の位置/速度と再計算した角度/骨長。 |

## サブフォルダ

| サブフォルダ | 役割 |
|---|---|
| [`order/`](order/) | `orderliness_data.py` から呼び出されるファイル/ディレクトリ整理スクリプト（リネーム・移動・削除・重複除去）。 |
| [`utils/`](utils/) | 数値前処理ヘルパ。速度/角度/骨長計算、FIM 分布・欠損動作・時間不足の品質チェック。 |

## 実行方法

```bash
source path.sh
python src/data/preprocessing/orderliness_data.py   # 生データ整理
python src/data/preprocessing/make_concise_data.py   # 数値パイプライン一括実行
```
