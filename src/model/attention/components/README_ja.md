# src/model/attention/components — attention 構成要素

`attention_part.py` がエンコーダの attention 段に組み立てる基本 attention モジュール群。
提案手法（本プロジェクト）の一部。

## ファイル

| ファイル | 役割 |
|---|---|
| `attention.py` | 基本 attention モジュール。attention 重みを計算し、重み付き特徴を出力する。 |
| `st_joint_attention.py` | ST-joint attention。フレームと関節をまとめて統合する空間時間同時 attention。 |
