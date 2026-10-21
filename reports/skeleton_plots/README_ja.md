# reports/skeleton_plots — 骨格と GradCAM の可視化

骨格系列を描画し、Attention / GradCAM 重み（`src/model/attention/cam_fusion.py` 由来、
評価時に保存）を動画・複数タスクのサブプロット・分布統計として重畳表示する。
`plot_skeleton.py` とアニメーションスクリプトは共通の 2D/3D レンダラを基盤とする。

## ファイル

| ファイル | 役割 |
|---|---|
| `skeleton_renderer_2d.py` | `Plot2D`。共通の 2D 骨格レンダラ（フレームごとの関節/骨）。 |
| `skeleton_renderer_3d.py` | 共通の 3D 骨格レンダラ。 |
| `plot_skeleton.py` | 2D レンダラで静止骨格図を描画する。 |
| `animate_gradcam_per_task.py` | GradCAM 重み付き骨格動画をアクションタスクごとに 1 本ずつ生成する。 |
| `animate_gradcam_all_tasks_subplot.py` | 全タスクをサブプロットのグリッドとしてまとめて動画化する。 |
| `animate_gradcam_all_tasks_compact.py` | 全タスクをコンパクトな統合ビューでまとめて動画化する。 |
| `export_gradcam_stats.py` | seed 単位の GradCAM 代表値（被験者ごとの Max/Mean/Median を action/joint/body-part レベルで）を書き出す。 |
