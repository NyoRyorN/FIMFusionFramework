import os
import sys
import math
import h5py
import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.model.gcn.components.graph import Graph
from src.utils.names import Joints, SimpleJoints, FIMitems, Tasks
from src.utils.set_plt_font import setPltFont


# MPEG-4 maximum resolution limit (mp4v)
MAX_VIDEO_DIM = 4096
setPltFont()


def _resize_for_video(img, max_dim=MAX_VIDEO_DIM):
    """Resize image to fit within MPEG-4 codec dimension limits while preserving aspect ratio."""
    h, w = img.shape[:2]
    if w <= max_dim and h <= max_dim:
        return img
    scale = min(max_dim / w, max_dim / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # width and height even number (video codec items)
    new_w = new_w - (new_w % 2)
    new_h = new_h - (new_h % 2)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


plot_data_dict = {
    "ntu-rgb+d": {
        "normal": os.path.join("data", "FilteredData"),
        "each_center": os.path.join("data", "FilteredCenterDataBasedSpineShoulder"),
    },
    "simple_ntu-rgb+d": {
        "normal": os.path.join("data", "ConciseData"),
        "each_center": os.path.join("data", "ConciseDataBasedSpineMid"),
    }
}
data_path = "data"
all_tasks = Tasks()


def PlotAllTasks3D_Combined(savePath, all_tasks_data, global_min_weight, global_max_weight,
                            skl_type="ntu-rgb+d", output_fmt="mp4"):
    """Plot all tasks in a single 3D axes, translating each skeleton to its grid position.

    Each skeleton is shifted in the (task_x, task_z) horizontal plane so all actions
    are visible simultaneously without subplots. x is pre-inverted by negation.

    Args:
        savePath (str): Base path to save the video/images (without extension).
        all_tasks_data (list[dict]): List of task data dicts. Each dict has:
            - 'task_name' (str): Task name
            - 'has_data' (bool): Whether skeleton CSV data exists
            - 'is_static' (bool): Whether to plot statically (no animation)
            - 'task_x', 'task_y', 'task_z' (pd.DataFrame): Skeleton data (if has_data)
            - 'weights' (pd.DataFrame or None): Weight data
        global_min_weight (float): Global min weight across all tasks.
        global_max_weight (float): Global max weight across all tasks.
        skl_type (str): Skeleton type.
        output_fmt (str): "mp4" or "svg".
    """
    n_tasks = len(all_tasks_data)
    n_rows = 2
    n_cols = math.ceil(n_tasks / n_rows)

    # Spacing between skeleton centers on the shared 3D canvas.
    # Original data: task_x, task_z ∈ [-0.75, 0.75]; task_y ∈ [-1, 0.5]
    x_step = 3.0   # column spacing along x (after x-inversion)
    z_step = 3.0   # row spacing along task_z (depth direction)

    max_frames = 0
    for d in all_tasks_data:
        if d['has_data'] and d.get('weights') is not None:
            max_frames = max(max_frames, min(
                d['weights'].shape[0], len(d['task_x'])))
    if max_frames == 0:
        max_frames = 1

    edges = Graph(skl_type).neighbor_link

    writer = None
    temp_fig_path = None
    if output_fmt == "mp4":
        parent_dir = os.path.dirname(
            savePath) if os.path.dirname(savePath) else "."
        os.makedirs(parent_dir, exist_ok=True)
        temp_fig_path = os.path.join(parent_dir, "temp_combined_3d_frame.png")
    else:
        svg_dir = os.path.join(savePath, "figures_3D_combined")
        os.makedirs(svg_dir, exist_ok=True)

    for i in tqdm(range(max_frames), leave=False, desc="Combined 3D"):
        fig = plt.figure(figsize=(18 * n_cols, 18 * n_rows))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.axis("off")

        for t_idx, td in enumerate(all_tasks_data):
            row = t_idx // n_cols
            col = t_idx % n_cols
            off_x = col * x_step   # column offset in x direction
            off_z = row * z_step   # row offset in task_z (depth) direction

            if td['has_data']:
                task_x = td['task_x']
                task_y = td['task_y']
                task_z = td['task_z']
                weights = td.get('weights')

                if weights is not None and not td.get('is_static', False):
                    frame_count = min(weights.shape[0], len(task_x))
                    frame_idx = min(i, frame_count - 1)
                    sizes = weights.iloc[frame_idx].values
                    if global_max_weight - global_min_weight < 1e-6:
                        colors = np.zeros(len(edges) + 1)
                    else:
                        colors = (sizes - global_min_weight) / \
                            (global_max_weight - global_min_weight)
                else:
                    frame_idx = 0
                    colors = np.ones(len(edges) + 1) * 0.5

                # Pre-apply x inversion (negate task_x) and add grid offsets
                fx = -task_x.iloc[frame_idx, :] + off_x
                fz = task_z.iloc[frame_idx, :] + off_z
                fy = task_y.iloc[frame_idx, :]

                ax.scatter(fx, fz, fy, c=colors, s=250, cmap='Reds',
                           edgecolors='black', vmin=0, vmax=1)
                for edge in edges:
                    ax.plot([fx.iloc[edge[0]], fx.iloc[edge[1]]],
                            [fz.iloc[edge[0]], fz.iloc[edge[1]]],
                            [fy.iloc[edge[0]], fy.iloc[edge[1]]], c='gray')

                ax.text(off_x, off_z, 0.7, td['task_name'],
                        ha='center', va='bottom', fontsize=26)
            else:
                ax.text(off_x, off_z, 0.0, f"{td['task_name']}\n(No Data)",
                        ha='center', va='center', fontsize=26, color='gray')

        ax.view_init(elev=30, azim=-60)
        x_lim_lo, x_lim_hi = -0.9, (n_cols - 1) * x_step + 0.9
        y_lim_lo, y_lim_hi = -0.9, (n_rows - 1) * z_step + 0.9
        z_lim_lo, z_lim_hi = -1.1, 0.8
        ax.set_xlim(x_lim_lo, x_lim_hi)
        ax.set_ylim(y_lim_lo, y_lim_hi)
        ax.set_zlim(z_lim_lo, z_lim_hi)
        # Make 1 data unit equal physical size in all directions
        ax.set_box_aspect([
            x_lim_hi - x_lim_lo,
            y_lim_hi - y_lim_lo,
            z_lim_hi - z_lim_lo,
        ])

        sm = plt.cm.ScalarMappable(
            cmap="Reds", norm=plt.Normalize(vmin=global_min_weight, vmax=global_max_weight))
        sm.set_array([])
        cax = fig.add_axes([0.82, 0.3, 0.011, 0.3])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_ticks([global_min_weight, global_max_weight])
        cbar.ax.set_yticklabels(
            [f"{global_min_weight:.4f}", f"{global_max_weight:.4f}"])
        cbar.ax.tick_params(labelsize=40)

        if output_fmt == "mp4":
            fig.savefig(temp_fig_path, format='png', dpi=100)
            img = cv2.imread(temp_fig_path)
            img = _resize_for_video(img)
            if writer is None:
                h, w = img.shape[:2]
                codec = cv2.VideoWriter_fourcc(*"mp4v")
                video_path = savePath + "_3D_combined.mp4"
                writer = cv2.VideoWriter(video_path, codec, 30, (w, h), True)
            writer.write(img)
        else:
            fig.savefig(os.path.join(
                svg_dir, f"frame{i:03d}_3d_combined.{output_fmt}"))

        plt.close()

    if output_fmt == "mp4" and writer is not None:
        writer.release()
        if temp_fig_path and os.path.exists(temp_fig_path):
            os.remove(temp_fig_path)
        video_path = savePath + "_3D_combined.mp4"
        print(f"Saved combined 3D video: {video_path}")


def PlotAllTasks2D_Combined(savePath, all_tasks_data, global_min_weight, global_max_weight,
                            skl_type="ntu-rgb+d", output_fmt="mp4"):
    """Plot all tasks in a single 2D axes, translating each skeleton to its grid position.

    Each skeleton is shifted by (col * x_step, -row * y_step) so all actions are
    visible simultaneously without subplots. No whitespace is wasted on empty regions.

    Args:
        savePath (str): Base path to save the video/images (without extension).
        all_tasks_data (list[dict]): List of task data dicts (same format as PlotAllTasks3D_Combined).
        global_min_weight (float): Global min weight across all tasks.
        global_max_weight (float): Global max weight across all tasks.
        skl_type (str): Skeleton type.
        output_fmt (str): "mp4" or "svg".
    """
    n_tasks = len(all_tasks_data)
    n_rows = 2
    n_cols = math.ceil(n_tasks / n_rows)

    # Spacing between skeleton centers on the shared canvas.
    # Original data range per task: task_x, task_y ∈ [-1, 1]
    x_step = 4.0   # column spacing along x
    y_step = 4.0   # row spacing along y (rows go downward)

    max_frames = 0
    for d in all_tasks_data:
        if d['has_data'] and d.get('weights') is not None:
            max_frames = max(max_frames, min(
                d['weights'].shape[0], len(d['task_x'])))
    if max_frames == 0:
        max_frames = 1

    edges = Graph(skl_type).neighbor_link

    writer = None
    temp_fig_path = None
    if output_fmt == "mp4":
        parent_dir = os.path.dirname(
            savePath) if os.path.dirname(savePath) else "."
        os.makedirs(parent_dir, exist_ok=True)
        temp_fig_path = os.path.join(parent_dir, "temp_combined_2d_frame.png")
    else:
        svg_dir = os.path.join(savePath, "figures_2D_combined")
        os.makedirs(svg_dir, exist_ok=True)

    for i in tqdm(range(max_frames), leave=False, desc="Combined 2D"):
        fig, ax = plt.subplots(1, 1, figsize=(12 * n_cols, 14 * n_rows))
        ax.set_aspect('equal', adjustable='datalim')
        ax.axis('off')

        for t_idx, td in enumerate(all_tasks_data):
            row = t_idx // n_cols
            col = t_idx % n_cols
            off_x = col * x_step
            off_y = -row * y_step

            # Task label above each skeleton
            ax.text(off_x, off_y + 1.15, td['task_name'],
                    ha='center', va='bottom', fontsize=28, fontweight='bold')

            if td['has_data']:
                task_x = td['task_x']
                task_y = td['task_y']
                weights = td.get('weights')

                if weights is not None and not td.get('is_static', False):
                    frame_count = min(weights.shape[0], len(task_x))
                    frame_idx = min(i, frame_count - 1)
                    sizes = weights.iloc[frame_idx].values
                    if global_max_weight - global_min_weight < 1e-6:
                        colors = np.zeros(len(edges) + 1)
                    else:
                        colors = (sizes - global_min_weight) / \
                            (global_max_weight - global_min_weight)
                else:
                    frame_idx = 0
                    colors = np.ones(len(edges) + 1) * 0.5

                G = nx.Graph()
                pos = {}
                for j in range(len(edges) + 1):
                    G.add_node(j)
                    pos[j] = (task_x.iloc[frame_idx, j] + off_x,
                              task_y.iloc[frame_idx, j] + off_y)
                G.add_edges_from(edges)

                nx.draw_networkx_edges(G, pos, edge_color='black', ax=ax)
                nx.draw_networkx_nodes(G, pos, node_color=colors, cmap="Reds", ax=ax,
                                       edgecolors='black', vmin=0, vmax=1)
            else:
                ax.text(off_x, off_y, "No Data",
                        ha='center', va='center', fontsize=32, color='gray')

        # Light separator lines between grid cells
        for c in range(1, n_cols):
            ax.axvline(x=(c - 0.5) * x_step, color='lightgray',
                       linestyle='--', linewidth=0.8, zorder=0)
        for r in range(1, n_rows):
            ax.axhline(y=-(r - 0.5) * y_step, color='lightgray',
                       linestyle='--', linewidth=0.8, zorder=0)

        ax.set_xlim(-1.5, (n_cols - 1) * x_step + 1.5)
        ax.set_ylim(-(n_rows - 1) * y_step - 1.5, 1.5)

        sm = plt.cm.ScalarMappable(
            cmap="Reds", norm=plt.Normalize(vmin=global_min_weight, vmax=global_max_weight))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.45, pad=0.01)
        cbar.set_ticks([global_min_weight, global_max_weight])
        cbar.ax.set_yticklabels(
            [f"{global_min_weight:.4f}", f"{global_max_weight:.4f}"])
        cbar.ax.tick_params(labelsize=32)

        if output_fmt == "mp4":
            fig.savefig(temp_fig_path, format='png', dpi=100)
            img = cv2.imread(temp_fig_path)
            img = _resize_for_video(img)
            if writer is None:
                h, w = img.shape[:2]
                codec = cv2.VideoWriter_fourcc(*"mp4v")
                video_path = savePath + "_2D_combined.mp4"
                writer = cv2.VideoWriter(video_path, codec, 30, (w, h), True)
            writer.write(img)
        else:
            fig.savefig(os.path.join(
                svg_dir, f"frame{i:03d}_2d_combined.{output_fmt}"),
                bbox_inches='tight')

        plt.close()

    if output_fmt == "mp4" and writer is not None:
        writer.release()
        if temp_fig_path and os.path.exists(temp_fig_path):
            os.remove(temp_fig_path)
        video_path = savePath + "_2D_combined.mp4"
        print(f"Saved combined 2D video: {video_path}")


# ------------------------------------------------------------------
# Directory traversal helpers (mirror export_gradcam_stats.py)
# ------------------------------------------------------------------

def _list_fold_dirs(seed_dir: str) -> list[str]:
    """List fold directories directly under one Stratified_Seed* directory."""
    if not os.path.isdir(seed_dir):
        return []

    def _fold_sort_key(name: str) -> tuple:
        suffix = name.removeprefix("Fold")
        return (0, int(suffix)) if suffix.isdigit() else (1, suffix)

    return sorted(
        [
            name for name in os.listdir(seed_dir)
            if name.startswith("Fold")
            and os.path.isdir(os.path.join(seed_dir, name))
        ],
        key=_fold_sort_key)


def _list_seed_dirs(item_dir: str) -> list[str]:
    """List Stratified_Seed* directories directly under one item directory."""
    if not os.path.isdir(item_dir):
        return []
    return sorted(
        os.path.join(item_dir, name)
        for name in os.listdir(item_dir)
        if name.startswith("Stratified_Seed")
        and os.path.isdir(os.path.join(item_dir, name))
    )


def _find_seed_dirs(root_dir: str) -> list[str]:
    """Recursively find all Stratified_Seed* directories that contain CAM weights.

    A directory qualifies if it contains weights_CAM.h5 directly or has at
    least one Fold*/weights_CAM.h5. Recursion stops at a qualifying directory
    so nested seeds are not double-counted.
    """
    seed_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if not os.path.basename(dirpath).startswith("Stratified_Seed"):
            continue
        has_cam = "weights_CAM.h5" in filenames or any(
            os.path.exists(os.path.join(dirpath, d, "weights_CAM.h5"))
            for d in dirnames
            if d.startswith("Fold")
        )
        if has_cam:
            seed_dirs.append(dirpath)
            dirnames.clear()  # do not recurse further into this seed directory
    return sorted(seed_dirs)


# ------------------------------------------------------------------
# Core animation processing
# ------------------------------------------------------------------

def _process_single_fold_animate(
        main_path: str,
        save_path_base: str,
        skl_type: str,
        is_center: bool,
        headers: list[str],
        data_path: str,
        mode: str,
        output_fmt: str) -> None:
    """Read weights_CAM.h5 and plot animations for each subject in one fold.

    Processing flow:
        1. Read weights_CAM.h5 and testDir.csv from main_path.
        2. For each subject, load skeleton CSVs and GradCAM weights.
        3. Trim and optionally center the skeleton on the first frame.
        4. Save combined 2D and/or 3D animations under save_path_base.

    Args:
        main_path: Directory containing weights_CAM.h5 and testDir.csv.
        save_path_base: Parent directory for output (data_dir name is appended).
        skl_type: Skeleton layout name (e.g. "simple_ntu-rgb+d").
        is_center: If True, skeleton data is already centered; otherwise center
            on the first frame using the spine reference joint.
        headers: Joint-name list matching the stored GradCAM arrays.
        data_path: Root path containing skeleton CSVs as data_dir/task/{p_x,p_y,p_z}.csv.
        mode: "2D", "3D", or "Both".
        output_fmt: "mp4" or "svg".
    """
    temp_weights_path = os.path.join(main_path, "weights_CAM.h5")
    if not os.path.exists(temp_weights_path):
        print(f"No weights: {temp_weights_path}")
        return

    testdir_csv = os.path.join(main_path, "testDir.csv")
    if not os.path.exists(testdir_csv):
        print(f"No testDir.csv: {testdir_csv}")
        return

    frame_len = 150

    with h5py.File(temp_weights_path, 'r') as f:
        data_df = pd.read_csv(testdir_csv, header=0)
        data_dirs = data_df["Directory"].values

        for index, data_dir in enumerate(data_dirs):
            print(f"{index} / {len(data_dirs)} - {data_dir}")
            if data_dir not in f:
                print(f"  Skip: {data_dir} not in weights file")
                continue
            temp_data = f[data_dir]
            available_tasks = set(temp_data.keys())

            all_tasks_data_list = []
            global_min = float('inf')
            global_max = float('-inf')

            for task_key in sorted(available_tasks, key=lambda x: int(x)):
                task_global_idx = int(task_key)
                task = all_tasks[task_global_idx]

                has_skeleton = os.path.exists(
                    os.path.join(data_path, data_dir, task))

                task_info = {
                    'task_name': task,
                    'task_idx': task_global_idx,
                    'has_data': has_skeleton,
                    'has_weights': True,
                    'is_static': False,
                }

                if has_skeleton:
                    data_x = pd.read_csv(os.path.join(
                        data_path, data_dir, task, "p_x.csv"), header=0)
                    data_y = pd.read_csv(os.path.join(
                        data_path, data_dir, task, "p_y.csv"), header=0)
                    data_z = pd.read_csv(os.path.join(
                        data_path, data_dir, task, "p_z.csv"), header=0)

                    weights_raw = pd.DataFrame(temp_data[task_key])
                    if weights_raw.shape[0] > weights_raw.shape[1]:
                        weights_raw = weights_raw.T
                    weights_raw.index = headers

                    # Convert to (T, V) format
                    weights_tv = (weights_raw.T
                                  if weights_raw.shape[0] < weights_raw.shape[1]
                                  else weights_raw)

                    # Expand short weights (Att weights)
                    if weights_tv.shape[0] < frame_len:
                        new_weights = weights_tv.loc[
                            weights_tv.index.repeat(4)
                        ].reset_index(drop=True)
                        new_weights = new_weights.iloc[
                            len(new_weights) // 2 - frame_len // 2:
                            len(new_weights) // 2 + frame_len // 2
                        ]
                    else:
                        new_weights = weights_tv

                    start_frame = data_df[
                        data_df["Directory"] == data_dir][task].values[0]

                    data_x_trim = data_x.iloc[
                        start_frame:start_frame + len(new_weights), :
                    ].reset_index(drop=True)
                    data_y_trim = data_y.iloc[
                        start_frame:start_frame + len(new_weights), :
                    ].reset_index(drop=True)
                    data_z_trim = data_z.iloc[
                        start_frame:start_frame + len(new_weights), :
                    ].reset_index(drop=True)

                    if not is_center:
                        # Center on the first frame of the trimmed sequence using
                        # the spine reference joint for this skeleton layout.
                        center_joint = ("SpineShoulder"
                                        if skl_type == "ntu-rgb+d"
                                        else "SpineMid")
                        data_x_trim = data_x_trim - \
                            data_x_trim.at[data_x_trim.index[0], center_joint]
                        data_y_trim = data_y_trim - \
                            data_y_trim.at[data_y_trim.index[0], center_joint]
                        data_z_trim = data_z_trim - \
                            data_z_trim.at[data_z_trim.index[0], center_joint]

                    global_min = min(global_min, new_weights.min().min())
                    global_max = max(global_max, new_weights.max().max())

                    task_info.update({
                        'task_x': data_x_trim,
                        'task_y': data_y_trim,
                        'task_z': data_z_trim,
                        'weights': new_weights.reset_index(drop=True),
                    })

                all_tasks_data_list.append(task_info)

            if global_min == float('inf'):
                global_min = 0
                global_max = 1

            combined_save_path = os.path.join(save_path_base, data_dir)

            if mode in ["Both", "3D"]:
                PlotAllTasks3D_Combined(
                    combined_save_path, all_tasks_data_list,
                    global_min, global_max, skl_type, output_fmt)
            if mode in ["Both", "2D"]:
                PlotAllTasks2D_Combined(
                    combined_save_path, all_tasks_data_list,
                    global_min, global_max, skl_type, output_fmt)

            print("=============================================")
            print("Combined Data Plotted")
            print(f"Finished_{data_dir}")
            print("=============================================")

    print(f"All data plotted for: {main_path}")


def _process_seed_dir_animate(
        seed_dir: str,
        skl_type: str,
        is_center: bool,
        headers: list[str],
        data_path: str,
        mode: str,
        output_fmt: str) -> None:
    """Process one Stratified_Seed* directory with automatic K-Fold detection.

    Processing flow:
        1. Auto-detect K-Fold layout by scanning Fold* subdirectories.
        2. Call _process_single_fold_animate for each fold (or the seed root
           when no folds exist).
        3. Output is saved under seed_dir/CAM_Animations[/fold_idx].

    Args:
        seed_dir: Stratified_Seed* directory for the target run.
        skl_type: Skeleton layout name.
        is_center: Whether skeleton data is pre-centered.
        headers: Joint-name list matching the stored GradCAM arrays.
        data_path: Root path containing skeleton CSV data.
        mode: "2D", "3D", or "Both".
        output_fmt: "mp4" or "svg".
    """
    fold_indices = _list_fold_dirs(seed_dir)
    if fold_indices:
        print(f"K-Fold detected: {fold_indices}")
    else:
        if not os.path.exists(os.path.join(seed_dir, "weights_CAM.h5")):
            print(f"No CAM weights found in {seed_dir}; skip.")
            return
        fold_indices = [None]

    cam_anim_dir = os.path.join(seed_dir, "CAM_Animations")

    for fold_idx in fold_indices:
        fold_label = fold_idx if fold_idx is not None else "NoFold"
        print(f"\n{'='*50}")
        print(f"Processing {fold_label}...")
        print(f"{'='*50}")

        main_path = (os.path.join(seed_dir, fold_idx)
                     if fold_idx is not None else seed_dir)
        save_path_base = (os.path.join(cam_anim_dir, fold_idx)
                          if fold_idx is not None else cam_anim_dir)

        _process_single_fold_animate(
            main_path=main_path,
            save_path_base=save_path_base,
            skl_type=skl_type,
            is_center=is_center,
            headers=headers,
            data_path=data_path,
            mode=mode,
            output_fmt=output_fmt)


def makePlotsFusionBulk(rootpath: str,
                        skl_type: str,
                        is_center: bool,
                        headers: list[str],
                        data_path: str,
                        mode: str,
                        output_fmt: str) -> None:
    """Export GradCAM animations for one interactively chosen item/seed/fold.

    Asks for item (0--12), seed, K-Fold flag, and fold index, then delegates
    all processing to _process_single_fold_animate. Output is saved to
    rootpath/Weights_CAM/{item}_Seed{seed}[_{fold_idx}]/.

    Args:
        rootpath: Parent directory containing FIM item subdirectories.
        skl_type: Skeleton layout name.
        is_center: Whether skeleton data is pre-centered.
        headers: Joint-name list matching the stored GradCAM arrays.
        data_path: Root path containing skeleton CSV data.
        mode: "2D", "3D", or "Both".
        output_fmt: "mp4" or "svg".
    """
    item = int(input("Enter the item(0--12): "))
    if item not in range(0, 13):
        sys.exit("Invalid item")
    item = "{:0>2d}-{}".format(item, FIMitems()[item])
    seed = input("Enter the seed(100): ")
    validate_mode = input("Did you use K-Fold? (y/n): ")
    fold_idx = "Fold"
    if "kfold" in rootpath or validate_mode == "y":
        temp_idx = input("Enter the fold index(0--4): ")
        if temp_idx not in ["0", "1", "2", "3", "4"]:
            sys.exit("Invalid fold index")
        fold_idx += temp_idx
    else:
        fold_idx = None

    main_path = os.path.join(rootpath, item, "Stratified_Seed" + seed)
    if fold_idx is not None:
        main_path = os.path.join(main_path, fold_idx)

    # Preserve the original rootpath/Weights_CAM/... output structure.
    save_path = os.path.join(rootpath, "Weights_CAM")
    save_path = (
        os.path.join(save_path, f"{item}_Seed{seed}")
        if fold_idx is None
        else os.path.join(save_path, f"{item}_Seed{seed}_{fold_idx}")
    )

    _process_single_fold_animate(
        main_path=main_path,
        save_path_base=save_path,
        skl_type=skl_type,
        is_center=is_center,
        headers=headers,
        data_path=data_path,
        mode=mode,
        output_fmt=output_fmt)


if __name__ == "__main__":
    is_center = False  # use "normal" data path and center on first frame manually
    headers = SimpleJoints()
    skl_type = "simple_ntu-rgb+d"
    data_path = plot_data_dict[skl_type]["normal" if not is_center else "each_center"]

    mode = input(
        "Enter the plot mode (2D / 3D / Both) [default: Both]: ").strip() or "Both"
    if mode not in ["2D", "3D", "Both"]:
        print("Invalid mode; defaulting to Both.")
        mode = "Both"
    output_fmt = input(
        "Enter the output format (mp4 / svg) [default: mp4]: ").strip() or "mp4"
    if output_fmt not in ["mp4", "svg"]:
        print("Invalid output format; defaulting to mp4.")
        output_fmt = "mp4"

    print("Select processing mode:")
    print("  1: Specify item directory (parent of Stratified_Seed*)"
          " — process all seeds in that item")
    print("  2: Specify a root directory — find and process all seeds"
          " containing CAM weights")
    print("  3: Specify root directory and enter item/seed/fold interactively")
    _mode = input("Mode (1/2/3): ").strip()

    if _mode == "1":
        # ------------------------------------------------------------------
        # Mode 1: item directory → all Stratified_Seed* inside it
        # ------------------------------------------------------------------
        item_dir = input(
            "Enter item directory path (parent of Stratified_Seed*): ").strip()
        seed_dirs = _list_seed_dirs(item_dir)
        if not seed_dirs:
            sys.exit(
                f"No Stratified_Seed* directories found under: {item_dir}")
        item = os.path.basename(os.path.normpath(item_dir))
        for seed_dir in seed_dirs:
            seed = os.path.basename(seed_dir).removeprefix("Stratified_Seed")
            print(f"\n{'#'*60}")
            print(f"# {item} / Seed {seed}")
            print(f"{'#'*60}")
            _process_seed_dir_animate(
                seed_dir=seed_dir,
                skl_type=skl_type,
                is_center=is_center,
                headers=headers,
                data_path=data_path,
                mode=mode,
                output_fmt=output_fmt)

    elif _mode == "2":
        # ------------------------------------------------------------------
        # Mode 2: root directory → recursively find all seeds with CAM weights
        # ------------------------------------------------------------------
        root_dir = input(
            "Enter root directory path to search recursively: ").strip()
        seed_dirs = _find_seed_dirs(root_dir)
        if not seed_dirs:
            sys.exit(
                f"No Stratified_Seed* directories with CAM weights found "
                f"under: {root_dir}")
        print(f"Found {len(seed_dirs)} seed director"
              f"{'y' if len(seed_dirs) == 1 else 'ies'}.")
        for seed_dir in seed_dirs:
            item = os.path.basename(os.path.dirname(seed_dir))
            seed = os.path.basename(seed_dir).removeprefix("Stratified_Seed")
            print(f"\n{'#'*60}")
            print(f"# {item} / Seed {seed}")
            print(f"{'#'*60}")
            _process_seed_dir_animate(
                seed_dir=seed_dir,
                skl_type=skl_type,
                is_center=is_center,
                headers=headers,
                data_path=data_path,
                mode=mode,
                output_fmt=output_fmt)

    elif _mode == "3":
        # ------------------------------------------------------------------
        # Mode 3: original interactive mode (rootpath + item + seed + fold)
        # ------------------------------------------------------------------
        rootpath = input(
            "Enter root path (parent of item directories, "
            "basename starts with Ex_...): ").strip()
        makePlotsFusionBulk(
            rootpath=rootpath,
            skl_type=skl_type,
            is_center=is_center,
            headers=headers,
            data_path=data_path,
            mode=mode,
            output_fmt=output_fmt)

    else:
        sys.exit("Invalid mode selection.")
