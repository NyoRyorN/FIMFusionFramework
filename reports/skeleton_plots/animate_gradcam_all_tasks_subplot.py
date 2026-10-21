import pandas as pd
import os
import math
from tqdm import tqdm
import numpy as np
import h5py
import matplotlib.pyplot as plt
import networkx as nx
import cv2
from src.utils.names import Joints, SimpleJoints, FIMitems, Tasks
from src.utils.set_plt_font import setPltFont
from src.model.gcn.components.graph import Graph


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
    """Plot all tasks as subplots in a single 3D figure, creating one combined video.

    Args:
        savePath (str): Base path to save the video/images (without extension).
        all_tasks_data (list[dict]): List of task data dicts. Each dict has:
            - 'task_name' (str): Task name for the subplot title
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

    # Find max number of animated frames
    max_frames = 0
    for d in all_tasks_data:
        if d['has_data'] and d.get('weights') is not None:
            max_frames = max(max_frames, min(
                d['weights'].shape[0], len(d['task_x'])))
    if max_frames == 0:
        max_frames = 1  # At least one frame for static display

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
        fig = plt.figure(figsize=(15 * n_cols, 10 * n_rows))

        for t_idx, td in enumerate(all_tasks_data):
            ax = fig.add_subplot(n_rows, n_cols, t_idx + 1, projection='3d')
            ax.set_title(td['task_name'], fontsize=24, pad=20)
            ax.axis("off")

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
                    # Static - use first frame with neutral color
                    frame_idx = 0
                    colors = np.ones(len(edges) + 1) * 0.5

                ax.scatter(task_x.iloc[frame_idx, :], task_z.iloc[frame_idx, :],
                           task_y.iloc[frame_idx, :], c=colors, s=250, cmap='Reds',
                           edgecolors='black', vmin=0, vmax=1)

                for edge in edges:
                    ax.plot([task_x.iloc[frame_idx, edge[0]], task_x.iloc[frame_idx, edge[1]]],
                            [task_z.iloc[frame_idx, edge[0]],
                                task_z.iloc[frame_idx, edge[1]]],
                            [task_y.iloc[frame_idx, edge[0]],
                                task_y.iloc[frame_idx, edge[1]]],
                            c='gray')
            else:
                ax.text2D(0.5, 0.5, "No Data", fontsize=20, ha='center', va='center',
                          transform=ax.transAxes)

            ax.view_init(elev=30, azim=-60)
            ax.set_xlim(-0.75, 0.75)
            ax.set_ylim(-0.75, 0.75)
            ax.set_zlim(-1, 0.5)
            ax.invert_xaxis()

        # Hide unused subplots
        for t_idx in range(n_tasks, n_rows * n_cols):
            ax_empty = fig.add_subplot(
                n_rows, n_cols, t_idx + 1, projection='3d')
            ax_empty.set_visible(False)

        # Single colorbar for the entire figure
        sm = plt.cm.ScalarMappable(
            cmap="Reds", norm=plt.Normalize(vmin=global_min_weight, vmax=global_max_weight))
        sm.set_array([])
        cax = fig.add_axes([0.93, 0.30, 0.008, 0.4])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_ticks([global_min_weight, global_max_weight])
        cbar.ax.set_yticklabels(
            [f"{global_min_weight:.4f}", f"{global_max_weight:.4f}"])
        cbar.ax.tick_params(labelsize=20)

        fig.subplots_adjust(right=0.91, wspace=0.05, hspace=0.15)

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
    """Plot all tasks as subplots in a single 2D figure, creating one combined video.

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

    # Find max number of animated frames
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
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(10 * n_cols, 10 * n_rows))
        axes = axes.flatten()

        for t_idx, td in enumerate(all_tasks_data):
            ax = axes[t_idx]
            ax.set_title(td['task_name'], fontsize=24, pad=10)

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
                    G.add_node(j, label=j)
                    pos[j] = (task_x.iloc[frame_idx, j],
                              task_y.iloc[frame_idx, j])
                G.add_edges_from(edges)

                ax.invert_xaxis()
                nx.draw_networkx_edges(G, pos, edge_color='black', ax=ax)
                nx.draw_networkx_nodes(G, pos, node_color=colors, cmap="Reds", ax=ax,
                                       edgecolors='black', vmin=0, vmax=1)
            else:
                ax.text(0.5, 0.5, "No Data", fontsize=20, ha='center', va='center',
                        transform=ax.transAxes)
                ax.invert_xaxis()

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)

        # Hide unused subplots
        for t_idx in range(n_tasks, len(axes)):
            axes[t_idx].set_visible(False)

        # Single colorbar for the entire figure
        sm = plt.cm.ScalarMappable(
            cmap="Reds", norm=plt.Normalize(vmin=global_min_weight, vmax=global_max_weight))
        sm.set_array([])
        cax = fig.add_axes([0.93, 0.30, 0.008, 0.4])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_ticks([global_min_weight, global_max_weight])
        cbar.ax.set_yticklabels(
            [f"{global_min_weight:.4f}", f"{global_max_weight:.4f}"])
        cbar.ax.tick_params(labelsize=20)

        fig.subplots_adjust(right=0.91, wspace=0.1, hspace=0.2)

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
                svg_dir, f"frame{i:03d}_2d_combined.{output_fmt}"))

        plt.close()

    if output_fmt == "mp4" and writer is not None:
        writer.release()
        if temp_fig_path and os.path.exists(temp_fig_path):
            os.remove(temp_fig_path)
        video_path = savePath + "_2D_combined.mp4"
        print(f"Saved combined 2D video: {video_path}")


def makePlotsFusionBulk(rootpath: str,
                        skl_type: str,
                        is_center: bool,
                        headers: list[str],
                        data_path: str):
    """
    Make plots for the skeleton data with fusion
    Args:
        rootpath (str): Path to the main directory containing weights and other files.
            - Ex-.../08-.../Stratified_Seed.../
        skl_type (str): Type of the skeleton (ntu-rgb+d or simple_ntu-rgb+d).
        is_center (bool): Whether to use each centering of the skeleton.
        headers (list[str]): List of joint names.
        data_path (str): Path to the data directory.
    """
    # with open(os.path.join(rootpath, "situations.yaml"), "r") as f:
    #     situations = yaml.safe_load(f)
    # task_idx_list = situations["combined_action_indices"]

    item = int(input("Enter the item(0--12): "))
    if item not in range(0, 13):
        exit("Invalid item")
    else:
        item = "{:0>2d}-{}".format(item, FIMitems()[item])
    seed = input("Enter the seed(100): ")
    validate_mode = input("Did you use K-Fold? (y/n): ")
    fold_idx = "Fold"
    if "kfold" in rootpath or validate_mode == "y":
        temp_idx = input("Enter the fold index(0--4): ")
        if temp_idx not in ["0", "1", "2", "3", "4"]:
            exit("Invalid fold index")
        else:
            fold_idx += temp_idx
    else:
        fold_idx = None
    mode = input("Enter the mode(2D or 3D or Both): ")
    if mode not in ["2D", "3D", "Both"]:
        print("Invalid mode and set to Both")
        mode = "Both"
    output_fmt = input(
        "Enter the output format(mp4 or svg)(default: mp4): ") or "mp4"
    if output_fmt not in ["mp4", "svg"]:
        print("Invalid output format and set to mp4")
        output_fmt = "mp4"

    main_path = os.path.join(
        rootpath, item, "Stratified_Seed" + seed)
    if fold_idx is not None:
        main_path = os.path.join(main_path, fold_idx)

    # weights_shifts = ["_Att", "_CAM"]
    weights_shifts = ["_CAM"]
    frame_len = 150

    for shift in weights_shifts:
        temp_weights_path = os.path.join(
            main_path, "weights" + shift + ".h5")
        if os.path.exists(temp_weights_path) is False:
            print("No weights: ", temp_weights_path)
            continue

        with h5py.File(temp_weights_path, 'r') as f:
            data_df = pd.read_csv(os.path.join(
                main_path, "testDir.csv"), header=0)
            data_dirs = data_df["Directory"].values

            for index, data_dir in enumerate(data_dirs):
                print(f"{index} / {len(data_dirs)} - {data_dir}")
                temp_data = f[data_dir]
                available_tasks = set(temp_data.keys())

                # Collect all tasks' data for this data_dir
                all_tasks_data_list = []
                global_min = float('inf')
                global_max = float('-inf')

                # Only process tasks that have weights in the h5 file
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

                        if not is_center:
                            if skl_type == "ntu-rgb+d":
                                center_joint = "SpineShoulder"
                            else:
                                center_joint = "SpineMid"
                            data_x = data_x - \
                                data_x.at[data_x.index[0], center_joint]
                            data_y = data_y - \
                                data_y.at[data_y.index[0], center_joint]
                            data_z = data_z - \
                                data_z.at[data_z.index[0], center_joint]

                        weights_raw = pd.DataFrame(
                            temp_data[task_key])
                        if weights_raw.shape[0] > weights_raw.shape[1]:
                            weights_raw = weights_raw.T
                        weights_raw.index = headers

                        # Convert to (T, V) format
                        weights_tv = weights_raw.T if weights_raw.shape[
                            0] < weights_raw.shape[1] else weights_raw

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

                        # Get start frame
                        start_frame = 0
                        if task in data_df.columns:
                            sf_val = data_df[data_df["Directory"]
                                             == data_dir][task].values
                            if len(sf_val) > 0 and not pd.isna(sf_val[0]):
                                start_frame = int(sf_val[0])

                        data_x_trim = data_x.iloc[
                            start_frame:start_frame + len(new_weights), :
                        ].reset_index(drop=True)
                        data_y_trim = data_y.iloc[
                            start_frame:start_frame + len(new_weights), :
                        ].reset_index(drop=True)
                        data_z_trim = data_z.iloc[
                            start_frame:start_frame + len(new_weights), :
                        ].reset_index(drop=True)

                        global_min = min(
                            global_min, new_weights.min().min())
                        global_max = max(
                            global_max, new_weights.max().max())

                        task_info.update({
                            'task_x': data_x_trim,
                            'task_y': data_y_trim,
                            'task_z': data_z_trim,
                            'weights': new_weights.reset_index(drop=True),
                        })

                    all_tasks_data_list.append(task_info)

                # Handle case where no weights found at all
                if global_min == float('inf'):
                    global_min = 0
                    global_max = 1

                # Setup save path
                save_path = os.path.join(rootpath, "Weights" + shift)
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(
                    save_path, f"{item}_Seed{seed}") if fold_idx is None else os.path.join(
                    save_path, f"{item}_Seed{seed}_{fold_idx}")
                os.makedirs(save_path, exist_ok=True)
                combined_save_path = os.path.join(save_path, data_dir)

                # Plot combined figure with all tasks as subplots
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
                print("Finished_" + data_dir)
                print("=============================================")
            print("All Data Plotted in " + shift)


if __name__ == "__main__":
    rootpath = input(
        "Enter the root path of the weights(basename starts with Ex_...).: ")
    # is_center = input(
    #     "Do you want to use the each centering of the skeleton? (y/n): ")
    # is_center = True if is_center == "y" else False
    is_center = False  # Default to True for simplicity
    headers = Joints()

    skl_type = "ntu-rgb+d"
    # if "Concise" in rootpath:
    headers = SimpleJoints()
    skl_type = "simple_ntu-rgb+d"
    data_path = plot_data_dict[skl_type]["normal" if not is_center else "each_center"]
    # else:
    #     data_path = plot_data_dict[skl_type]["normal" if not is_center else "each_center"]
    # if skl_type not in ["ntu-rgb+d", "simple_ntu-rgb+d"]:
    #     exit("Invalid skeleton type")

    makePlotsFusionBulk(rootpath=rootpath,
                        skl_type=skl_type,
                        is_center=is_center,
                        headers=headers,
                        data_path=data_path)
