import matplotlib.pyplot as plt
import networkx as nx
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.model.gcn.components.graph import Graph
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2


def Plot3D(savePath, task_x, task_y, task_z, weights=None, skl_type="ntu-rgb+d", start_frame=0, output_fmt="mp4"):
    """Plot 3D graph

    Args:
        savePath (str): Path to save the plot
        task_x (pd.DataFrame): x-axis data
        task_y (pd.DataFrame): y-axis data
        task_z (pd.DataFrame): z-axis data
        weights (pd.DataFrame, optional): Node weights for sizes and colors. Defaults to None.
        start_frame (int, optional): Starting frame index for the data. Defaults to 0.
        output_fmt (str, optional): Output format. "mp4" for video, "svg" for individual SVG files. Defaults to "mp4".
    """
    length = len(task_x)
    max_weight = max(weights.max()) if weights is not None else 1
    min_weight = min(weights.min()) if weights is not None else 0

    # mp4 case VideoWriter
    writer = None
    temp_fig_path = None
    if output_fmt == "mp4":
        # first frame create get
        temp_fig_path = os.path.join(savePath, "temp_frame.png")
    else:
        # svg case create directory
        if not os.path.exists(os.path.join(savePath, "figures_3D")):
            os.mkdir(os.path.join(savePath, "figures_3D"))

    for i in tqdm(range(min(weights.shape[0], length)), leave=False):
        data_idx = i + start_frame
        # create
        G = nx.Graph()
        edges = Graph(skl_type).neighbor_link  # nodes append
        pos = {}
        for j in range(len(edges)):
            G.add_node(j, label=j)
            pos[j] = (task_x.iloc[i, j], task_y.iloc[i, j])

        # edges append(skeleton-like structure)
        G.add_edges_from(edges)

        # node size and colormap set
        sizes = np.ones(len(edges)+1)  # default size
        colors = np.ones(len(edges)+1)  # default color
        if weights is not None:
            sizes = weights.iloc[i].values
            if max_weight - min_weight < 1e-6:
                colors = np.zeros(len(edges)+1)
            else:
                colors = (sizes - min_weight) / (max_weight - min_weight)

        # draw the graph
        fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={
            'projection': '3d'})
        ax.axis("off")

        ax.scatter(task_x.iloc[i, :], task_z.iloc[i, :],
                   task_y.iloc[i, :], c=colors, s=250, cmap='Reds', edgecolors='black')
        sm = plt.cm.ScalarMappable(
            cmap="Reds", norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
        sm.set_array([])

        cax = fig.add_axes([0.65, 0.2, 0.02, 0.5])
        cbar = plt.colorbar(sm, cax=cax, fraction=0.02)

        # cbar = plt.colorbar(sm, ax=ax)
        cbar.set_ticks([min_weight, max_weight])
        cbar.ax.set_yticklabels([f"{min_weight:.4f}", f"{max_weight:.4f}"])
        # cbar.ax.set_yticklabels([min_weight, max_weight])
        cbar.ax.tick_params(labelsize=20)

        for edge in edges:
            ax.plot([task_x.iloc[i, edge[0]], task_x.iloc[i, edge[1]]],
                    [task_z.iloc[i, edge[0]], task_z.iloc[i, edge[1]]],
                    [task_y.iloc[i, edge[0]], task_y.iloc[i, edge[1]]],
                    c='gray')
        # ax.plot([-1, 1], [0, 0], [0, 0], "r-")
        # ax.plot([0, 0], [2.5, 3.5], [0, 0], "r-")
        # ax.plot([0, 0], [0, 0], [-1, 1], "r-")
        ax.view_init(elev=30, azim=-60)
        ax.set_xlim(-0.75, 0.75)
        ax.set_ylim(-0.75, 0.75)
        ax.set_zlim(-1, 0.5)
        ax.invert_xaxis()

        # fig.tight_layout()

        if output_fmt == "mp4":
            # mp4 case temporary file saveVideoWriter append
            fig.savefig(temp_fig_path, format='png', dpi=300)

            # first frame VideoWriter initialize
            if writer is None:
                img = cv2.imread(temp_fig_path)
                height, width, layers = img.shape[:3]
                codec = cv2.VideoWriter_fourcc(*"mp4v")
                # savePath parent directory save
                parent_dir = os.path.dirname(savePath)
                video_name = os.path.basename(savePath) + "_3D.mp4"
                video_path = os.path.join(parent_dir, video_name)
                writer = cv2.VideoWriter(
                    video_path, codec, 30, (width, height), True)

            # add frame
            img = cv2.imread(temp_fig_path)
            writer.write(img)
        else:
            # svg case individual file as save
            fig.savefig(os.path.join(savePath, "figures_3D",
                        "frame{:03}_3d.{}".format(data_idx, output_fmt)))

        plt.close()

    # mp4 case release VideoWriterdelete temporary file
    if output_fmt == "mp4" and writer is not None:
        writer.release()
        if temp_fig_path and os.path.exists(temp_fig_path):
            os.remove(temp_fig_path)
        parent_dir = os.path.dirname(savePath)
        video_name = os.path.basename(savePath) + "_3D.mp4"
        print(f"Saved video: {os.path.join(parent_dir, video_name)}")
