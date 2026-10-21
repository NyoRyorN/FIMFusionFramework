import matplotlib.pyplot as plt
import networkx as nx
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.model.gcn.components.graph import Graph
import cv2


def Plot2D(savePath, task_x, task_y, weights: pd.DataFrame = None, skl_type: str = "ntu-rgb+d", start_frame: int = 0, output_fmt: str = "mp4"):
    """Plot 2D graph

    Args:
        savePath (str): Path to save the plot
        task_x (pd.DataFrame): x-axis data
        task_y (pd.DataFrame): y-axis data
        weights (pd.DataFrame, optional): Node weights for sizes and colors. Defaults to None.
        start_frame (int, optional): Starting frame index for the data. Defaults to 0.
        output_fmt (str, optional): Output format. "mp4" for video, "svg" for individual files. Defaults to "mp4".
    """
    length = len(weights) if weights is not None else len(task_x)
    max_weight = max(weights.max()) if weights is not None else 1
    min_weight = min(weights.min()) if weights is not None else 0

    # mp4 case VideoWriter
    writer = None
    temp_fig_path = None
    if output_fmt == "mp4":
        temp_fig_path = os.path.join(savePath, "temp_frame.png")
    else:
        # svg case create directory
        if not os.path.exists(os.path.join(savePath, "figures_2D")):
            os.makedirs(os.path.join(savePath, "figures_2D"))

    for i in tqdm(range(min(weights.shape[0], length)), leave=False):
        data_idx = i + start_frame
        # create
        G = nx.Graph()
        edges = Graph(skl_type).neighbor_link

        # nodes append
        pos = {}
        for j in range(len(edges)+1):
            G.add_node(j, label=j)
            pos[j] = (task_x.iloc[i, j], task_y.iloc[i, j])

        # edges append(skeleton-like structure)
        G.add_edges_from(edges)

        # node size and colormap set
        sizes = np.ones(len(edges)+1)  # default size
        colors = np.ones(len(edges)+1)  # default color
        if weights is not None:
            sizes = weights.iloc[i].values
            colors = (sizes - min_weight) / (max_weight - min_weight)

        # draw the graph
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.invert_xaxis()

        nx.draw_networkx_edges(G, pos, edge_color='black', ax=ax)
        nodes = nx.draw_networkx_nodes(
            G, pos, node_color=colors, cmap="Reds", ax=ax, edgecolors='black')

        # append
        sm = plt.cm.ScalarMappable(
            cmap="Reds", norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
        sm.set_array([])  # Translated note
        cbar = plt.colorbar(sm, ax=ax)

        cbar.set_ticks([min_weight, max_weight])
        cbar.ax.set_yticklabels([min_weight, max_weight])
        cbar.ax.tick_params(labelsize=20)

        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

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
                video_name = os.path.basename(savePath) + "_2D.mp4"
                video_path = os.path.join(parent_dir, video_name)
                writer = cv2.VideoWriter(
                    video_path, codec, 30, (width, height), True)

            # add frame
            img = cv2.imread(temp_fig_path)
            writer.write(img)
        else:
            # svg case individual file as save
            name = "frame{:03}.{}".format(data_idx, output_fmt)
            plt.savefig(os.path.join(savePath, "figures_2D", name))

        plt.close()

    # mp4 case release VideoWriterdelete temporary file
    if output_fmt == "mp4" and writer is not None:
        writer.release()
        if temp_fig_path and os.path.exists(temp_fig_path):
            os.remove(temp_fig_path)
        parent_dir = os.path.dirname(savePath)
        video_name = os.path.basename(savePath) + "_2D.mp4"
        print(f"Saved video: {os.path.join(parent_dir, video_name)}")
