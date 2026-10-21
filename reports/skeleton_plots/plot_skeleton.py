import pandas as pd
import os
from tqdm import tqdm
from src.utils.names import Joints, SimpleJoints, Tasks
from reports.skeleton_plots.skeleton_renderer_2d import Plot2D
from reports.skeleton_plots.skeleton_renderer_3d import Plot3D
from src.utils.set_plt_font import setPltFont


def makePlots(data_path, data_dir, task, mode, skl_type, is_first_center=False):
    """
    Make plots for the skeleton data
    """
    setPltFont()

    data_x = pd.read_csv(os.path.join(
        data_path, data_dir, task, "p_x.csv"), header=0)
    data_y = pd.read_csv(os.path.join(
        data_path, data_dir, task, "p_y.csv"), header=0)
    data_z = pd.read_csv(os.path.join(
        data_path, data_dir, task, "p_z.csv"), header=0)
    if is_first_center:
        if skl_type == "ntu-rgb+d":
            data_x = data_x - data_x["SpineShoulder"][0]
            data_y = data_y - data_y["SpineShoulder"][0]
            data_z = data_z - data_z["SpineShoulder"][0]
        else:
            data_x = data_x - data_x["SpineMid"][0]
            data_y = data_y - data_y["SpineMid"][0]
            data_z = data_z - data_z["SpineMid"][0]

    tempSavePath = os.path.join(
        "reports", "skeleton_plots", data_path.split("/")[-1])
    if not os.path.exists(tempSavePath):
        os.makedirs(tempSavePath)
    tempSavePath = os.path.join(tempSavePath, data_dir)
    if not os.path.exists(tempSavePath):
        os.makedirs(tempSavePath)
    tempSavePath = os.path.join(tempSavePath, task)
    if not os.path.exists(tempSavePath):
        os.makedirs(tempSavePath)

    # Original Data(TimeComplemented)
    if mode == "Both" or mode == "2D":
        Plot2D(tempSavePath, data_x, data_y, None,
               skl_type=skl_type, output_fmt="png")
    if mode == "Both" or mode == "3D":
        Plot3D(tempSavePath, data_x, data_y, data_z,
               None, skl_type=skl_type, output_fmt="png")


if __name__ == "__main__":
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
    skl_type = input(
        "Enter the skeleton type(ntu-rgb+d or simple_ntu-rgb+d): ")
    if skl_type not in ["ntu-rgb+d", "simple_ntu-rgb+d"]:
        ValueError("Invalid skeleton type")
    else:
        skl_type = "simple_ntu-rgb+d" if skl_type == "simple_ntu-rgb+d" else "ntu-rgb+d"
    center_mode = input(
        "Enter the center mode(normal: n, each_center: e, or first_center:f): ")
    if center_mode not in ["n", "e", "f"]:
        print("Invalid center mode and set to normal")
        center_mode = "n"
    elif center_mode == "n" or center_mode == "f":
        data_path = plot_data_dict[skl_type]["normal"]
    else:  # center_mode == "e"
        data_path = plot_data_dict[skl_type]["each_center"]
    headers = Joints() if skl_type == "ntu-rgb+d" else SimpleJoints()

    all_tasks = Tasks()[:-1]
    mode = input("Enter the mode(2D or 3D or Both): ")
    if mode not in ["2D", "3D", "Both"]:
        print("Invalid mode and set to Both")
        mode = "Both"

    data_dirs = [d for d in os.listdir(data_path) if os.path.isdir(
        os.path.join(data_path, d))]

    for data_dir in tqdm(data_dirs):
        for task in all_tasks:
            if not os.path.exists(os.path.join(data_path, data_dir, task)):
                print("No Data for " + data_dir + "_" + task)
                continue
            makePlots(data_path, data_dir, task, mode,
                      skl_type, is_first_center=(center_mode == "f"))
            print("=============================================")
            print("Data Plotted")
            print("Finished_" + data_dir + "_" + task)
            print("=============================================")
