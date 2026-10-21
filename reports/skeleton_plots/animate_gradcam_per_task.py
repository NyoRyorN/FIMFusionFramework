import pandas as pd
import os
from tqdm import tqdm
from src.utils.names import Joints, SimpleJoints, FIMitems, Tasks
from reports.skeleton_plots.skeleton_renderer_2d import Plot2D
from reports.skeleton_plots.skeleton_renderer_3d import Plot3D
from src.utils.set_plt_font import setPltFont
import h5py

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


def makeWeightPlots(data_path: str,
                    weights: pd.DataFrame,
                    save_path: str,
                    data_dir: str,
                    task_idx: int,
                    task: str,
                    mode: str,
                    skl_type: str,
                    is_each_center: bool = False,
                    start_frame: int = 0,
                    output_fmt: str = "mp4",
                    frame_len: int = 150):
    """
    Make plots for the skeleton data

    Args:
        data_path (str): Path to the data directory.
        save_path (str): Path to the main directory containing weights and other files.
            - Ex-.../08-.../Stratified_Seed.../
        data_dir (str): Directory name of the data.
        task_idx (int): Index of the task.
        task (str): Name of the task.
        item (str): Item name.
        seed (str): Seed value.
        mode (str): Mode of the plot (2D, 3D, or Both).
        skl_type (str): Type of the skeleton (ntu-rgb+d or simple_ntu-rgb+d).
        is_each_center (bool): Whether to use each centering of the skeleton.
        start_frame (int): Starting frame for the plot.
        output_fmt (str): Output format of the plot (mp4 or svg).
        frame_len (int): Length of the frame to be plotted.
    """
    setPltFont()
    if os.path.exists(os.path.join(data_path, data_dir, task)) is False:
        print("No data: ", os.path.join(data_path, data_dir, task))
        return

    data_x = pd.read_csv(os.path.join(
        data_path, data_dir, task, "p_x.csv"), header=0)
    data_y = pd.read_csv(os.path.join(
        data_path, data_dir, task, "p_y.csv"), header=0)
    data_z = pd.read_csv(os.path.join(
        data_path, data_dir, task, "p_z.csv"), header=0)

    # (T, V)
    weights = weights.T if weights.shape[0] < weights.shape[1] else weights
    # (T, V) columns are joint names
    new_weights = pd.DataFrame(columns=weights.columns)

    # Att weights length is 38 (default)
    if weights.shape[0] < frame_len:
        # Repeat each row 4 times
        new_weights = weights.loc[weights.index.repeat(
            4)].reset_index(drop=True)
        # Truncate to frame_len if necessary
        new_weights = new_weights.iloc[len(
            new_weights) // 2 - frame_len // 2:len(new_weights) // 2 + frame_len // 2]
    # CAM weights length is 150 (default)
    else:
        new_weights = weights

    data_x = data_x.iloc[start_frame:(start_frame + len(new_weights)), :]
    data_y = data_y.iloc[start_frame:(start_frame + len(new_weights)), :]
    data_z = data_z.iloc[start_frame:(start_frame + len(new_weights)), :]

    if not is_each_center:
        if skl_type == "ntu-rgb+d":
            data_x = data_x - data_x.at[data_x.index[0], "SpineShoulder"]
            data_y = data_y - data_y.at[data_y.index[0], "SpineShoulder"]
            data_z = data_z - data_z.at[data_z.index[0], "SpineShoulder"]
        else:
            data_x = data_x - data_x.at[data_x.index[0], "SpineMid"]
            data_y = data_y - data_y.at[data_y.index[0], "SpineMid"]
            data_z = data_z - data_z.at[data_z.index[0], "SpineMid"]

    tempSavePath = os.path.join(save_path, data_dir)
    if not os.path.exists(tempSavePath):
        os.makedirs(tempSavePath)
    tempSavePath = os.path.join(
        tempSavePath, "{:0>2d}_{}".format(task_idx if type(task_idx) is int else int(task_idx), task))
    if not os.path.exists(tempSavePath):
        os.makedirs(tempSavePath)

    # Original Data(TimeComplemented)
    if mode == "Both" or mode == "2D":
        Plot2D(tempSavePath, data_x, data_y, new_weights,
               skl_type=skl_type, start_frame=start_frame, output_fmt=output_fmt)
    if mode == "Both" or mode == "3D":
        Plot3D(tempSavePath, data_x, data_y, data_z, new_weights,
               skl_type=skl_type, start_frame=start_frame, output_fmt=output_fmt)


def makePlotsFusion(rootpath: str,
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

    weights_shifts = ["", "_Att", "_CAM"]

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

            index = 0
            for data_dir in data_dirs:
                temp_data = f[data_dir]
                for task_idx in temp_data.keys():
                    task = all_tasks[int(task_idx)]
                    start_frame = data_df[data_df["Directory"]
                                          == data_dir][task].values[0]
                    print(index, " / ", len(data_dirs),
                          " - ", data_dir, " - ", task)
                    weights = pd.DataFrame(
                        temp_data[task_idx if type(task_idx) == str else str(task_idx)])
                    if weights.shape[0] > weights.shape[1]:
                        weights = weights.T
                    weights.index = headers
                    save_path = os.path.join(rootpath,  "Weights" + shift)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_path = os.path.join(
                        save_path, f"{item}_Seed{seed}") if fold_idx is None else os.path.join(
                        save_path, f"{item}_Seed{seed}_{fold_idx}")
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    makeWeightPlots(
                        data_path=data_path,
                        weights=weights,
                        data_dir=data_dir,
                        save_path=save_path,
                        task_idx=task_idx,
                        task=task,
                        # item=item,
                        mode=mode,
                        skl_type=skl_type,
                        is_each_center=is_center,
                        start_frame=start_frame,
                        output_fmt=output_fmt,
                        frame_len=150)
                index += 1
                print("=============================================")
                print("Data Plotted")
                print("Finished_" + data_dir)
                print("=============================================")
            print("All Data Plotted in " + shift)


def makePlotsSingle(rootpath: str,
                    skl_type: str,
                    is_center: bool,
                    headers: list[str],
                    data_path: str):
    """
    Make plots for the skeleton data without fusion
    Args:
        rootpath (str): Path to the main directory containing weights and other files.
            - Ex-.../08-.../Stratified_Seed.../
        skl_type (str): Type of the skeleton (ntu-rgb+d or simple_ntu-rgb+d).
        is_center (bool): Whether to use each centering of the skeleton.
        headers (list[str]): List of joint names.
        data_path (str): Path to the data directory.
    """
    task_idx = int(input("Enter the task(1--11): "))
    task = ""
    if task_idx not in range(1, 12):
        exit("Invalid task")
    else:
        task = all_tasks[task_idx-1]
    item = int(input("Enter the item(0--12): "))
    if item not in range(0, 13):
        exit("Invalid item")
    else:
        item = "{:0>2d}-{}".format(item, FIMitems()[item])
    seed = input("Enter the seed(100): ")
    fold_idx = "Fold"
    if "kfold" in rootpath:
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
        rootpath, task + "_to_" + item, "Stratified_Seed" + seed)
    if fold_idx is not None:
        main_path = os.path.join(main_path, fold_idx)

    weights_shifts = ["", "_Att", "_CAM"]
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

            index = 0
            for data_dir in tqdm(data_dirs):
                print(index, " / ", len(data_dirs))
                start_frame = data_df["FrameStart"][data_df["Directory"]
                                                    == data_dir].values[0]
                index += 1
                obj = f[data_dir]
                # skip if this entry is a Dataset or contains 'Spatial'
                for key in obj.keys() if hasattr(obj, "keys") else []:
                    if isinstance(obj[key], tuple):
                        print(f"Skipping: obj[{key}] is a tuple")
                        exit()
                weights = pd.DataFrame(f[data_dir], index=headers)
                save_path = os.path.join(rootpath,  "Weights" + shift)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_path = os.path.join(
                    save_path, f"{task}_to_{item}_Seed{seed}") if fold_idx is None else os.path.join(
                        save_path, f"{task}_to_{item}_Seed{seed}_{fold_idx}")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                makeWeightPlots(
                    data_path=data_path,
                    weights=weights,
                    data_dir=data_dir,
                    save_path=save_path,
                    task_idx=task_idx,
                    task=task,
                    mode=mode,
                    skl_type=skl_type,
                    is_each_center=is_center,
                    start_frame=start_frame,
                    output_fmt=output_fmt,
                    frame_len=150)

            print("=============================================")
            print("Data Plotted")
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

    if "multi-model" in rootpath or "Step" in rootpath:
        makePlotsFusion(rootpath=rootpath,
                        skl_type=skl_type,
                        is_center=is_center,
                        headers=headers,
                        data_path=data_path)
    else:
        makePlotsSingle(rootpath=rootpath,
                        skl_type=skl_type,
                        is_center=is_center,
                        headers=headers,
                        data_path=data_path)
