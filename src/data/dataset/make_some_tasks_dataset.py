"""
make_some_tasks_dataset.py - multipleintegratedatasetcreatemodule

module, FIM(evaluate) for
multipleaction task integratedataset create.

Key features:
1. multipleaction task integrate by datasetgenerate
2. complement/
3. fusion individualdataset
4. K-fold cross-validation split

Processing flow:
multiple -> integrate/complement -> /split -> individualdatasetgenerate
"""

import numpy as np
import os
import glob
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.utils.names import Tasks, FIMitems, SimpleJoints
from src.data.dataset.utils import dataframeToNumpy


def makeSomeTasksDataset(seed,
                         config,
                         action_tasks,
                         fim_item,
                         target_path):
    """
    multipleintegratedataset createfunction

    specifymultiple action task integrate, FIM item
    dataset generate.
    complement and, individual save.

    Args:
        seed (int): seed value(for reproducibility)
        config (dict): experiment config dictionary
        action_tasks (list[str]): integratetarget action tasklist
        fim_item (str): target FIM item
        target_path (str): datasetdestination path

    Processing details:
        1. multiple load and integrate
        2. complement/processing
        3. K-fold cross-validation split
        4. per actiondataset(fusion)
        5. and
    """

    # config file from get
    centering = config["preprocess"]["norm_centering_mode"]  # normalization centering mode
    attributes = config["attributes"]                    # data attributes(/velocity)
    classes = config["classes"]                         # class-definition list
    frame_len = config["preprocess"]["frame_length"]    # frame length(time-series window size)
    is_moving_frame = config["preprocess"]["use_moving_frame"]  # moving frameset
    center_point = config["preprocess"]["norm_center_point"]    # normalization center point
    data_dir_list_path = config["data_list_path"]       # data directory listpath
    is_fill_short = config["preprocess"]["is_fill_short"]      # short data complementset
    # set
    is_fill_actionless = config["preprocess"]["is_fill_actionless"]

    # action taskobjectinitialize
    tasks = Tasks()

    # fallback chain for missing actions: task index -> ordered list of
    # candidate task indices to try in turn (first existing one wins).
    # Tasks not listed here are never complemented and fall through to
    # zero fill. Default preserves the original seated<->standing reach
    # pairing; config["preprocess"]["complement_map"] overrides it entirely
    # so custom fallback chains (e.g. to a static sitting/standing task)
    # can be defined without touching this file.
    default_complement_map = {
        2: [6],  # SitFR -> StdFR
        3: [7],  # SitDRR -> StdDRR
        4: [8],  # SitDLR -> StdDLR
        6: [2],  # StdFR -> SitFR
        7: [3],  # StdDRR -> SitDRR
        8: [4],  # StdDLR -> SitDLR
    }
    complement_map_cfg = config["preprocess"].get("complement_map")
    if not complement_map_cfg:
        complement_map_cfg = default_complement_map
    for key, candidates in complement_map_cfg.items():
        if int(key) not in range(len(tasks)) or \
                any(int(c) not in range(len(tasks)) for c in candidates):
            raise ValueError(
                "complement_map contains an invalid task index: {} -> {}"
                .format(key, candidates))
    complement_map = {
        tasks[int(task_idx)]: [tasks[int(c)] for c in candidates]
        for task_idx, candidates in complement_map_cfg.items()
    }
    # complementenabledset
    complement_actionless = config["preprocess"]["complement_actionless"]

    # FIM scorerelated
    max_class_len = 7    # FIM score maximum number of classes(1-7points)
    len_classes = len(classes)  # number of classes

    # name managementobject initialize
    items = FIMitems()   # FIM-item name
    joints = SimpleJoints()  # ConciseData always holds the reduced skeleton

    # input
    if fim_item not in items:
        print("Invalid fim_item.")     # invalid FIM item name
        exit()

    # data source is always the reduced-joint ConciseData; centering is
    # applied later at load time via norm_centering_mode ("First")
    data_dir = os.path.join("data", "ConciseData")

    # output directorycreate
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # multipleintegratedictionaryinitialize
    # each subject multiple,,
    fim_datas = dict()

    # data directory list get
    dirs = None
    if os.path.exists(data_dir_list_path) or data_dir_list_path != "":
        # CSV file from specifydirectorylist load
        dir_df = pd.read_csv(data_dir_list_path, header=0)
        dirs = sorted(dir_df["Directory"].tolist())
    else:
        # data directory from automatic detection
        print("Data list file not found.")
        dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(
            os.path.join(data_dir, d)) and "Unused" not in d])

    # each subject load and multipleintegrate
    # subjectdirectory, multiple and FIM integrateprocessing
    for dir in dirs:
        # FIM scorefile() search
        label_path = glob.glob(data_dir + "/" + dir + "/FIM.csv")
        if len(label_path) == 0:
            print(f"{dir} is label file not found.")  # FIMfile
            continue

        # multipleintegratevariableinitialize
        actions_data = np.array([])      # integratecolumn
        window_start_list = []           # window start frameinformationlist
        action_list = []                 # action namelist

        # FIM score load and convert
        fim_df = pd.read_csv(label_path[0], header=0, index_col=0)
        fim_score = fim_df.at[fim_item, "Score"]  # specifyFIM item get

        # invalid FIM score exclude
        if fim_score <= 0 or np.isnan(fim_score):
            print(f"{dir}'s label is invalid.")
            actions_data = None
            continue

        # FIM score(1-7points) class label(0-based) convert
        label = int(fim_score) - (max_class_len + 1 - len_classes)
        if label < 0:  # negative values minimum class(0) set
            label = 0

        # specifymultipleaction task integrateprocessing
        for task in action_tasks:
            # data directory pathbuild
            temp_path = os.path.join(data_dir, dir, task)

            # data directory check and processing
            if not os.path.exists(temp_path):
                if not is_fill_actionless:
                    # disabled:abort processing
                    print(f"{dir}'s {task} is not found.")
                    actions_data = None
                    break
                elif task in complement_map and complement_actionless:
                    # try fallback candidates in priority order; the first
                    # one that exists on disk is used
                    for candidate in complement_map[task]:
                        candidate_path = os.path.join(
                            data_dir, dir, candidate)
                        if os.path.exists(candidate_path):
                            print(f"  => Complement {task} with {candidate}")
                            temp_path = candidate_path
                            break
                if not os.path.exists(temp_path):
                    # complement does not existcase:
                    print(f"{dir}'s {os.path.basename(temp_path)} is not found.")
                    data_temp = np.array(np.zeros(
                        (3 * len(attributes), frame_len, len(joints)), dtype=np.float32))
                    # integratecolumn append
                    if len(actions_data) <= 0:
                        actions_data = np.array([data_temp])  # append
                    else:
                        actions_data = np.append(
                            actions_data, [data_temp], axis=0)  # append
                    window_start_list.append(0)       # window start frame(0freeze/fix)
                    action_list.append("None")        # action name(None)
                    continue

            # action name record
            action_list.append(os.path.basename(temp_path))

            # beforeprocessing and conversion to NumPy arrays
            # time-series data split,, attribute extraction run
            data_temp, window_start   \
                = dataframeToNumpy(root_path=temp_path,      # data root path
                                   frame_len=frame_len,      # frame length
                                   attributes=attributes,    # data attributes(/velocity)
                                   is_moving=is_moving_frame,        # moving frameset
                                   is_fill_short=is_fill_short,      # short data complementset
                                   centering_mode=centering,         # normalization centering mode
                                   center_point=joints.index(center_point))  # center-point index

            # invalid data(data that is too short) exclude
            if data_temp is None:
                print(
                    f"{dir}'s {task} is shorter than the frame length.")
                actions_data = None
                break

            # integratecolumn to append
            if len(actions_data) <= 0:
                actions_data = np.array([data_temp])  # append
            else:
                actions_data = np.append(
                    actions_data, [data_temp], axis=0)  # append
            window_start_list.append(window_start)  # window start frameinformationappend

        # invalid data subject skip
        if actions_data is None:
            continue

        # subject integrate
        fim_datas[dir] = {
            "label": label,              # converted class label
            "fim_score": fim_score,      # original FIM score
            "datas": actions_data,       # integratemultiple
            "window_start": window_start_list,  # each action window start frameinformation
            "actions": action_list       # action namelist
        }
        print(f"{dir} is loaded.")  # subjectloadcompletedlog

    # valid data subject listget
    target_dirs = list(fim_datas.keys())
    # all subjects label arraycreate
    labels = np.array([fim_datas[dir]["label"] for dir in target_dirs])

    # all subjects integratecolumncreate
    datas_all = []
    for dir in target_dirs:
        datas_all.append(fim_datas[dir]["datas"])
    datas_all = np.array(datas_all)

    # output directorycreate
    if not os.path.exists(target_path+"/"):
        os.makedirs(target_path+"/")

    # K-fold cross-validation split run
    # subject by and duplicates StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=seed)
    # subjectID grouping for create(_)
    target_groups = np.array([t.split("_")[0] for t in target_dirs])
    # directorylist columnformat convert
    target_dirs = np.array(target_dirs, dtype=str)
    # 5 Fold cross-validation for /dataset create
    for fold, (train_index, test_index) in enumerate(
            sgkf.split(datas_all, labels, target_groups)):
        # training data saveprocessing
        # Get train data
        # each fold training datasavecreate directory
        train_path = os.path.join(target_path, "Fold" + str(fold))
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        train_path = os.path.join(train_path, "train")
        if not os.path.exists(train_path):
            os.makedirs(train_path)

        # training data class distribution calculate/save
        values, counts = np.unique(labels[train_index], return_counts=True)
        df = pd.DataFrame(counts, index=classes)
        df.loc["Total"] = len(labels[train_index])
        df.to_csv(os.path.join(train_path, "train_distribution.csv"))
        # training data and ().npyformat save
        np.save(os.path.join(train_path, "data.npy"),
                datas_all[train_index])
        np.save(os.path.join(train_path, "target.npy"),
                labels[train_index])
        # directory information listcreate
        dir_list = pd.DataFrame(
            target_dirs[train_index], columns=["Directory"])
        # dynamic frame eachaction task start-position information append
        if is_moving_frame:
            for i in range(len(action_tasks)):
                start_list = []
                for dir in target_dirs[train_index]:
                    start_list.append(fim_datas[dir]["window_start"][i])
                dir_list[action_tasks[i]] = start_list
        # actual FIM scoreinformation append
        dir_list["RealScore"] = [fim_datas[dir]["fim_score"]
                                 for dir in target_dirs[train_index]]
        # eachaction task record
        for i in range(len(action_tasks)):
            dir_list["use_" + action_tasks[i]] = [fim_datas[dir]["actions"][i]
                                                  for dir in target_dirs[train_index]]
        # directory information CSV file as save
        dir_list.to_csv(os.path.join(
            train_path, "trainDir.csv"), index=False)

        # test data saveprocessing
        # Get test data
        # each fold test datasavecreate directory
        test_path = os.path.join(target_path, "Fold" + str(fold), "test")
        if not os.path.exists(test_path):
            os.makedirs(os.path.join(test_path))
        # test data class distribution calculate/save
        values, counts = np.unique(
            labels[test_index], return_counts=True)
        df = pd.DataFrame(counts, index=classes)
        df.loc["Total"] = len(target_dirs[test_index])
        df.to_csv(os.path.join(test_path, "test_distribution.csv"))
        # test data and ().npyformat save
        np.save(os.path.join(test_path, "data.npy"),
                datas_all[test_index])
        np.save(os.path.join(test_path, "target.npy"),
                labels[test_index])
        # test data directory informationlistcreate
        dir_list = pd.DataFrame(
            target_dirs[test_index], columns=["Directory"])
        # dynamic frame eachaction task start-position information append
        if is_moving_frame:
            for i in range(len(action_tasks)):
                start_list = []
                for dir in target_dirs[test_index]:
                    start_list.append(fim_datas[dir]["window_start"][i])
                dir_list[action_tasks[i]] = start_list
        # actual FIM scoreinformation append
        dir_list["RealScore"] = [fim_datas[dir]["fim_score"]
                                 for dir in target_dirs[test_index]]
        # eachaction task record
        for i in range(len(action_tasks)):
            dir_list["use_" + action_tasks[i]] = [fim_datas[dir]["actions"][i]
                                                  for dir in target_dirs[test_index]]
        # test data directory information CSV file as save
        # Save directory list
        dir_list.to_csv(os.path.join(
            test_path, "testDir.csv"), index=False)

        # each fold savecompleted
        print("Seed ", seed, " Fold ", fold, " Data Saved.")
