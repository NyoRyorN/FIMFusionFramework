import numpy as np
import os
import glob
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.utils.names import Tasks, FIMitems, SimpleJoints
from src.data.dataset.utils import dataframeToNumpy


def makeSingleDataset(seed,
                      config,
                      action_task,
                      fim_item,
                      target_path):
    """
    single actiondataset createfunction

    specifyaction task from FIM item to
    /evaluatedataset generate.
    K-fold cross-validation or holdout validation,
    split by class distribution.

    Args:
        seed (int): seed value(for reproducibility)
        config (dict): experiment config dictionary
        action_task (str): input action task
        fim_item (str): target FIM item
        target_path (str): datasetdestination path

    Processing details:
        1. set load
        2. data directory and file
        3. beforeprocessing and
        4. FIM score based on generate
        5. split by /split
        6. datasetsave and output
    """

    # config file from get
    # split, beforeprocessing, model settings experiment config extract
    centering = config["preprocess"]["norm_centering_mode"]  # normalization centering mode
    attributes = config["attributes"]                    # data attributes(/velocity)
    classes = config["classes"]                         # class-definition list
    frame_len = config["preprocess"]["frame_length"]    # frame length(time-series window size)
    is_moving_frame = config["preprocess"]["use_moving_frame"]  # moving frameset
    center_point = config["preprocess"]["norm_center_point"]    # normalization center point
    data_dir_list_path = config["data_list_path"]       # data directory listpath
    is_fill_short = config["preprocess"]["is_fill_short"]      # short data complementset
    is_fill_actionless = config["preprocess"]["is_fill_actionless"]  # nonecomplementset

    # FIM scorerelated
    max_class_len = 7    # FIM score maximum number of classes(1-7points)
    len_classes = len(classes)  # number of classes

    # name managementobject initialize
    tasks = Tasks()      # action task
    items = FIMitems()   # FIM-item name
    joints = SimpleJoints()  # ConciseData always holds the reduced skeleton

    # input
    if action_task not in tasks:
        print("Invalid action_task.")  # disabledaction task
        exit()

    if fim_item not in items:
        print("Invalid fim_item.")     # invalid FIM item name
        exit()

    # data source is always the reduced-joint ConciseData; centering is
    # applied later at load time via norm_centering_mode ("First")
    data_dir = os.path.join("data", "ConciseData")

    # output directorycreate
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # FIMdictionaryinitialize
    # each subject,,
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

    # each subject load and beforeprocessing
    # subjectdirectory, and FIM processing
    for dir in dirs:
        # FIM scorefile() search
        label_path = glob.glob(data_dir + "/" + dir + "/FIM.csv")
        # data directory pathbuild
        temp_path = os.path.join(data_dir, dir, action_task)

        # data directory check
        if not os.path.exists(temp_path):
            print(f"{dir}'s data is not found.")
            continue

        # FIMfile processing
        if len(label_path) == 1:
            # FIM scoreCSV file load
            fim_df = pd.read_csv(label_path[0], header=0, index_col=0)
            fim_score = fim_df.at[fim_item, "Score"]  # specifyFIM item get

            # invalid FIM score exclude
            if fim_score <= 0 or np.isnan(fim_score):
                print(f"{dir}'s FIM label is not found.")
                continue

            # FIM score(1-7points) class label(0-based) convert
            # :7class->3classset case, 5-7points 2-4points
            label = int(fim_score) - (max_class_len + 1 - len_classes)
            if label < 0:  # negative values minimum class(0) set
                label = 0

            # beforeprocessing and conversion to NumPy arrays
            # time-series data split,, attribute extraction run
            data_temp, window_start   \
                = dataframeToNumpy(root_path=temp_path,      # data root path
                                   # task=action_task, # action task
                                   frame_len=frame_len,      # frame length
                                   attributes=attributes,    # data attributes(/velocity)
                                   is_fill_short=is_fill_short,      # short data complementset
                                   is_moving=is_moving_frame,        # moving frameset
                                   centering_mode=centering,         # normalization centering mode
                                   center_point=joints.index(center_point))  # center-point index

            # subject integrate
            fim_datas[dir] = {
                "label": label,              # converted class label
                "fim_score": fim_score,      # original FIM score
                "datas": data_temp,          # beforeprocessing
                "window_start": window_start  # window start frameinformation
            }

            # invalid data(data that is too short) exclude
            if fim_datas[dir]["datas"] is None:
                if is_fill_actionless:
                    # none complementprocessing
                    temp_data = np.zeros(
                        (frame_len, len(attributes) * 3), dtype=np.float32)
                    fim_datas[dir]["datas"] = temp_data
                    fim_datas[dir]["window_start"] = 0  # frame 0 set
                else:
                    print(f"{dir} is shorter than the frame length not found.")
                    del fim_datas[dir]
                    continue
        else:
            # FIMfile from case
            print(f"{dir} is label file not found.")
            continue

    # valid data subject listget
    target_dirs = list(fim_datas.keys())
    # all subjects label arraycreate
    labels = np.array([fim_datas[dir]["label"] for dir in target_dirs])

    # all subjects columncreate
    datas_all = []
    for dir in target_dirs:
        datas_all.append(fim_datas[dir]["datas"])
    datas_all = np.array(datas_all)

    # output directorycreate
    if not os.path.exists(target_path+"/"):
        os.makedirs(target_path+"/")

    # K-fold cross-validation:5split by datasetcreate
    # split subject
    sgkf = StratifiedGroupKFold(
        n_splits=5,           # 5split
        shuffle=True,         # enabled
        random_state=seed)    # for reproducibility
    # subjectID by split(before subjectID as)
    target_groups = np.array([t.split("_")[0] for t in target_dirs])
    target_dirs = np.array(target_dirs, dtype=str)  # columncolumn as convert

    # each fold /split
    for fold, (train_index, test_index) in enumerate(
            sgkf.split(datas_all, labels, target_groups)):

        # training data saveprocessing(per-fold)
        train_path = os.path.join(
            target_path, "Fold" + str(fold))  # per-folddirectory
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        train_path = os.path.join(train_path, "train")  # training datadirectory
        if not os.path.exists(train_path):
            os.makedirs(train_path)

        # training data class distributioncreate
        values, counts = np.unique(labels[train_index], return_counts=True)
        df = pd.DataFrame(counts, index=classes)  # class
        df.loc["Total"] = len(labels[train_index])  # Translated note
        # CSV filesave
        df.to_csv(os.path.join(train_path, "train_distribution.csv"))

        # training datacolumn save
        np.save(os.path.join(train_path, "data.npy"),
                datas_all[train_index])    # save
        np.save(os.path.join(train_path, "target.npy"),
                labels[train_index])       # save

        # filecreate
        dir_list = pd.DataFrame(
            target_dirs[train_index], columns=["Directory"])  # directory list
        window_starts = [fim_datas[dir]["window_start"]
                         for dir in target_dirs[train_index]]  # window start frame
        dir_list["FrameStart"] = window_starts  # frame-start informationappend
        dir_list["RealScore"] = [fim_datas[dir]["fim_score"]  # original FIM scoreinformationappend
                                 for dir in target_dirs[train_index]]
        dir_list.to_csv(os.path.join(
            train_path, "trainDir.csv"), index=False)  # CSVsave

        # test data saveprocessing(per-fold)
        test_path = os.path.join(
            target_path, "Fold" + str(fold), "test")  # test datapath
        if not os.path.exists(test_path):
            os.makedirs(os.path.join(test_path))  # data directorycreate

        # test data class distributioncreate
        values, counts = np.unique(
            labels[test_index], return_counts=True)
        df = pd.DataFrame(counts, index=classes)  # class
        df.loc["Total"] = len(target_dirs[test_index])  # Translated note
        # CSV filesave
        df.to_csv(os.path.join(test_path, "test_distribution.csv"))

        # test datacolumn save
        np.save(os.path.join(test_path, "data.npy"),
                datas_all[test_index])     # save
        np.save(os.path.join(test_path, "target.npy"),
                labels[test_index])        # save

        # filecreate
        dir_list = pd.DataFrame(
            target_dirs[test_index], columns=["Directory"])  # directory list
        window_starts = [fim_datas[dir]["window_start"]
                         for dir in target_dirs[test_index]]  # window start frame
        dir_list["FrameStart"] = window_starts  # frame-start informationappend
        dir_list["RealScore"] = [fim_datas[dir]["fim_score"]  # original FIM scoreinformationappend
                                 for dir in target_dirs[test_index]]
        # CSV file save
        dir_list.to_csv(os.path.join(
            test_path, "testDir.csv"), index=False)  # CSVsave

        print("Seed ", seed, " Fold ", fold, " Data Saved.")  # Foldprocessing completedlog
