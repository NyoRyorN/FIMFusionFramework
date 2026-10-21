# DataFrame from NumPycolumnconvert
# CSV file load, NumPycolumn convert
# , data lengthadjust, beforeprocessing integrate

import numpy as np
import pandas as pd
import os
from src.data.dataset.utils.detect_movement_onset import detectMovementOnset


def dataframeToNumpy(root_path: str,
                     frame_len: int,
                     attributes: list[str],
                     is_fill_short: bool = False,
                     is_moving: bool = False,
                     centering_mode: str = None,
                     center_point: int = 0) -> np.ndarray:
    """CSV file from load, NumPycolumn convertfunction

    multiple (x, y, z) and data attributes(position, velocity, acceleration) integrate,
    format convert., data lengthadjust,
    beforeprocessing run.

    Args:
        root_path (str): directory path
        frame_len (int): output target frame length
        attributes (list[str]): getdata attributes list("p":, "v":velocity, "a":acceleration)
        is_fill_short (bool): data length zero paddingenabledflag(default: False)
        is_moving (bool): row flag(default: False)
        centering_mode (str): ("First", "Each", None)
        center_point (int): points(default: 0)

    Returns:
        tuple[np.ndarray, int]: (convertNumPycolumn, frame)
        processing case (None, None) return

    Note:
        - ("ing") and processing apply
        - in velocity from frame automatic detection
        - by value
        - data lengthadjust by frame length
    """

    # frame value(in 0)
    window_start = 0
    # convertafter list
    data = []
    # specify
    if not set(attributes).issubset(set(["p", "v", "a"])):
        print("Invalid attribute.")
        exit()

    # Translated note
    if centering_mode not in ["First", "Each", None]:
        centering_mode = None

    # and processing(directory "ing" case)
    if os.path.basename(root_path)[-3:] == "ing":
        # specifyeach(, velocity, acceleration) processing
        for attribute in attributes:
            # each coordinate axis(x, y, z) process data separately
            for i in ["x", "y", "z"]:
                # CSV file load(:p_x.csv, v_y.csv)
                df = pd.read_csv(os.path.join(
                    root_path, attribute + "_" + i + ".csv"), header=0)
                # data length target frame lengthcase processing
                if len(df) < frame_len:
                    if not is_fill_short:
                        # zero padding when disabled abort processing
                        return None, None
                    else:
                        # zero padding frame length adjust
                        temp_data = np.zeros(
                            (frame_len, df.shape[1]), dtype=np.float32)
                        temp_data[:len(df), :] = df.to_numpy()
                        data.append(temp_data)
                else:
                    # data length case end from frame_len get
                    window_start = max(len(df) - int(frame_len), 0)
                    df = df.iloc[-frame_len:, :]
                    # position data (First case)
                    if centering_mode == "First" and attribute == "p":
                        df = df - df.iloc[0, center_point]
                    data.append(np.array(df.to_numpy()))
        # print(window_start)

    # () processing
    else:
        # frame processing
        # Detect the start frame
        if is_moving:
            # 3coordinate dimensions
            shaft = ["x", "y", "z"]
            # velocity integrate to DataFrameinitialize
            v_df = pd.DataFrame()
            times_windows = None
            # each coordinate axis velocity integrateprocessing
            for i in shaft:
                # velocityCSV file pathbuild
                v_path = os.path.join(root_path, "v_" + i + ".csv")

                # velocity load, and spine extract
                temp_df = pd.read_csv(v_path, header=0)
                temp_df = temp_df[[
                    c for c in temp_df.columns if "Shoulder" in c or "Spine" in c]]
                # velocity calculate(calculate)
                temp_df = temp_df.map(lambda x: x ** 2)
                # processing, and later
                if v_df.empty:
                    v_df = temp_df
                else:
                    v_df = v_df.add(temp_df, fill_value=0)
            # eachpoints velocity calculate,
            times_windows = v_df.map(
                lambda x: np.sqrt(x)
            ).mean(axis=1)
            # row in value (velocity 1.0 case 0.01)
            if "Walk" not in os.path.basename(root_path):
                times_windows = times_windows.map(
                    lambda x: x if x < 1.0 else 0.01
                )

            # run
            window_start = detectMovementOnset(
                times_windows.to_numpy(),
                frame_len=frame_len
            )

        # specifyeach processing
        for attribute in attributes:
            # each coordinate axis(x, y, z) process data separately
            for i in ["x", "y", "z"]:
                # CSV file load
                df = pd.read_csv(os.path.join(
                    root_path, attribute + "_" + i + ".csv"), header=0)
                # data length target frame lengthcase processing
                if len(df) < frame_len:
                    if not is_fill_short:
                        # zero padding when disabled abort processing
                        return None, None
                    else:
                        # position data (short data case)
                        if centering_mode == "First" and attribute == "p":
                            df = df - df.iloc[0, center_point]
                        # zero padding frame length adjust
                        temp_data = np.zeros(
                            (frame_len, df.shape[1]), dtype=np.float32)
                        temp_data[:len(df), :] = df.to_numpy()
                        data.append(temp_data)
                else:
                    # frame from specify extract
                    df = df.iloc[window_start: window_start + frame_len, :]
                    # position data (First case)
                    if centering_mode == "First" and attribute == "p":
                        df = df - df.iloc[0, center_point]
                    data.append(np.array(df.to_numpy()))

    # processing NumPycolumn as return(column and frame)
    return np.array(data), window_start
