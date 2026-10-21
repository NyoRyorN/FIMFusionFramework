"""
angle/bone lengthcalculate

from angle and bone length calculate.
graph structurebased onparentchild, features generate.

Key features:
- by get
- parentchild calculate
- anglecalculate()
- bone lengthcalculate()
- (X, Y, Z)
"""

import pandas as pd
from src.model.gcn.components import Graph
import math


def CalcAngleLength(df: pd.DataFrame, joints: list) -> pd.DataFrame:
    """ angle/bone lengthcalculate

    graph structurebased on angle and bone length calculate.
    parentchild from features extract.

    Args:
        df (pd.DataFrame): position data(X, Y, Z)
        joints (list): joint-name list

    Returns:
        tuple: (angledata frame, bone lengthdata frame)

    Processing details:
    1. get
    2. edgeslist based on parentchildprocessing
    3. each 3calculate
    4. angle/bone length calculate
    """
    # get(information)
    graph = Graph()
    # Graph has no get_edge_list; get_adj_for_cal_angle returns the
    # [child, parent] bone list (with the center self-link) used here
    edge_list = graph.get_adj_for_cal_angle(joints)

    # column (temporal/frameinformation)
    if "Frame" in df.columns:
        df = df.drop(columns="Frame")

    if "Time" in df.columns:
        df = df.drop(columns="Time")

    # angle/bone length initialize
    df_a = pd.DataFrame()
    df_l = pd.DataFrame()

    # eachedges() processing
    for edge in edge_list:
        child = edge[0]    # child()
        parent = edge[1]   # parent(/points)

        # edges processing()
        if child == parent:
            shaft = ["X", "Y", "Z"]
            for i in shaft:
                # angle 0 initialize
                df_a[edge[0] + "_" + i] = [0] * df.shape[0]
                # bone length 0 initialize; per-axis columns so that callers
                # can select joint_X/Y/Z uniformly for every joint
                df_l[edge[0] + "_" + i] = [0] * df.shape[0]
        else:
            # parentchild 3coordinate dimensionsget
            child_x = df[child + "_X"]
            child_y = df[child + "_Y"]
            child_z = df[child + "_Z"]
            parent_x = df[parent + "_X"]
            parent_y = df[parent + "_Y"]
            parent_z = df[parent + "_Z"]

            # calculate(child-parent)
            bones = pd.DataFrame({
                "X": child_x - parent_x,  # X
                "Y": child_y - parent_y,  # Y
                "Z": child_z - parent_z   # Z
            })

            # calculate()
            norm_bones = (bones["X"] ** 2 + bones["Y"]
                          ** 2 + bones["Z"] ** 2) ** 0.5

            # each anglecalculate()
            shaft = ["X", "Y", "Z"]
            for i in shaft:
                # from anglecalculate
                temp_angle = bones[i] / norm_bones
                # angleconvert
                angle = temp_angle.map(lambda x: math.acos(x))
                df_a[edge[0] + "_" + i] = angle
                # bone lengthsave
                df_l[edge[0] + "_" + i] = bones[i]
            # # all bone lengthsave(commented out)
            # df_l[edge[0]] = norm_bones

    return df_a, df_l
