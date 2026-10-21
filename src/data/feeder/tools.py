"""beforeprocessing

PyTorchDataLoader skeletonconvert/.
for.

Key features:
-: column
-: time-series data
- convert: rowmove/
- temporal: enabledframe adjust

:
- 3Dskeleton (x, y, z) convert
- frame
- by Data augmentation effects

Usage example:
    >>> padded_data = auto_padding(skeleton_data, target_size=100)
    >>> augmented_data = random_move(skeleton_data)
"""

import random
import numpy as np


def auto_padding(data_numpy, size, random_pad=False):
    """time-series data

    specifycolumn zero padding.
    batchprocessing for.

    Args:
        data_numpy (np.ndarray): input data (C, T, V, M)
            C: number of channels, T: temporalframe, V: points, M:
        size (int): temporalframe
        random_pad (bool):
            True: (Data augmentation effects)
            False: beginning (freeze/fix)

    Returns:
        np.ndarray: (C, size, V, M)
            case: zero paddingapply
            or case: return

    processing:
    1. input case processing
    2. resolve(random_pad=True case)
    3. columncreate/
    """
    C, T, V, M = data_numpy.shape
    if T < size:
        # resolve
        begin = random.randint(0, size - T) if random_pad else 0
        # zero paddingcolumn create
        data_numpy_paded = np.zeros((C, size, V, M))
        # Translated note
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    """time-series data

    specify column extract.
    and calculate.

    Args:
        data_numpy (np.ndarray): input data (C, T, V, M)
        size (int): extracttemporalframe
        auto_pad (bool): case enabled

    Returns:
        np.ndarray: (C, size, V, M)

    processing:
    - T == size: return
    - T < size: auto_pad=True, False
    - T > size: from specify

    Data augmentation effects:
    - column from
    - temporal
    """
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_padding(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        # resolve
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy: np.ndarray,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1],
                rotation_axis=[0, 1, 2]):
    """3Dskeleton convert

    /rowmove by.
    and.

    Args:
        data_numpy (np.ndarray): input data (C, T, V)
            C: number of channels(x,y,z), T: temporalframe, V: points
        angle_candidate (list): angle(current)
        scale_candidate (list): convert [0.9, 1.0, 1.1]
        transform_candidate (list): rowmove [-0.2, -0.1, 0.0, 0.1, 0.2]
        move_time_candidate (list): convert
        rotation_axis (list): (current)

    Returns:
        np.ndarray: convert (C, T, V)
            3D for /rowmove apply

    convert:
    1. temporal multiple split
    2. each /move select
    3. linear interpolation convert
    4. frame 3Dconvert apply

    processing:
    - rowcolumn: diag(s, s, s)
    - rowmove: (t_x, t_y, t_z)
    - convert: new_xyz = S * xyz + T
    """
    # input: C,T,V
    C, T, V = data_numpy.shape
    # convert resolve
    move_time = random.choice(move_time_candidate)
    # axis = random.choice(rotation_axis)

    # temporalsplitnodes generate
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    # eachnodes convert select
    # A = np.random.choice(angle_candidate, num_node) # angle()
    S = np.random.choice(scale_candidate, num_node)          # Translated note
    T_x = np.random.choice(transform_candidate, num_node)    # x to rowmove
    T_y = np.random.choice(transform_candidate, num_node)    # y to rowmove
    T_z = np.random.choice(transform_candidate, num_node)    # z to rowmove

    # eachframe convertcolumn initialize
    # a = np.zeros(T) # anglecolumn()
    s = np.zeros(T)    # column
    t_x = np.zeros(T)  # xrowmovecolumn
    t_y = np.zeros(T)  # yrowmovecolumn
    t_z = np.zeros(T)  # zrowmovecolumn

    # nodeslinear interpolation by convertgenerate
    for i in range(num_node - 1):
        # each linear interpolation apply
        # a[node[i]:node[i + 1]] = np.linspace(A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]
          ] = np.linspace(S[i], S[i + 1], node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]
            ] = np.linspace(T_x[i], T_x[i + 1], node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]
            ] = np.linspace(T_y[i], T_y[i + 1], node[i + 1] - node[i])
        t_z[node[i]:node[i + 1]
            ] = np.linspace(T_z[i], T_z[i + 1], node[i + 1] - node[i])

    # convertrowcolumn generate
    # theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
    # [np.sin(a) * s, np.cos(a) * s]]) # ()
    theta = np.zeros((T, 3, 3))
    for i_frame in range(T):
        # 3Drowcolumn
        theta[i_frame] = [[s[i_frame], 0, 0],
                          [0, s[i_frame], 0],
                          [0, 0, s[i_frame]]]  # Translated note

    # rowcolumn adjust: (T,3,3) -> (3,3,T)
    theta = np.transpose(theta, (1, 2, 0))

    # each processing(current commented out)
    # r_ax = [0, 1]
    # if axis == 0:
    #     r_ax = [1, 2]
    # elif axis == 1:
    #     r_ax = [2, 0]
    # elif axis == 2:
    #     r_ax = [0, 1]
    # else:
    #     print("Invalid axis")

    # frame 3Dconvertrun
    for i_frame in range(T):
        # 2Dconvertprocessing(current commented out)
        # xyz = data_numpy[r_ax, i_frame, :]
        # new_xyz = np.dot(theta[:, :, i_frame], xyz.reshape(2, -1))
        # if axis == 0:
        # new_xyz[0] += t_y[i_frame] # y to rowmove
        # new_xyz[1] += t_z[i_frame] # z to rowmove
        # elif axis == 1:
        # new_xyz[0] += t_z[i_frame] # z to rowmove
        # new_xyz[1] += t_x[i_frame] # x to rowmove
        # elif axis == 2:
        # new_xyz[0] += t_x[i_frame] # x to rowmove
        # new_xyz[1] += t_y[i_frame] # y to rowmove
        # data_numpy[r_ax, i_frame, :] = new_xyz.reshape(2, V)

        # 3Dconvertprocessing
        xyz = data_numpy[:3, i_frame, :]  # (3, V) - x,y,zget
        # convert apply
        xyz = np.dot(theta[:, :, i_frame], xyz.reshape(3, -1))
        # rowmove apply
        xyz[0] += t_x[i_frame]  # xmove
        xyz[1] += t_y[i_frame]  # ymove
        xyz[2] += t_z[i_frame]  # zmove
        # convert
        data_numpy[:3, i_frame, :] = xyz

    return data_numpy


def random_shift(data_numpy):
    """temporal

    enabledframe move.
    temporal for.

    Args:
        data_numpy (np.ndarray): input data (C, T, V, M)

    Returns:
        np.ndarray: (C, T, V, M)
            enabledframe temporal

    Processing steps:
    1. enabledframe ()
    2. enabled calculate
    3. resolve
    4. enabled to

    Data augmentation effects:
    -
    - temporal
    """
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    # column initialize
    data_shift = np.zeros(data_numpy.shape)

    # enabledframe (allchannels/points/)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0

    # enabled /frame
    begin = valid_frame.argmax()  # enabledframe
    end = len(valid_frame) - valid_frame[::-1].argmax()  # after enabledframe

    # enabled calculate
    size = end - begin
    # resolve
    bias = random.randint(0, T - size)
    # enabled to
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift
