# frame
# from automatic detection to module
# velocity and information, points

from typing import Optional
import numpy as np


def detectMovementOnset(
        times_windows: np.ndarray,
        frame_len: int = 150,
        window_size: int = 50) -> Optional[int]:
    """ from frame automatic detectionfunction

    velocity and by,.
    mean value frame and, value combination
    points.

    Parameters:
        times_windows (np.ndarray): columnvelocity(1column)
        frame_len (int): after frame(default: 150)
        window_size (int): move(default: 50)

    Returns:
        Optional[int]: frame(None processing)

    Note:
        - velocity by,
        - mean value and valueprocessing
        - information by points resolve
        - end from all processing
    """
    # input data get
    length = len(times_windows)

    # move by velocity
    # , velocity
    smooth_windows = np.convolve(
        times_windows, np.ones(window_size) / window_size, mode='same')

    # velocity mean value calculate
    # value as
    windows_mean = smooth_windows.mean()
    # mean value points value and column
    above_mean_velo = smooth_windows > windows_mean
    # specifymean value
    consecutive_frames_velo = np.convolve(
        above_mean_velo, np.ones(window_size), mode='same') == window_size

    # valueset(mean value 1.25%)
    threshold = windows_mean * 0.0125  # 1/80

    # velocity (gradients) calculate
    # velocity from
    times_windows_diff = np.gradient(smooth_windows)

    # value value points
    # velocity points(points)
    zero_crossings = np.where(
        np.diff(np.sign(times_windows_diff - threshold)) != 0)[0]

    # frame valueset
    first_point = 0
    # velocityframe case processing
    if np.any(consecutive_frames_velo):
        # frame points calculate(compensate)
        first_point = np.argmax(
            consecutive_frames_velo) - (window_size - 1) / 2
        # points, framebefore case
        if len(zero_crossings) > 0:
            # framebefore after points select
            valid_zero_crossings = zero_crossings[zero_crossings < first_point]
            if len(valid_zero_crossings) > 0:
                first_point = valid_zero_crossings[-1]
            else:
                # points case frame0 from
                first_point = 0
        else:
            # points allcase
            first_point = 0

    # frame from frame case adjust
    if length - first_point < frame_len:
        # end from framebefore adjust
        first_point = length - frame_len
    # negative values case 0frame from
    if first_point < 0:
        first_point = 0

    # frame return
    return first_point
