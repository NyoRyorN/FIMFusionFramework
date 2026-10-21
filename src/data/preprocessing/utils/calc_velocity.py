"""
velocitycalculate

time-series data from velocity calculate.
temporal by frame calculate.

Key features:
- temporalcolumn check
- framecalculate
- temporal velocitycalculate
- value 0processing
"""

import pandas as pd


def CalcVelocity(df: pd.DataFrame) -> pd.DataFrame:
    """time-series data velocitycalculate

    frame temporal velocity calculate.
    all for processing run.

    Args:
        df (DataFrame): columnposition data(Timecolumn)

    Returns:
        DataFrame: velocity(Timecolumn)

    Processing details:
    1. Timecolumn check
    2. Framecolumn (case)
    3. calculate(diff)
    4. temporal by velocitycalculate
    5. frame 0
    """

    # Timecolumn check()
    if "Time" not in df.columns:
        print("Error: Time column is not found.")
        return None

    # Framecolumn (case)
    if "Frame" in df.columns:
        df = df.drop(columns="Frame")

    # frame calculate
    df_v = df.diff()

    # temporal velocitycalculate(Timecolumn exclude)
    df_v = df_v.div(df_v["Time"], axis=0).drop(columns="Time")

    # frame(NaN) 0
    df_v = df_v.fillna(0)

    return df_v
