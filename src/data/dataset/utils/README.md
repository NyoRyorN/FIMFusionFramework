# src/data/dataset/utils — Dataset-build helpers

Low-level helpers used while assembling HDF5 datasets: locating the start of the movement
in a sequence and converting per-trial CSV DataFrames into fixed-length NumPy arrays.

## Files

| File | Role |
|---|---|
| `detect_movement_onset.py` | Detects the movement-onset frame from a velocity/time window so sequences are cropped to the active motion. |
| `dataframe_to_numpy.py` | Loads trial CSVs, converts them to NumPy arrays, and adjusts sequence length (padding/cropping) for a uniform data length. |
