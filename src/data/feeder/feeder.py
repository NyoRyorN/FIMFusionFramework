"""PyTorchDataLoaderclass

FIMevaluate for skeletonload/beforeprocessing.
.

Key features:
- skeleton/ load
- apply
- PyTorch Dataset
- information

format:
- input: 3Dskeletontime-series data
- output: FIMevaluate()
-: convert by

Usage example:
    >>> dataset = Feeder('data.npy', 'labels.npy', 'dirs.csv', use_transform=True)
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
"""

import torch
import numpy as np
import pandas as pd
import random
from src.data.feeder.tools import random_move
# load function


class Feeder(torch.utils.data.Dataset):
    """FIMevaluatePyTorchdataset

    skeleton from evaluate to
    DataLoaderclass./beforeprocessing integrate.

    Attributes:
        data (np.ndarray): skeleton (N, C, T, V, M)
            N: number of samples, C: number of channels, T: temporalframe
            V: points, M:
        label (np.ndarray): FIMevaluate (N,)
        dirs (list): data directorypathlist
        real_score (list): evaluate(value)
        use_transform (bool): applyflag

    :
    - NaNvalue/processing
    -
    - convert
    """

    def __init__(self,
                 data_path,
                 label_path,
                 dir_path,
                 use_transform=False):
        """datasetinitialize

        Args:
            data_path (str): skeletonfilepath (.npy)
            label_path (str): filepath (.npy)
            dir_path (str): directory informationfilepath (.csv)
            use_transform (bool): applyflag
                True: convert by apply
                False:

        Processing details:
        1. //directory information load
        2. information get(case)
        3. (NaN)
        4. convertset save
        """
        super().__init__()
        # load
        self.label = np.load(label_path)
        # skeleton load
        self.data = np.load(data_path)

        # directory information load
        df = pd.read_csv(dir_path)
        self.dirs = df["Directory"].tolist()

        # information get(case)
        if "RealScore" in df.columns:
            self.real_score = df["RealScore"].tolist()

        # set save
        self.use_transform = use_transform

        # ------------------------------------------------------------------
        # Derive per-sample per-action availability mask (1=present, 0=missing)
        # ------------------------------------------------------------------
        # Missing actions are stored as all-zero (C, T, V) slices by
        # makeSomeTasksDataset, so an all-zero slice <=> truly missing.
        if self.data.ndim == 5:  # fusion data (N, A, C, T, V)
            N, A = self.data.shape[:2]
            self.mask = (np.abs(self.data).reshape(N, A, -1).sum(axis=2)
                         > 0).astype(np.float32)
        else:  # single-action data (N, C, T, V)
            self.mask = np.ones((len(self.data), 1), dtype=np.float32)

        # (NaNvalue)
        if np.any(np.isnan(self.data)):
            print("Data contains NaN.")
            print(np.where(np.isnan(self.data)))
            exit()

        # class distributioncalculate()
        # self.labelDist = np.bincount(self.label)

    def __len__(self):
        """dataset get

        Returns:
            int: number of samples
        """
        return len(self.label)

    def __iter__(self):
        """

        Returns:
            Feeder: return
        """
        return self

    def __getitem__(self, index):
        """ get

        PyTorch DataLoader from method.
        /beforeprocessing allget.

        Args:
            index (int):

        Returns:
            tuple: (data, label, dir, real_score, mask)
                data (np.ndarray): processingskeleton (C, T, V, M)
                label (int): FIMevaluate
                dir (str): data directorypath
                real_score (float or None): evaluate
                mask (np.ndarray): action availability mask (A,);
                    1 = action present, 0 = missing (all-zero slice)

        Processing flow:
        1. specify /get
        2. (directory/)get
        3. apply(probability50%)
        4. processing return
        """
        # get
        data = np.array(self.data[index])
        label = self.label[index]
        dir = self.dirs[index]

        # get(case)
        real_score = None
        if hasattr(self, "real_score"):
            real_score = self.real_score[index]

        # processing(50% probability apply)
        if self.use_transform and random.random() < 0.5:
            data = random_move(data)

        return data, label, dir, real_score, self.mask[index]

    # classcalculate()
    # def getweight(self):
    #     class_weight = 1 / self.labelDist
    #     return [class_weight[self.label[i]] for i in range(len(self.label))]
