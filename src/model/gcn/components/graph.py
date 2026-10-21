"""
row for graph structuremodule

human skeleton joints connection relationships,
adjacency matrix generate. The joint layout is fixed to the reduced
17-joint skeleton produced by the preprocessing pipeline;
split(uniform, distance, spatial) select.

class:
    Graph: graph structure and adjacency matrixgenerate

References:
    Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition
    https://arxiv.org/abs/1801.07455
"""

import numpy as np


class Graph():
    """ graph structureclass

    joints points, and build,
    ST-GCN adjacency matrix generate.split by,
    and extract adjust.

    Args:
        strategy (str): split select
            - uniform: (all points class split)
            - distance: split(from based on split)
            - spatial: spatialstructure(based on split)
        max_hop (int): ()
        dilation (int): points

    information:
        split "Partition Strategies"
        https://arxiv.org/abs/1801.07455
    """

    def __init__(self,
                 strategy='uniform',
                 norm_mode='DA',
                 max_hop=1,
                 dilation=1):
        """ initialize and edges build"""
        self.max_hop = max_hop
        self.dilation = dilation
        self.norm_mode = norm_mode
        if norm_mode not in ['DAD', 'AD', 'DA']:
            print("Invalid norm_mode. Choose either 'DAD' or 'AD/DA'.")
            norm_mode = 'DA'

        # edges
        self.get_edge()
        # eachjoints () calculate
        self.hop_dis = get_hop_distance(self.num_node,
                                        self.edge)
        # select based on adjacency matrix generate
        self.get_adjacency(strategy)

    def __str__(self):
        """ column as adjacency matrix return"""
        return self.A

    def get_edge(self):
        """ edges

        each jointsconnection relationships,
        self.edge(list) generate.
        The joint layout is fixed to the reduced 17-joint skeleton
        (formerly "simple_ntu-rgb+d"), which is what the preprocessing
        pipeline produces.
        """
        # edge [child, parent] list

        # NTU RGB+D 17jointsstructure
        self.num_node = 17
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_1base = [(0, 1),   # spine base - spine mid
                          (2, 16),  # neck - spine shoulder
                          (3, 2),   # head - neck
                          (4, 16),  # left shoulder - spine shoulder
                          (5, 4),   # left elbow - left shoulder
                          (6, 5),   # left wrist - left elbow
                          (7, 16),  # right shoulder - spine shoulder
                          (8, 7),   # right elbow - right shoulder
                          (9, 8),   # right wrist - right elbow
                          (10, 0),  # left hip - spine base
                          (11, 10),  # left knee - left hip
                          (12, 11),  # left ankle - left knee
                          (13, 0),  # right hip - spine base
                          (14, 13),  # right knee - right hip
                          (15, 14),  # right ankle - right knee
                          (16, 1)]  # spine shoulder - spine mid
        self.neighbor_link = [(i, j) for (i, j) in neighbor_1base]
        self.edge = self_link + self.neighbor_link
        self.center = 1  # center point as set


    def get_adj_for_cal_angle(self, joints):
        """anglecalculate get

        anglecalculate for, joints list as return.
        append.

        Args:
            joints: joints

        Returns:
            list: list(list)
        """
        self.neighbor_link.append((self.center, self.center))
        bone_lists = []
        for i in self.neighbor_link:
            bone_lists.append([joints[i[0]], joints[i[1]]])
        return bone_lists

    def get_adjacency(self, strategy):
        """select based on adjacency matrix generate

        splitaccording to, format adjacency matrix generate.
        each ST-GCN based on.

        Args:
            strategy (str): split
                - uniform: adjacency matrix
                - distance: multipleadjacency matrix
                - spatial: spatialstructure multipleadjacency matrix
        """
        # enabled calculate(dilation by adjust)
        valid_hop = range(0, self.max_hop + 1, self.dilation)

        # adjacency matrix build
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1

        # adjacency matrix (as processing)
        # normalize_adjacency = normalize_digraph(adjacency)
        normalize_adjacency = normalize_undigraph(
            adjacency) if self.norm_mode == 'DAD' else normalize_digraph(adjacency)

        # adjacency matrix()
        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        # multipleadjacency matrix
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        # spatialstructuremultipleadjacency matrix
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                # 3type spatialrowcolumn initialize
                a_root = np.zeros(
                    (self.num_node, self.num_node))      # Translated note
                # Translated note
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros(
                    (self.num_node, self.num_node))   # from

                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            # from by split
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                # from
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                # j i from ()
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                # j i ()
                                a_further[j, i] = normalize_adjacency[j, i]

                if hop == 0:
                    # case rowcolumn
                    A.append(a_root)
                else:
                    # case and
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge):
    """ allnodes calculate

    eachjoints () calculate.
    adjacency matrix k-hop.

    Args:
        num_node (int): nodes(joints)
        edge (list): edgeslist(list)

    Returns:
        numpy.ndarray: rowcolumn (num_node, num_node)
                      [i, j]element i from j to
    """
    # adjacency matrix build(as processing)
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1  # edges (i, j) append
        A[i, j] = 1  # set

    # rowcolumn initialize()
    hop_dis = np.zeros((num_node, num_node)) + np.inf

    # each calculate
    # A^d d-hop nodes
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(num_node + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)

    # from reverse order set
    # ,
    for d in range(num_node, -1, -1):
        hop_dis[arrive_mat[d]] = d

    return hop_dis


def normalize_digraph(A):
    """ adjacency matrix

    column row, probabilityrowcolumn create.
    eachcolumn 1.

    Args:
        A (numpy.ndarray): inputadjacency matrix

    Returns:
        numpy.ndarray: adjacency matrix(ADformat)
    """
    # eachnodes (column) calculate
    Dl = np.sum(A, 0)
    num_node = A.shape[0]

    # by rowcolumn build
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)  # Translated note

    # A * D^(-1) by column
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    """ adjacency matrix

    D^(-1/2) * A * D^(-1/2) by run.
    format.

    Args:
        A (numpy.ndarray): inputadjacency matrix()

    Returns:
        numpy.ndarray: adjacency matrix(DADformat)
    """
    # eachnodes () calculate
    Dl = np.sum(A, 0)
    num_node = A.shape[0]

    # by rowcolumn build
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)  # Translated note

    # D^(-1/2) * A * D^(-1/2) by
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
