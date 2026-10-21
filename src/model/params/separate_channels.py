"""
fusion channelscalculatemodule

fusion, GCN and fusion structure
fusion_positionbased on.shallow/deepfusionaccording to,
output channel list split, each structure resolve.

Key features:
- output channel list analyze and duplicateschannels
- fusion(shallow/deep) split
- GCN and fusion channelssetgenerate

fusion:
- shallow: fusion(integrate)
- deep: fusion(integrate)
"""


def separate_channels(out_channel_list: list[int], fusion_position: str) -> tuple[list[int], list[int]]:
    """fusion based on channelsprocessing

    specifyoutput channel list, fusionaccording to
    GCN and fusion split.

    Args:
        out_channel_list (list[int]): all output channelslist
                                     : [32, 64, 64, 128, 128, 256]
        fusion_position (str): fusionrun specify
            - "shallow": fusion()
            - "deep": fusion()

    Returns:
        tuple[list[int], list[int]]: (GCNchannelslist, fusionchannelslist)

    Raises:
        ValueError: fusion_position 'shallow' or 'deep' case

    processing:
    1. duplicatesnumber of channels analyze and value extract
    2. fusion splitpoints resolve
    3. GCN and fusion channelssetgenerate

    Example:
        >>> separate_channels([32, 64, 64, 128], "shallow")
        ([32], [64, 64, 128])

        >>> separate_channels([32, 64, 128, 128], "deep")
        ([32, 64], [128, 128])
    """
    # duplicates channelsvalue order extract
    # sorted(..., key=out_channel_list.index) by order
    unique_channels = sorted(
        list(set(out_channel_list)), key=out_channel_list.index)

    # variable initialize
    gcn_channels = None
    fusion_channels = None

    # gcn_config = config
    # fusion_config = config

    if fusion_position == "shallow":
        # fusion: GCN, fusion
        if len(unique_channels) > 1:
            # channels multiplecase: value GCN
            gcn_channels = [
                c for c in out_channel_list if c == unique_channels[0]]
            fusion_channels = out_channel_list[len(gcn_channels):]
        else:
            # channels 1type case:all 1/3 GCN
            gcn_channels = out_channel_list[:len(out_channel_list)//3 + 1]
            fusion_channels = out_channel_list[len(gcn_channels):]

    elif fusion_position == "deep":
        # fusion: fusion, 〜intermediate layer GCN
        if len(unique_channels) > 1:
            # channels multiplecase:after value GCN
            gcn_channels = [
                c for c in out_channel_list if c != unique_channels[-1]]
            fusion_channels = out_channel_list[len(gcn_channels):]
        else:
            # channels 1type case:before2/3 GCN, after1/3 fusion
            split_point = len(out_channel_list) - len(out_channel_list)//3
            gcn_channels = out_channel_list[:split_point]
            fusion_channels = out_channel_list[split_point:]
    else:
        # disabledfusion specifycase processing
        raise ValueError(
            "Invalid fusion layer position. Must be 'deep' or 'shallow'.")

    return gcn_channels, fusion_channels
