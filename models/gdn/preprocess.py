import torch


def build_loc_net(struc: dict, columns: list) -> torch.Tensor:
    """
    Build a location network from a given structure and column list.

    This function creates an edge index tensor for a graph based on the provided
    structure and column list. It maps the relationships between nodes (columns)
    as defined in the structure dictionary.

    Args:
        struc (dict): A dictionary representing the structure of the graph.
                      Keys are node names, and values are lists of connected nodes.
        columns (list): A list of column names to be included in the graph.

    Returns:
        torch.Tensor: A tensor of shape (2, num_edges) representing the edge indices
                      of the graph. The first row contains the indices of the source
                      nodes, and the second row contains the indices of the target nodes.

    Note:
        - The function only considers nodes (columns) that are present in the input 'columns' list.
        - The resulting edge indices represent a directed graph where edges point from
          child nodes to parent nodes as defined in the 'struc' dictionary.
    """

    index_feature_map = columns.copy()
    edge_indexes = [[], []]
    for node_name, node_list in struc.items():
        if node_name not in columns:
            continue

        if node_name not in index_feature_map:
            index_feature_map.append(node_name)

        p_index = index_feature_map.index(node_name)
        for child in node_list:
            if child not in columns:
                continue

            if child not in index_feature_map:
                raise ValueError(f"Error: {child} not in index_feature_map")

            c_index = index_feature_map.index(child)
            edge_indexes[0].append(c_index)
            edge_indexes[1].append(p_index)

    edge_indexes = torch.tensor(edge_indexes, dtype=torch.long)
    return edge_indexes
