def build_loc_net(struc, columns: list):

    index_feature_map = columns
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
                print(f"error: {child} not in index_feature_map")

            c_index = index_feature_map.index(child)
            edge_indexes[0].append(c_index)
            edge_indexes[1].append(p_index)

    return edge_indexes
