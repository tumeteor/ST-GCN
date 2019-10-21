import pandas as pd
import numpy as np


def get_berlin_graph(g, berlin_lat=52, berlin_lon=13):
    """
    Get only the berlin area from JURBEY
    Args:
        g (JURBEY):
        berlin_lat (int): the rounded Berlin lat
        berlin_lon (int): the rounded Berlin lon

    Returns:

    """
    nodes = list(g.nodes())
    for node in nodes:
        coord = g.nodes[node]['data'].coord
        if abs(coord.lat - berlin_lat) > 1 or abs(coord.lon - berlin_lon) > 1:
            g.remove_node(node)
    return g


def get_node_coord(g, node):
    """Fetch coordonates of a node using the map

    Args:
        g (JURBEY): The map
        node (int): Node id

    Returns:
        (float, float): Coordinates tuple

    """
    try:
        from_coord = g.nodes[node]['data'].coord
        from_coord_lat = from_coord.lat
        from_coord_lon = from_coord.lon
    except KeyError:
        from_coord_lat = float('nan')
        from_coord_lon = float('nan')
    return from_coord_lat, from_coord_lon


def get_bounding_box(g, nb_cols, nb_rows):
    nodes = list(g.nodes)
    coords = [get_node_coord(g, node) for node in nodes]
    (min_lat, min_lon), (max_lat, max_lon) = (min(x for x, y in coords), min(y for x, y in coords)),\
                                             (max(x for x, y in coords), max(y for x, y in coords))
    return np.linspace(min_lat, max_lat, nb_cols), np.linspace(min_lon, max_lon, nb_rows)


def get_cluster_index(g, lat_grid, lon_grid, nb_col):
    """Compute the cluster ID

    Number of columns in the grid is used here to determine the ID. It first
    find in which lat / lon bucket the node belongs to using
    :func:`np.searchsorted` with the np.linspace buckets.

    Args:
        g (JURBEY): the map
        lat_grid (np.linspace): Latitude buckets
        lon_grid (np.linspace): Longitude buckets

    Returns:
        int: The cluster ID of the node

    """
    nodes = list(g.nodes)
    coords = [get_node_coord(g, node) for node in nodes]
    tmp = pd.DataFrame(np.searchsorted(lat_grid, [coord[0] for coord in coords]))
    tmp[1] = np.searchsorted(lon_grid, [coord[1] for coord in coords])
    return tmp[0] * nb_col + tmp[1]


