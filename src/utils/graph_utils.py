import random
import itertools
import networkx as nx
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pandas as pd
import numpy as np
import operator


def sample_graph_by_edges(g, num_of_edges):
    sampled_edges = random.sample(g.edges, num_of_edges)
    sampled_nodes = set(itertools.chain(*sampled_edges))
    g_sampled = nx.DiGraph(g).subgraph(sampled_nodes)
    return g_sampled


def sample_graph_by_nodes(g, num_of_nodes):
    sampled_nodes = random.sample(g.nodes, num_of_nodes)
    return nx.DiGraph(g).subgraph(sampled_nodes)


def partition_graph_by_lonlat(g):
    """
    We partition the graph using geographical data
    Args:
        g (JURBEY): the graph

    Returns:
       networkx.DiGraph: the partition graph
    """
    selected_nodes = list()
    for node in g.nodes():
        coord = g.nodes[node]['data'].coord
        if _is_inside_selected_area(lon=coord.lon, lat=coord.lat):
            selected_nodes.append(node)
    return nx.DiGraph(g).subgraph(selected_nodes)


def _is_inside_selected_area(lon, lat):
    """
    Check if the point (lon, lat) is in the predefined area. Hard-assigned the area
    to be an area in Charlottenburg, Berlin, with approx. 6000 edges.
    Args:
        lon (double): longitude of the interested point
        lat (double): latitude of the interested point

    Returns:
       Bool: whether the point is in the area
    """
    top_left = [13.292427062988281, 52.50556442091497]
    bottom_left = [13.332939147949219, 52.50556442091497]
    bottom_right = [13.332939147949219, 52.52144366674759]
    top_right = [13.292427062988281, 52.52144366674759]

    point = Point(lon, lat)
    area = Polygon((top_left, bottom_left, bottom_right, top_right))
    return area.contains(point)


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
    return (tmp[0]*nb_col + tmp[1]).to_dict()


