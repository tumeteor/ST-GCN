import random
import itertools
import networkx as nx
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def sample_graph_by_edges(g, num_of_edges):
    sampled_edges = random.sample(g.edges, num_of_edges)
    sampled_nodes = set(itertools.chain(*sampled_edges))
    g_sampled = nx.DiGraph(g).subgraph(sampled_nodes)
    return g_sampled


def sample_graph_by_nodes(g, num_of_nodes):
    sampled_nodes = random.sample(g.nodes, num_of_nodes)
    return nx.DiGraph(g).subgraph(sampled_nodes)


def partition_graph_by_lonlat(g):
    selected_nodes = list()
    for node in g.nodes():
        coord = g.nodes[node]['data'].coord
        if _is_inside_selected_area(lon=coord.lon, lat=coord.lat):
            selected_nodes.append(node)
    return nx.DiGraph(g).subgraph(selected_nodes)


def _is_inside_selected_area(lon, lat):
    top_left = [13.292427062988281, 52.50556442091497]
    bottom_left = [13.332939147949219, 52.50556442091497]
    bottom_right = [13.332939147949219, 52.52144366674759]
    top_right = [13.292427062988281, 52.52144366674759]

    point = Point(lon, lat)
    area = Polygon((top_left, bottom_left, bottom_right, top_right))
    return area.contains(point)


