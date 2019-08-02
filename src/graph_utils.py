import random
import itertools
import networkx as nx


def sample_graph_by_edges(g, num_of_edges):
    sampled_edges = random.sample(g.edges, num_of_edges)
    sampled_nodes = set(itertools.chain(*sampled_edges))
    g_sampled = nx.DiGraph(g).subgraph(sampled_nodes)
    return g_sampled


def sample_graph_by_nodes(g, num_of_nodes):
    sampled_nodes = random.sample(g.nodes, num_of_nodes)
    return nx.DiGraph(g).subgraph(sampled_nodes)

