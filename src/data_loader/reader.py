import tempfile
from collections import defaultdict

from src.module import DATACONFIG_GETTER
from jurbey.jurbey import JURBEY
from src.configs.db_config import Config
import networkx as nx

from src.utils.graph_utils import get_berlin_graph

cfg = Config()
datacfg = DATACONFIG_GETTER()


def read_jurbey_from_minio(bucket_name, object_name):
    from src.module import MINIO_CLIENT

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tempf:
        MINIO_CLIENT.fget_object(bucket_name, object_name, tempf.name)

    with open(tempf.name, 'rb') as tempf:
        g = JURBEY.load(tempf.read())

    return g


def read_jurbey(f='data/1558537930325.jurbey'):
    with open(f, 'rb') as tempf:
        g = JURBEY.load(tempf.read())
        return get_berlin_graph(g)


def read_cluster_mapping():
    import csv
    mapping = defaultdict(list)
    with open(datacfg['cluster_mapping']) as f:
        r = csv.reader(f)
        for row in r:
            v, k = row
            mapping[k].append(float(v))
    return mapping


def get_adj_from_subgraph(g, cluster_id, edges):
    print(f"cluster_id: {cluster_id}, number of edges: {len(edges)}")

    L = nx.line_graph(nx.DiGraph(g)).subgraph(edges)

    adj = nx.to_scipy_sparse_matrix(L, format="coo", nodelist=edges)
    print(f"number of edges: {L.number_of_nodes()}")

    return adj, L
