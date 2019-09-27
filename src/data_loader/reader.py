import tempfile
from jurbey.jurbey import JURBEY
from src.config import Config
import networkx as nx

from src.utils.graph_utils import get_berlin_graph

cfg = Config()


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
    mapping = dict()
    with open('cluster-mapping.csv') as f:
        r = csv.reader(f)
        for row in r:
            v, k = row
            mapping.setdefault(int(k), []).append(float(v))
    return mapping


def get_adj_from_subgraph(g, cluster_id, mapping):
    print(f"cluster_id: {cluster_id}")
    nodes = mapping.get(cluster_id)
    L = nx.line_graph(nx.DiGraph(g).subgraph(nodes))
    print(f"number of nodes/segments: {len(L.nodes)}")
    adj = nx.to_scipy_sparse_matrix(L, format="coo")

    return adj, L
