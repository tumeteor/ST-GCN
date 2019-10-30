import pytest
from fastparquet import ParquetFile
import networkx as nx
from src.data_loader.datasets import DatasetBuilder


@pytest.fixture
def segment_graph():
    g = nx.DiGraph()
    g.add_edge(151139635, 4898034404)
    g.add_edge(3992457487, 316186499)
    g.add_edge(66616871, 3869622341)

    class RoadClass(object):
        name = 'residential'

    class Data(object):
        metadata = {'highway': 'LocalRoad'}
        roadClass = RoadClass()

    data = Data()
    nx.set_node_attributes(g, data, 'data')
    nx.set_edge_attributes(g, data, 'data')
    return g


@pytest.fixture
def speed_df():
    pf = ParquetFile("tests/data/clusters/test.snappy.parquet")
    df = pf.to_pandas()
    return df[0:3]


def test_construct_batches(segment_graph, speed_df):
    db = DatasetBuilder(segment_graph)
    data, target, mask = db.construct_batches(df=speed_df, L=nx.line_graph(nx.DiGraph(segment_graph)))
    assert all(k in data for k in ["train", "valid", "test"]), "wrong keys in data"
    assert all(k in target for k in ["train", "valid", "test"]), "wrong keys in target"
    assert all(k in mask for k in ["train", "valid", "test"]), "wrong keys in mask"

    gt_speed = [23.56613254547119, 13.8, 13.8]
    gt_mask = [0, 0, 0]
    assert all([speed == gt_speed for speed, gt_speed in zip(data["train"][0][0][0].tolist()[0:3], gt_speed)]), \
        "wrong speed features"
    assert all([speed == 13.8 for speed in target["train"][0][0]]), "wrong labels"
    assert all([m == gt_m for m, gt_m in zip(mask["train"][0], gt_mask)]), "wrong masks"
    assert data["train"].shape == (251, 3, 53, 29), "wrong shape"







