from src.data_loader.traffic_dataset import TrafficDataset
from jurbey.jurbey import JURBEY
import pytest

@pytest.fixture
def dataset():
    g = None
    with open('data/1558537930325.jurbey', 'rb') as f:
        g = JURBEY.load(f.read())

    return TrafficDataset(g)

def test_shape(dataset):
    features, target, mask = next(iter(dataset))
    assert features.size(0) == 99
    assert features.size(2) == 30
    assert mask.size(1) == features.size(2)
    assert features.size(1) == target.size(0)
    assert mask.size(0) == target.size(0)
