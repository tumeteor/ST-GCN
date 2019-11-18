from src.data_loader.traffic_dataset import (
            TrafficDataset,
            series_one_hot,
            get_encoder
        )
from jurbey.jurbey import JURBEY
from itertools import product
import pytest
import torch

@pytest.fixture
def dataset():
    g = None
    with open('data/1558537930325.jurbey', 'rb') as f:
        g = JURBEY.load(f.read())

    return TrafficDataset(g)

# def test_shape(dataset):
    # features, target, mask = next(iter(dataset))
    # assert features.size(0) == 99
    # assert features.size(2) == 30
    # assert mask.size(1) == features.size(2)
    # assert features.size(1) == target.size(0)
    # assert mask.size(0) == target.size(0)

def test_get_encoder():
    assert {(i,): i for i in range(10)} == get_encoder(range(10))
    assert {(i, 'a'): i for i in range(3)} == get_encoder(range(3), ['a'])

def test_series_one_hot():
    tensor = torch.zeros(3, 64, 128)
    idxs = torch.ones(1, 64, 128)
    tensor = torch.cat((tensor, idxs), dim=0)
    one_hot = series_one_hot(tensor, num_feats=4)
    assert one_hot.size() == (7, 64, 128)
    target = torch.zeros(*one_hot.size())
    target[4, :, :] = 1.
    assert (one_hot == target).all()
    assert one_hot.sum().item() == 64. * 128.
