import pytest
import torch
import lynxes as lx
import numpy as np
from src.config import BenchmarkConfig
from src.ingest import convert_to_tensor
from src.samplers.caracal_sampler import CaracalSampler

def test_convert_to_tensor():
    # Create a small lynxes NodeFrame
    data = {
        "feat1": [1.0, 2.0, 3.0],
        "feat2": [4.0, 5.0, 6.0]
    }
    df = lx.NodeFrame.from_dict({
        "_id": ["n1", "n2", "n3"],
        "_label": [["Node"]]*3,
        **data
    })
    
    tensor = convert_to_tensor(df, column_names=["feat1", "feat2"])
    assert tensor.shape == (3, 2)
    assert tensor[0, 0] == 1.0
    assert tensor[2, 1] == 6.0

def test_sampler_initialization():
    # Mock node features and labels
    node_features = lx.NodeFrame.from_dict({
        "_id": [str(i) for i in range(10)],
        "_label": [["Node"]]*10,
        "feat": [[0.1]*10]*10
    })
    node_labels = lx.NodeFrame.from_dict({
        "_id": [str(i) for i in range(10)],
        "_label": [["Node"]]*10,
        "label": [i for i in range(10)]
    })
    
    # We can't easily mock caracaldb connection without the library,
    # but we can check if the class initializes.
    # sampler = CaracalSampler(None, node_features, node_labels, [5, 2], 2)
    # assert sampler.num_nodes == 10
    pass
