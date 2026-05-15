import lynxes as lx
import numpy as np
import pyarrow as pa
from src.ingest import convert_to_tensor, create_lynxes_node_frame
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
    node_features = lx.NodeFrame.from_dict({
        "_id": [str(i) for i in range(10)],
        "_label": [["Node"]]*10,
        "feat1": [float(i) for i in range(10)],
        "feat2": [float(i + 10) for i in range(10)],
    })
    node_labels = lx.NodeFrame.from_dict({
        "_id": [str(i) for i in range(10)],
        "_label": [["Node"]]*10,
        "label": [i for i in range(10)]
    })

    class FakeDb:
        def __init__(self):
            self.calls = []
            self.queries = []

        def neighbor_loader(
            self,
            input_nodes,
            fanouts,
            edge_types,
            *,
            batch_size,
            shuffle=True,
            filter=None,
            warm_start=False,
            num_workers=0,
            return_format="pyg",
        ):
            self.calls.append(
                (input_nodes, tuple(fanouts), tuple(edge_types), batch_size, filter, warm_start, num_workers)
            )
            seeds = np.arange(batch_size, dtype=np.int64)
            unique_nodes = np.unique(np.concatenate([seeds, (seeds + 1) % 10]))
            edge_index = np.vstack([seeds, (seeds + 1) % 10]).astype(np.int64)
            yield edge_index, unique_nodes

    db = FakeDb()
    sampler = iter(CaracalSampler(db, node_features, node_labels, [2, 1], 4))
    x, y, edge_index, sample_time = next(sampler)

    assert x.shape[1] == 2
    assert y.shape[0] == 4
    assert edge_index.shape == (2, 4)
    assert sample_time >= 0
    assert db.calls[0] == (None, (2, 1), ("CITES",), 4, None, False, 0)


def test_sampler_routes_scenarios_to_official_loader():
    node_features = lx.NodeFrame.from_dict({
        "_id": [str(i) for i in range(10)],
        "_label": [["Node"]]*10,
        "feat1": [float(i) for i in range(10)],
    })
    node_labels = lx.NodeFrame.from_dict({
        "_id": [str(i) for i in range(10)],
        "_label": [["Node"]]*10,
        "label": [i for i in range(10)]
    })
    node_years = lx.NodeFrame.from_dict({
        "_id": [str(i) for i in range(10)],
        "_label": [["Node"]]*10,
        "year": [2010, 2011, 2011, 2012, 2013, 2014, 2014, 2015, 2016, 2017]
    })

    class FakeDb:
        def __init__(self):
            self.calls = []

        def sql(self, query):
            self.calls.append(("sql", query))
            return self

        def arrow(self):
            return pa.Table.from_pydict({"node_id": np.array([0, 1], dtype=np.int64)})

        def neighbor_loader(self, input_nodes, fanouts, edge_types, **kwargs):
            self.calls.append(
                (
                    "loader",
                    tuple(input_nodes.tolist()) if hasattr(input_nodes, "tolist") else input_nodes,
                    kwargs["filter"],
                    kwargs["warm_start"],
                    kwargs["num_workers"],
                )
            )
            yield np.array([[0], [1]], dtype=np.int64), np.array([0, 1], dtype=np.int64)

    db = FakeDb()
    next(iter(CaracalSampler(db, node_features, node_labels, [2], 2, scenario="filtered", filter_data=node_years, filter_year=2014)))
    next(iter(CaracalSampler(db, node_features, node_labels, [2], 2, scenario="warm_start")))
    next(iter(CaracalSampler(db, node_features, node_labels, [2], 2, scenario="multi_worker", num_workers=8)))

    assert db.calls == [
        ("sql", "MATCH (n:Node) WHERE n.year >= 2014 RETURN n.node_id AS node_id"),
        ("loader", (0, 1), None, False, 0),
        ("loader", None, None, True, 0),
        ("loader", None, None, False, 8),
    ]


def test_create_lynxes_node_frame_accepts_empty_arrow_table():
    table = pa.Table.from_arrays(
        [pa.array([], type=pa.int64()), pa.array([], type=pa.int64())],
        names=["src", "dst"],
    )

    frame = create_lynxes_node_frame(table, "Edge")
    assert frame.len() == 0
