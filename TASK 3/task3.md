# GNN Data Backend Benchmark System

## 1. Implementation Specification

### 1.1 Project Overview

This project aims to build a benchmark system to evaluate data backend performance for Graph Neural Network (GNN) training pipelines. The system will measure how much the combination of lynxes (DataFrame) and caracaldb (DB) reduces or increases the K-hop sampling and feature fetching bottlenecks compared to existing graph training loops (e.g., In-memory loaders or traditional Graph DBs).

The data processing layer must use lynxes as the DataFrame library, while the topology storage and query layer must use caracaldb.

The purpose of this project is to demonstrate data modeling, graph sampling efficiency, out-of-core scalability, and backend system comparison skills using a standard GraphSAGE model on publicly available datasets.

### 1.2 Dataset

The recommended datasets are from the Open Graph Benchmark (OGB):

```text
ogbn-arxiv (Small)
ogbn-products (Medium)
ogbn-papers100M (Large)
```

The system assumes the following raw inputs (standardized from OGB formats):

```text
nodes.csv / nodes.parquet
edges.csv / edges.parquet
```

The raw data is expected to include:

```text
Node ID
Node features (vector, e.g., 100-128 dimensions)
Node labels (target classes)
Node metadata (e.g., timestamp/year)
Edge Source ID
Edge Destination ID
Edge metadata (e.g., edge type, timestamp)
```

### 1.3 Technology Stack

The system must use the following technologies:

```text
Programming Language: Python
Machine Learning Framework: PyTorch & PyTorch Geometric (PyG)
Database: caracaldb
DataFrame Library: lynxes
Data Source Format: CSV / Parquet
Output Format: CLI output, CSV reports, or JSON reports
```

caracaldb and lynxes are mandatory components. lynxes must be used for the feature processing and tensor conversion layer, and caracaldb must be used for the graph topology storage layer.

### 1.4 System Architecture

The system should follow this pipeline:

```text
1. Load raw OGB data
2. Clean and transform node/edge data
3. Extract graph topology and feature vectors
4. Store topology in caracaldb and features via lynxes
5. Execute GNN K-hop sampling queries
6. Feed data into PyTorch GraphSAGE training loop
7. Generate metric results (TTFB, GPU Starvation, Memory)
8. Produce a real 4x4 comparison benchmark output
```

Overall flow:

```text
OGB Dataset
   ->
lynxes DataFrame
   ->
Topology / Feature Extraction
   ->
caracaldb (Topology) + lynxes (Features)
   ->
GNN Sampler Layer
   ->
PyTorch Training Loop -> Benchmark Output
```

### 1.5 Data Model

The system must include at least the following entities to support graph representation:

```text
Node
Edge
```

Entity meanings:

```text
Node: A vertex in the graph (e.g., Paper, Product) containing feature vectors and labels
Edge: A directed or undirected link between two nodes (e.g., Citation, Co-purchase) containing relationships
```

### 1.6 Entity Schema

Node:

```text
node_id: integer
features: array[float] (or expanded flat columns depending on DB support)
label: integer
timestamp: date or integer (year)
node_type: string
```

Edge:

```text
src_id: integer
dst_id: integer
edge_type: string
timestamp: date or integer
```

### 1.7 Relationship Model

The system must support at least the following topology traversal relationships:

```text
Node - CITES -> Node (ogbn-arxiv / ogbn-papers100M)
Node - BOUGHT_TOGETHER -> Node (ogbn-products)
```

Relationship meanings:

```text
CITES: A paper node cites another paper node
BOUGHT_TOGETHER: A product node is co-purchased with another product node
```

### 1.8 caracaldb Storage Structure

The system should store topology and node features efficiently in caracaldb.

Required storage objects:

```text
nodes
edges
```

nodes:

```text
node_id
features
label
timestamp
node_type
```

edges:

```text
src_id
dst_id
edge_type
timestamp
```

### 1.9 lynxes DataFrame Processing Specification

lynxes must be used for the following tasks:

```text
Load CSV/Parquet files
Handle missing feature values
Filter nodes/edges based on time or type
Perform dynamic feature joins
Format data for caracaldb ingestion
Convert query outputs into PyTorch Tensors
```

Processing rules:

```text
Data loading must use official lynxes read APIs.
The implementation may use lynxes projection options to read only required feature columns.
Node data and edge data must be aligned using node IDs.
Features must be normalized into numerical formats suitable for GNNs.
Relationship tables (edges) must not contain references to non-existing node IDs.
lynxes must handle the zero-copy conversion (or minimal copy) to PyTorch Tensors for the training loop.
```

### 1.10 Main Features

#### Feature 1: K-Hop Neighbor Sampling Bottleneck

The system must evaluate the speed of extracting 1-hop and 2-hop neighborhood topologies for target nodes.

Measurement metrics:

```text
Edges Sampled/sec
Data Wait Ratio
CPU Usage
```

#### Feature 2: Feature Fetching & Tensor Conversion

Given a batch of sampled Node IDs, the system must measure the time to retrieve their feature vectors and convert them to PyTorch tensors.

Measurement metrics:

```text
Feature Fetch Time
Conversion Overhead
```

#### Feature 3: Temporal/Dynamic Graph Filtering

The system must allow sampling on a filtered graph based on edge timestamps or types.

Measurement metrics:

```text
TTFB (Time To First Batch)
Filtered Edges/sec
Epoch Time
```

#### Feature 4: Heterogeneous Graph Join (Feature Generation)

The system must dynamically join features from different node types during the sampling process.

Measurement metrics:

```text
Feature Generation Time
Memory Peak
```

#### Feature 5: Out-of-Core Graph Training

The system must test stability and streaming capability when the graph data significantly exceeds available RAM (e.g., using ogbn-papers100M).

Measurement metrics:

```text
Steady-state Edges/sec
OOM (Out Of Memory) Status
Memory Peak
```

### 1.11 Recommended Module Structure

```text
gnn_benchmark_project/
├── configs/
│   ├── ogbn_arxiv.yaml
│   ├── ogbn_products.yaml
│   └── ogbn_papers100M.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── config.py
│   ├── ingest.py
│   ├── models/
│   │   └── graphsage.py
│   ├── backends/
│   │   ├── base.py
│   │   ├── caracal_lynxes_backend.py
│   │   ├── pyg_native_backend.py
│   │   ├── neo4j_backend.py
│   │   └── duckdb_backend.py
│   ├── samplers/
│   │   ├── caracal_sampler.py
│   │   └── pyg_sampler.py
│   ├── pipeline.py
│   └── compare_systems.py
├── runners/
│   ├── run_khop_sampling.py
│   ├── run_feature_fetch.py
│   ├── run_end_to_end_train.py
│   └── run_out_of_core.py
├── outputs/
│   ├── benchmark_report.txt
│   ├── benchmark_metrics.csv
│   └── hardware_profile.json
├── tests/
│   ├── test_ingest.py
│   └── test_samplers.py
├── README.md
└── main.py
```


### 1.12 Execution

The full pipeline should be executable with:

```bash
uv run main.py --dataset ogbn-arxiv --data-dir data/raw/
```

Optional commands may include:

```bash
python main.py ingest --dataset ogbn-arxiv
python main.py run-khop --backend caracal_lynxes --fanout 15,10
python main.py run-feature-fetch --backend caracal_lynxes
python main.py run-filtered --timestamp 2023-01-01
python main.py run-out-of-core --dataset ogbn-papers100M
python main.py compare
```

### 1.13 Deliverables

The final submission must include:

```text
Source code
README with setup and execution instructions
caracaldb data generation scripts
lynxes-based data transformation code
PyTorch GraphSAGE training loop script
4x4 comparison benchmark report
Comparison benchmark CSV
Basic tests
```

## 2. Requirements

### 2.1 Functional Requirements

#### FR-001. Data Loading

The system shall load the provided OGB graph datasets into lynxes DataFrames.

Acceptance criteria:

```text
The system can read node features, labels, and edge index files.
File paths are provided through configuration or CLI arguments.
If loading fails, the system returns a clear error message.
```

#### FR-002. Data Cleaning

The system shall clean missing or invalid feature/edge values.

Acceptance criteria:

```text
Nodes with corrupted features are handled.
Self-loops or duplicated edges are optionally removed based on config.
Features are cast to appropriate numeric float types.
```

#### FR-003. Graph Extraction

The system shall extract valid node and edge structures required for GNNs.

Acceptance criteria:

```text
The nodes table contains a contiguous mapping of Node IDs.
The edges table stores valid src_id to dst_id mappings.
```

#### FR-004. caracaldb Storage

The system shall store the processed topology and features in caracaldb.

Acceptance criteria:

```text
All required node and edge tables are created.
The number of stored edges matches the original cleaned dataset.
Repeated pipeline execution does not create unintended duplicate records.
```

#### FR-005. K-Hop Sampling Query Support

The system shall support neighbor sampling queries.

Acceptance criteria:

```text
Given a batch of target node IDs, it returns 1-hop and 2-hop neighbor subgraphs.
The system supports configuring the fan-out sizes (e.g., [15, 10]).
```

#### FR-006. Standardized Training Loop

The system shall use a standardized PyTorch GraphSAGE training loop.

Acceptance criteria:

```text
The PyTorch code remains identical across all tested backend systems.
The data loader output format is standardized.
```

#### FR-007. Metric Collection

The system shall collect and log GNN-specific metrics per batch and per epoch.

Acceptance criteria:

```text
The system records TTFB, Edges/sec, and Data Wait Ratio.
The system records GPU Memory and RAM Peak usage.
```

#### FR-008. Out-of-Core Handling

The system shall handle data larger than system RAM without crashing.

Acceptance criteria:

```text
The system successfully streams mini-batches for ogbn-papers100M.
OOM crashes in baseline systems are caught and logged properly as failed statuses.
```

#### FR-009. Report Generation

The system shall export benchmark results as files.

Acceptance criteria:

```text
Metric logs per epoch are exported as CSV.
Aggregated throughput and memory statistics are exported.
```

#### FR-010. Comparison System Output

The system shall provide a runnable 4x4 comparison matrix evaluating DataFrame and Database backend combinations.

Acceptance criteria:

```text
The DataFrame dimension must include lynxes, pandas, Polars, and Dask DataFrame.
The Database dimension must include caracaldb, PyG Native, DuckDB, and Neo4j.
Each runnable cell must measure Edges/sec, TTFB, Memory Peak, status, and notes.
The benchmark must use a GNN-style workload (K-hop sampling + feature fetching).
Neo4j and PyG Native may be skipped based on memory constraints or missing setups, but executable adapters must be implemented.
The comparison must be exported as a human-readable report and a CSV file.
```

### 2.2 Non-Functional Requirements

#### NFR-001. Reproducibility

Given the same input graph, configuration, and random seed, the system shall produce the same sampling behavior and comparable benchmark timings.

#### NFR-002. Modularity

Data loading, graph transformation, backend samplers, the PyTorch training loop, and metric collection shall be separated into independent modules.

#### NFR-003. Extensibility

The system shall be designed so that new models (e.g., Graph Attention Networks) or new datasets can be added easily.

#### NFR-004. Maintainability

Functions and classes shall use clear names. Core metric calculation logic shall include comments.

#### NFR-005. Error Handling

The system shall handle the following cases:

```text
Missing OGB dataset files
Out of Memory (OOM) during sampling
Database connection failure
Zero-degree nodes during sampling
```

#### NFR-006. Performance

For the ogbn-products dataset on a typical local development environment:

Recommended targets:

```text
Data loading and DB ingestion: within 5 minutes
TTFB (Time To First Batch): within 5 seconds
Sampling throughput: consistently supplying GPU to maintain > 70% utilization
```

#### NFR-007. Testability

Core samplers and query functions shall be unit-testable.

Test targets include:

```text
Neighbor count validation
Feature vector dimension validation
Tensor conversion integrity
```

## 3. Comparison System

### 3.1 Purpose

The comparison system explains and measures why caracaldb and lynxes are effective for this project, compared with alternative baseline databases and DataFrame technologies.

The comparison must be a runnable benchmark, demonstrating actual bottlenecks in GNN data loaders such as GPU starvation, memory cliffs, and feature serialization overhead.

### 3.2 Database Comparison

Comparison targets:

```text
caracaldb
PyG Native (In-memory)
DuckDB
Neo4j
```

Comparison criteria:

```text
Installation and execution difficulty
Suitability for in-memory processing
Suitability for out-of-core scalability
K-hop graph traversal efficiency
Feature fetching speed
Large-scale scalability
```

### 3.3 Database Comparison Table

| Criterion | caracaldb | PyG Native | DuckDB | Neo4j |
|---|---|---|---|---|
| Required by this project | High | Low | Low | Low |
| Local setup convenience | High | Very High | Very High | Medium |
| In-memory speed | Medium-High | Very High | Medium | Low |
| Graph traversal efficiency | High | Very High | Low | Very High |
| Feature fetching efficiency | High assumed | Very High | High | Low |
| Out-of-core scalability | High | Low | High | High |
| Learning curve | Medium | High | Low | Medium |
| Large-scale scalability | High | Low | Medium | High |
| GNN benchmark suitability | High | Very High | Medium | Medium |

### 3.4 Database Evaluation

#### caracaldb

caracaldb is the required database for this project. It serves as the main storage layer for the graph topology and assists in neighbor retrieval.

Advantages:

```text
It directly satisfies the client-specified technology requirement.
It bridges the gap between scalable out-of-core storage and fast structural querying.
It can reduce the extreme memory burden of holding billion-scale graphs in RAM.
```

Limitations:

```text
Its raw in-memory speed may theoretically lag slightly behind highly optimized C++ pointers used in PyG Native.
```

Recommended use in this project:

```text
Store edge topology
Execute 1-hop and 2-hop neighborhood queries
Handle dynamic temporal filtering dynamically
```

#### PyG Native (In-memory)

PyG Native represents the standard PyTorch Geometric in-memory approach.

Advantages:

```text
It is the industry standard baseline.
It provides extremely fast in-memory C++ based sampling (CSR/CSC formats).
```

Limitations:

```text
It strictly requires the entire graph to fit in RAM.
It fails immediately (OOM) on massive graphs like ogbn-papers100M on standard workstations.
Filtering edges on-the-fly is computationally expensive.
```

Reason for not selecting PyG Native as primary:

```text
The project requires caracaldb.
It is used as the high-speed baseline for small/medium datasets.
```

#### DuckDB

DuckDB is a relational analytical baseline capable of querying Parquet files.

Advantages:

```text
Excellent analytical processing and feature joining.
Handles out-of-core data well.
```

Limitations:

```text
K-hop sampling requires self-joins (Edge table joined with Edge table), leading to join explosions and poor traversal performance.
```

Reason for not selecting DuckDB:

```text
Graph topology traversal is unnatural and slow in purely relational engines.
It serves as the relational baseline comparison.
```

#### Neo4j

Neo4j is the industry standard for general-purpose graph databases.

Advantages:

```text
Natural graph modeling and multi-hop Cypher queries.
Excellent out-of-core capability.
```

Limitations:

```text
Network overhead and object serialization overhead when fetching large feature vectors (e.g., 128-dimensional arrays) per node cause severe bottlenecks for GNN data loading.
```

Reason for not selecting Neo4j:

```text
The project requires caracaldb.
It serves as the generic Graph DB baseline to demonstrate GNN-specific bottlenecks.
```

### 3.5 DataFrame Library Comparison

Comparison targets:

```text
lynxes
pandas
Polars
Dask DataFrame
```

Comparison criteria:

```text
Data cleaning convenience
Suitability for feature vector processing
Zero-copy tensor conversion efficiency
Large-data handling
Memory peak during feature joins
```

### 3.6 DataFrame Comparison Table

| Criterion | lynxes | pandas | Polars | Dask DataFrame |
|---|---|---|---|---|
| Required by this project | High | Low | Low | Low |
| Feature vector handling | High assumed | Medium | Very High | Medium |
| Zero-copy to Tensor | High assumed | Low | High | Low |
| Small-data suitability | High | Very High | High | Low-Medium |
| Large-data suitability | Medium | Medium | High | Very High |
| Memory Peak efficiency | Medium-High | Low | Very High | Medium |
| GNN benchmark suitability | High | Medium | High | Low |

### 3.7 DataFrame Library Evaluation

#### lynxes

lynxes is the required DataFrame library for this project. It is used for feature ingestion, transformation, dynamic joins, and tensor preparation.

Advantages:

```text
It directly satisfies the client-specified DataFrame requirement.
It provides a structured way to handle tabular node features before feeding them to the GNN.
```

Limitations:

```
Specific PyTorch tensor conversion zero-copy optimizations may require custom handling compared to more mature ecosystems.
```

Recommended use in this project:

```text
Load node/edge files
Filter properties
Perform dynamic Heterogeneous feature joins
Batch feature fetching for GNN
```

#### pandas

pandas is the most widely used Python DataFrame library.

Advantages:

```text
Highly familiar syntax and broad compatibility.
```

Limitations:

```text
High memory overhead and object copying. Converting pandas series to PyTorch tensors often involves memory duplication, increasing peak memory.
```

Reason for not selecting pandas:

```text
The project requires lynxes. pandas is used as the baseline.
```

#### Polars

Polars is a high-performance, Arrow-backed DataFrame library.

Advantages:

```text
Exceptional speed and strict columnar memory layout.
Arrow backend allows zero-copy conversion to PyTorch tensors.
```

Limitations:

```text
It is not the required library for this project.
```

Reason for not selecting Polars:

```text
It serves as the high-performance DataFrame baseline in the comparison matrix.
```

#### Dask DataFrame

Dask DataFrame handles distributed environments.

Advantages:

```text
Can process datasets larger than memory by splitting them.
```

Limitations:

```text
Massive overhead for real-time mini-batch sampling in a tight GNN training loop.
```

Reason for not selecting Dask:

```text
Too slow for real-time GNN feature fetching. Used only to complete the comparison matrix.
```

### 3.8 Final Technology Selection

The project uses:

```text
Database: caracaldb
DataFrame Library: lynxes
Machine Learning: PyTorch & PyTorch Geometric
Programming Language: Python
```

Selection rationale:

```text
The selected stack satisfies the client-specified technology constraints.
It provides a clear architecture demonstrating how a specialized DB (caracaldb) combined with a DataFrame (lynxes) can solve the GPU starvation and Memory Wall issues in GNN pipelines.
```

### 3.9 Benchmark Workload

The comparison system must evaluate the 4x4 matrix using a rigorous GNN training workload. Standard relational workloads are insufficient to measure GPU starvation.

The required benchmark workload is:

```text
Cold Start K-Hop Sampling (TTFB)
Warm Start K-Hop Sampling Throughput
Feature Fetching and Tensor Conversion latency
Temporal Graph Filtering
Out-of-Core Batch Streaming
```

The benchmark must run standard epochs (e.g., 5 epochs) to calculate stable steady-state metrics.

Recommended default for the matrix evaluation:

```text
Benchmark dataset: ogbn-products
Batch size: 1024
Fan-out sizes: [15, 10, 5]
Workload repeats: 5 epochs
```

For each matrix cell, the benchmark should report:

```text
dataframe
database
status
edges_per_sec
ttfb_seconds
data_wait_ratio
memory_peak_mb
notes
```

### 3.10 Example Comparison System Output

The comparison system may be implemented as a separate module:

```text
src/compare_systems.py
```

Example output:

```markdown
# 4x4 GNN Data Backend Comparison Matrix

Each cell is:
status / edges_per_sec / data_wait_ratio / memory_peak_mb

| DataFrame \ Database | caracaldb | PyG Native | DuckDB | Neo4j |
| -------------------- | --------: | ---------: | -----: | ----: |
| lynxes               | ok        | ok         | ok     | skipped/ok |
| pandas               | ok        | ok         | ok     | skipped/ok |
| Polars               | ok        | ok         | ok     | skipped/ok |
| Dask DataFrame       | ok        | skipped/oom| ok     | skipped/ok |
```

The system should also export a detailed CSV file containing every matrix cell. If baseline systems (like PyG Native on large graphs) fail due to memory limits, the cell should accurately report OOM (Out of Memory).

### 3.11 Evaluation Criteria

From a client perspective, the project can be evaluated using the following criteria:

```text
Whether caracaldb and lynxes are used as required in the graph processing pipeline
Whether the OGB data is loaded, cleaned, and properly mapped to nodes and edges
Whether the PyTorch GNN training loop remains standard and unbiased
Whether the system successfully isolates and measures data loading metrics (TTFB, Wait Ratio) distinct from model computation
Whether the comparison system runs a real DataFrame x Database matrix on GNN workloads
Whether out-of-core or high-memory scenarios are tested to show architectural advantages
Whether throughput, latency, memory peak, status, and notes are properly exported
Whether the README and execution instructions are clear
```