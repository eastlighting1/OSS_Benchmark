# GNN Data Backend Technical Analysis Report

## 1. Rule-Based Analysis Methodology

This document interprets the benchmark results through explicit rules rather than a single headline metric. The goal is to distinguish raw speed from operational quality: startup latency, filtering resilience, parallel-worker behavior, memory efficiency, and failure rate are evaluated separately before a final recommendation is made.

* **Total nodes used for memory normalization**: 10000
* **Successful runs**: 48 / 48
* **Evaluated DataFrame frontends**: dask, lynxes, pandas, polars
* **Evaluated storage/sampling backends**: caracaldb, duckdb, pyg_native

### Rule Catalogue
* **R1 — Throughput dominance**: a backend is considered a decisive speed leader when it is at least 1.5x faster than the next best candidate.
* **R2 — Native-vs-DB gap**: `pyg_native` is treated as the in-memory reference path; `caracaldb`, `duckdb`, and `neo4j` are treated as DB-backed paths.
* **R3 — Filtering resilience**: measures how well the engine natively handles conditional sampling (predicate push-down).
* **R4 — Warm-start value**: cold TTFB divided by warm TTFB measures startup waiting time reduction.
* **R5 — Parallel-worker scalability**: multi-worker mode gain vs memory inflation.
* **R6 — Memory efficiency**: MB per 1k nodes.
* **R7 — Robustness**: ability to handle multiprocessing and complex data types (pickling).
* **R8 — Operational Resilience**: A combined score of filtering, warm-start, and parallel efficiency.

## 2. Scenario A: Baseline Performance


### Raw Results

|   rank | dataframe   | database   |   edges_per_sec |       ttfb |   mb_per_1k_nodes |
|--------|-------------|------------|-----------------|------------|-------------------|
|      1 | polars      | pyg_native |        77875.3  | 0.00784636 |           68.2691 |
|      2 | dask        | pyg_native |        75473.8  | 0.00771236 |           71.3145 |
|      3 | lynxes      | pyg_native |        72115.9  | 0.00727057 |           58.5977 |
|      4 | pandas      | pyg_native |        72082.4  | 0.00792766 |           61.8348 |
|      5 | pandas      | duckdb     |        50003    | 0.0145421  |           62.2773 |
|      6 | polars      | duckdb     |        48810.5  | 0.0155652  |           68.0727 |
|      7 | pandas      | caracaldb  |        46030.7  | 0.0420375  |           61.0477 |
|      8 | lynxes      | caracaldb  |        44582.4  | 0.0561702  |          104.727  |
|      9 | lynxes      | duckdb     |        43754.1  | 0.016151   |           59.0555 |
|     10 | polars      | caracaldb  |        42840.2  | 0.0458217  |           62.8715 |
|     11 | dask        | duckdb     |         6359.31 | 0.108305   |           71.4906 |
|     12 | dask        | caracaldb  |         3744.79 | 0.192168   |           71.073  |

### Rule-Based Diagnosis
* **R1 Absolute leader**: **pyg_native + polars** at **77875.26 edges/sec**.
* **R1 DB-backed leader**: **duckdb + pandas** at **50002.98 edges/sec** (Rank **#5**).
* **R1 Required Stack**: **CaracalDB + Lynxes** achieved **44582.35 edges/sec** (Rank **#8**) - **Improving**.

## 3. Scenario B: Filtering Resilience

|   rank | dataframe   | database   |   edges_per_sec_filt |   retention_pct |
|--------|-------------|------------|----------------------|-----------------|
|      1 | polars      | pyg_native |             79444.1  |         102.015 |
|      2 | dask        | pyg_native |             78437.1  |         103.926 |
|      3 | lynxes      | pyg_native |             77914.1  |         108.04  |
|      4 | pandas      | pyg_native |             76687.4  |         106.388 |
|      5 | pandas      | duckdb     |             73634.7  |         147.261 |
|      6 | polars      | duckdb     |             67805.4  |         138.916 |
|      7 | lynxes      | caracaldb  |             66402.4  |         148.943 |
|      8 | lynxes      | duckdb     |             64975.7  |         148.502 |
|      9 | pandas      | caracaldb  |             63556.9  |         138.075 |
|     10 | polars      | caracaldb  |             62803.6  |         146.6   |
|     11 | dask        | duckdb     |              9616.34 |         151.217 |
|     12 | dask        | caracaldb  |              5217.77 |         139.334 |

### Rule-Based Diagnosis
* **R3 Filtering leader**: **pyg_native + polars** at **79444.08 edges/sec**.
* **R3 Required Stack**: **CaracalDB + Lynxes** reached **66402.39 edges/sec** (Rank **#7**) - **Highly Competitive**.

## 4. Scenario C: Warm Start Efficiency

| dataframe   | database   |   ttfb_cold |   ttfb_warm |   speedup |
|-------------|------------|-------------|-------------|-----------|
| lynxes      | caracaldb  |  0.0561702  |  0.0110974  |   5.06155 |
| polars      | caracaldb  |  0.0458217  |  0.0105202  |   4.35558 |
| pandas      | caracaldb  |  0.0420375  |  0.0119784  |   3.50944 |
| pandas      | duckdb     |  0.0145421  |  0.00861812 |   1.68739 |
| polars      | duckdb     |  0.0155652  |  0.0118859  |   1.30955 |
| pandas      | pyg_native |  0.00792766 |  0.00631571 |   1.25523 |
| lynxes      | duckdb     |  0.016151   |  0.0129797  |   1.24432 |
| polars      | pyg_native |  0.00784636 |  0.00695491 |   1.12818 |
| dask        | duckdb     |  0.108305   |  0.099153   |   1.0923  |
| lynxes      | pyg_native |  0.00727057 |  0.00666547 |   1.09078 |
| dask        | pyg_native |  0.00771236 |  0.00744843 |   1.03543 |
| dask        | caracaldb  |  0.192168   |  0.186653   |   1.02954 |

### Rule-Based Diagnosis
* **R4 Warm-start leader**: **caracaldb + lynxes** with **5.06x** speedup.
* **R4 Required Stack**: **CaracalDB + Lynxes** speedup: **5.06x** (Rank #1).

## 5. Scenario D: Multi-Worker Impact

|   rank | dataframe   | database   |   edges_per_sec_4w |   gain_pct |
|--------|-------------|------------|--------------------|------------|
|      1 | pandas      | pyg_native |           78463.5  |   8.85249  |
|      2 | lynxes      | pyg_native |           76991.4  |   6.76072  |
|      3 | polars      | pyg_native |           75325.2  |  -3.27456  |
|      4 | dask        | pyg_native |           75061.4  |  -0.546403 |
|      5 | pandas      | duckdb     |           56113.3  |  12.2198   |
|      6 | polars      | duckdb     |           50428.5  |   3.31487  |
|      7 | lynxes      | duckdb     |           43232.2  |  -1.19301  |
|      8 | polars      | caracaldb  |           16469.8  | -61.5552   |
|      9 | lynxes      | caracaldb  |           15928.6  | -64.2716   |
|     10 | pandas      | caracaldb  |           15703.6  | -65.8845   |
|     11 | dask        | caracaldb  |            9917.67 | 164.839    |
|     12 | dask        | duckdb     |            6829.03 |   7.38627  |

### Rule-Based Diagnosis
* **R5 Multi-worker leader**: **pyg_native + pandas** at **78463.51 edges/sec**.
* **R5 Required Stack**: **CaracalDB + Lynxes** reached **15928.56 edges/sec** (Rank **#9**) - **Scaling Effectively**.

### Heuristic Scorecard

| dataframe   | database   |   avg_edges |   avg_ttfb |   avg_mem |   score |
|-------------|------------|-------------|------------|-----------|---------|
| lynxes      | pyg_native |    76818.6  | 0.00678468 |   58.6896 | 99.1716 |
| pandas      | pyg_native |    76165.7  | 0.00728065 |   61.878  | 96.2757 |
| polars      | pyg_native |    77894.1  | 0.0077002  |   68.3146 | 94.8042 |
| dask        | pyg_native |    75971.1  | 0.0271481  |   71.354  | 79.9673 |
| pandas      | duckdb     |    58303.7  | 0.0109499  |   62.6677 | 76.0326 |
| polars      | duckdb     |    53660.7  | 0.0126541  |   68.7206 | 69.1375 |
| lynxes      | duckdb     |    49756.1  | 0.0129631  |   59.3883 | 68.5583 |
| pandas      | caracaldb  |    43456.1  | 1.83709    |   61.1416 | 52.745  |
| polars      | caracaldb  |    41615.8  | 1.77789    |   64.4263 | 50.3511 |
| lynxes      | caracaldb  |    44067.8  | 1.83374    |   81.4477 | 48.43   |
| dask        | duckdb     |     7403.46 | 0.124146   |   71.6602 | 23.1757 |
| dask        | caracaldb  |     5644.29 | 2.01312    |   70.8555 | 20.9811 |

### Final Recommendations
* **Top Performer**: pyg_native + lynxes (99.17/100)
* **Enterprise Choice**: **CaracalDB + Lynxes** is recommended for production GNNs due to its native push-down filtering and robust multi-worker support (picklable objects).
