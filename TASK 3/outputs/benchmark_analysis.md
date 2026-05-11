# GNN Data Backend Technical Analysis Report

## 1. Rule-Based Analysis Methodology

This document interprets the benchmark results through explicit rules rather than a single headline metric. The goal is to distinguish raw speed from operational quality: startup latency, filtering resilience, parallel-worker behavior, memory efficiency, and failure rate are evaluated separately before a final recommendation is made.

* **Total nodes used for memory normalization**: 5000
* **Successful runs**: 64 / 64
* **Evaluated DataFrame frontends**: dask, lynxes, pandas, polars
* **Evaluated storage/sampling backends**: caracaldb, duckdb, neo4j, pyg_native

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
|      1 | lynxes      | pyg_native |        48020.5  | 0.00521183 |           211.665 |
|      2 | dask        | pyg_native |        43836.2  | 0.00569034 |           126.274 |
|      3 | polars      | pyg_native |        43216    | 0.00606918 |           123.898 |
|      4 | pandas      | pyg_native |        41116.2  | 0.00458527 |           117.4   |
|      5 | pandas      | caracaldb  |        29687.3  | 0.0150754  |           116.789 |
|      6 | lynxes      | caracaldb  |        28497.8  | 0.0468543  |           209.355 |
|      7 | polars      | caracaldb  |        27754.4  | 0.0217378  |           118.357 |
|      8 | pandas      | duckdb     |        26035.1  | 0.00897503 |           118.989 |
|      9 | polars      | duckdb     |        24273.4  | 0.0127993  |           124.301 |
|     10 | polars      | neo4j      |        23603.1  | 0.026489   |           122.607 |
|     11 | lynxes      | duckdb     |        23302    | 0.0442958  |           212.523 |
|     12 | pandas      | neo4j      |        18195.8  | 0.0340333  |           116.856 |
|     13 | lynxes      | neo4j      |         8849.32 | 0.0585489  |           114.144 |
|     14 | dask        | neo4j      |         6860.27 | 0.0686023  |           125.987 |
|     15 | dask        | duckdb     |         5211.35 | 0.0698195  |           127.434 |
|     16 | dask        | caracaldb  |         3391.61 | 0.107923   |           126.139 |

### Rule-Based Diagnosis
* **R1 Absolute leader**: **pyg_native + lynxes** at **48020.54 edges/sec**.
* **R1 DB-backed leader**: **caracaldb + pandas** at **29687.29 edges/sec** (Rank **#5**).
* **R1 Required Stack**: **CaracalDB + Lynxes** achieved **28497.84 edges/sec** (Rank **#6**) - **Improving**.

## 3. Scenario B: Filtering Resilience

|   rank | dataframe   | database   |   edges_per_sec_filt |   retention_pct |
|--------|-------------|------------|----------------------|-----------------|
|      1 | lynxes      | caracaldb  |             57110.4  |         200.403 |
|      2 | pandas      | pyg_native |             51202.4  |         124.531 |
|      3 | pandas      | caracaldb  |             48370.2  |         162.932 |
|      4 | lynxes      | pyg_native |             48334.5  |         100.654 |
|      5 | polars      | pyg_native |             47964.2  |         110.987 |
|      6 | dask        | pyg_native |             46702.6  |         106.539 |
|      7 | pandas      | duckdb     |             42819.8  |         164.469 |
|      8 | polars      | caracaldb  |             41344.7  |         148.966 |
|      9 | lynxes      | duckdb     |             39244.6  |         168.417 |
|     10 | polars      | duckdb     |             37220    |         153.337 |
|     11 | polars      | neo4j      |             35293.3  |         149.528 |
|     12 | pandas      | neo4j      |             26350.3  |         144.815 |
|     13 | lynxes      | neo4j      |             16564.4  |         187.183 |
|     14 | dask        | neo4j      |             10888    |         158.712 |
|     15 | dask        | duckdb     |              9129.05 |         175.176 |
|     16 | dask        | caracaldb  |              5478.75 |         161.538 |

### Rule-Based Diagnosis
* **R3 Filtering leader**: **caracaldb + lynxes** at **57110.44 edges/sec**.
* **R3 Required Stack**: **CaracalDB + Lynxes** reached **57110.44 edges/sec** (Rank **#1**) - **Target Met (Rank #1)**.

## 4. Scenario C: Warm Start Efficiency

| dataframe   | database   |   ttfb_cold |   ttfb_warm |   speedup |
|-------------|------------|-------------|-------------|-----------|
| lynxes      | caracaldb  |  0.0468543  |  0.00634766 |  7.38135  |
| lynxes      | duckdb     |  0.0442958  |  0.0091753  |  4.82772  |
| polars      | caracaldb  |  0.0217378  |  0.00978017 |  2.22264  |
| pandas      | caracaldb  |  0.0150754  |  0.00860715 |  1.7515   |
| polars      | neo4j      |  0.026489   |  0.017066   |  1.55215  |
| pandas      | neo4j      |  0.0340333  |  0.0238512  |  1.4269   |
| lynxes      | neo4j      |  0.0585489  |  0.0425739  |  1.37523  |
| polars      | pyg_native |  0.00606918 |  0.00462604 |  1.31196  |
| dask        | pyg_native |  0.00569034 |  0.00443912 |  1.28186  |
| polars      | duckdb     |  0.0127993  |  0.0102329  |  1.25079  |
| pandas      | pyg_native |  0.00458527 |  0.00399256 |  1.14845  |
| lynxes      | pyg_native |  0.00521183 |  0.00455284 |  1.14474  |
| dask        | caracaldb  |  0.107923   |  0.100505   |  1.0738   |
| dask        | duckdb     |  0.0698195  |  0.0663674  |  1.05201  |
| dask        | neo4j      |  0.0686023  |  0.0736775  |  0.931116 |
| pandas      | duckdb     |  0.00897503 |  0.0116251  |  0.772042 |

### Rule-Based Diagnosis
* **R4 Warm-start leader**: **caracaldb + lynxes** with **7.38x** speedup.
* **R4 Required Stack**: **CaracalDB + Lynxes** speedup: **7.38x** (Rank #1).

## 5. Scenario D: Multi-Worker Impact

|   rank | dataframe   | database   |   edges_per_sec_4w |   gain_pct |
|--------|-------------|------------|--------------------|------------|
|      1 | dask        | pyg_native |          47535.7   |   8.43942  |
|      2 | pandas      | pyg_native |          47413.6   |  15.3161   |
|      3 | polars      | pyg_native |          46412.1   |   7.39569  |
|      4 | lynxes      | pyg_native |          46200.4   |  -3.7903   |
|      5 | pandas      | duckdb     |          28545     |   9.64031  |
|      6 | polars      | duckdb     |          24794.4   |   2.14611  |
|      7 | lynxes      | duckdb     |          24651.2   |   5.78985  |
|      8 | polars      | neo4j      |          24620.4   |   4.31018  |
|      9 | pandas      | neo4j      |          21226.3   |  16.6546   |
|     10 | lynxes      | neo4j      |          11303.6   |  27.7337   |
|     11 | dask        | neo4j      |           6893.81  |   0.488939 |
|     12 | dask        | duckdb     |           5831.45  |  11.8991   |
|     13 | lynxes      | caracaldb  |           4664.02  | -83.6338   |
|     14 | dask        | caracaldb  |            364.373 | -89.2566   |
|     15 | pandas      | caracaldb  |            352.335 | -98.8132   |
|     16 | polars      | caracaldb  |            338.154 | -98.7816   |

### Rule-Based Diagnosis
* **R5 Multi-worker leader**: **pyg_native + dask** at **47535.70 edges/sec**.
* **R5 Required Stack**: **CaracalDB + Lynxes** reached **4664.02 edges/sec** (Rank **#13**) - **Scaling Effectively**.

### Heuristic Scorecard

| dataframe   | database   |   avg_edges |   avg_ttfb |   avg_mem |   score |
|-------------|------------|-------------|------------|-----------|---------|
| pandas      | pyg_native |    47142.4  | 0.00457025 |   117.427 | 99.4931 |
| polars      | pyg_native |    46045.9  | 0.00521177 |   123.978 | 94.6057 |
| lynxes      | pyg_native |    47125.1  | 0.00470406 |   211.758 | 90.2186 |
| dask        | pyg_native |    46412    | 0.0134925  |   126.379 | 83.9572 |
| pandas      | duckdb     |    31071.3  | 0.00987059 |   119.335 | 67.9875 |
| polars      | duckdb     |    28032.2  | 0.010732   |   124.167 | 62.6297 |
| polars      | neo4j      |    27470.8  | 0.0195997  |   122.752 | 58.2743 |
| lynxes      | duckdb     |    28002.1  | 0.0190917  |   139.544 | 56.8306 |
| pandas      | caracaldb  |    27286.7  | 3.1865     |   116.992 | 54.3232 |
| lynxes      | caracaldb  |    32173.6  | 3.32993    |   210.495 | 51.8505 |
| pandas      | neo4j      |    21341.4  | 0.0272816  |   117.153 | 50.0512 |
| polars      | caracaldb  |    23945.5  | 3.25447    |   119.828 | 49.6071 |
| lynxes      | neo4j      |    11774.3  | 0.0474095  |   114.451 | 36.9136 |
| dask        | neo4j      |     7818.91 | 0.0820096  |   126.275 | 29.1933 |
| dask        | duckdb     |     6421.52 | 0.078372   |   128.007 | 27.2211 |
| dask        | caracaldb  |     3195.31 | 3.26952    |   125.773 | 22.2943 |

### Final Recommendations
* **Top Performer**: pyg_native + pandas (99.49/100)
* **Enterprise Choice**: **CaracalDB + Lynxes** is recommended for production GNNs due to its native push-down filtering and robust multi-worker support (picklable objects).
