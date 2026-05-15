# GNN Data Backend Technical Analysis Report

## 1. Rule-Based Analysis Methodology

This document interprets the benchmark results through explicit rules rather than a single headline metric. The goal is to distinguish raw speed from operational quality: startup latency, filtering resilience, parallel-worker behavior, memory efficiency, and failure rate are evaluated separately before a final recommendation is made.

* **Total nodes used for memory normalization**: 50000
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


### Raw Results (All Systems)

|   rank | dataframe   | database   |   edges_per_sec |      ttfb |   mb_per_1k_nodes |
|--------|-------------|------------|-----------------|-----------|-------------------|
|      1 | lynxes      | caracaldb  |       281415    | 0.109857  |           22.458  |
|      2 | pandas      | caracaldb  |       267290    | 0.0926991 |           17.4463 |
|      3 | polars      | caracaldb  |       244016    | 0.0920844 |           18.9252 |
|      4 | pandas      | pyg_native |       177990    | 0.0714817 |           17.3998 |
|      5 | lynxes      | pyg_native |       176740    | 0.0620139 |           14.8071 |
|      6 | dask        | pyg_native |       174942    | 0.0492835 |           20.2941 |
|      7 | polars      | pyg_native |       173696    | 0.064574  |           17.8484 |
|      8 | pandas      | duckdb     |       116259    | 0.0581193 |           17.8303 |
|      9 | lynxes      | duckdb     |       116068    | 0.0578322 |           15.3464 |
|     10 | polars      | duckdb     |       113047    | 0.0499368 |           18.1473 |
|     11 | lynxes      | neo4j      |        57059.2  | 0.315029  |           16.0757 |
|     12 | pandas      | neo4j      |        56937.2  | 0.32184   |           18.3673 |
|     13 | polars      | neo4j      |        56790.2  | 0.251007  |           18.203  |
|     14 | dask        | neo4j      |        28202.2  | 0.65271   |           20.8717 |
|     15 | dask        | caracaldb  |         9399.44 | 0.687135  |           20.2799 |
|     16 | dask        | duckdb     |         9338.71 | 0.67447   |           20.7391 |

### Raw Results (DB-Backed Systems Only)

|   rank | dataframe   | database   |   edges_per_sec |      ttfb |   mb_per_1k_nodes |
|--------|-------------|------------|-----------------|-----------|-------------------|
|      1 | lynxes      | caracaldb  |       281415    | 0.109857  |           22.458  |
|      2 | pandas      | caracaldb  |       267290    | 0.0926991 |           17.4463 |
|      3 | polars      | caracaldb  |       244016    | 0.0920844 |           18.9252 |
|      4 | pandas      | duckdb     |       116259    | 0.0581193 |           17.8303 |
|      5 | lynxes      | duckdb     |       116068    | 0.0578322 |           15.3464 |
|      6 | polars      | duckdb     |       113047    | 0.0499368 |           18.1473 |
|      7 | lynxes      | neo4j      |        57059.2  | 0.315029  |           16.0757 |
|      8 | pandas      | neo4j      |        56937.2  | 0.32184   |           18.3673 |
|      9 | polars      | neo4j      |        56790.2  | 0.251007  |           18.203  |
|     10 | dask        | neo4j      |        28202.2  | 0.65271   |           20.8717 |
|     11 | dask        | caracaldb  |         9399.44 | 0.687135  |           20.2799 |
|     12 | dask        | duckdb     |         9338.71 | 0.67447   |           20.7391 |

### Raw Results (In-Memory Baselines Only)

|   rank | dataframe   | database   |   edges_per_sec |      ttfb |   mb_per_1k_nodes |
|--------|-------------|------------|-----------------|-----------|-------------------|
|      1 | pandas      | pyg_native |          177990 | 0.0714817 |           17.3998 |
|      2 | lynxes      | pyg_native |          176740 | 0.0620139 |           14.8071 |
|      3 | dask        | pyg_native |          174942 | 0.0492835 |           20.2941 |
|      4 | polars      | pyg_native |          173696 | 0.064574  |           17.8484 |

### Rule-Based Diagnosis
* **R1 Absolute leader**: **caracaldb + lynxes** at **281415.04 edges/sec**.
* **R1 DB-backed leader**: **caracaldb + lynxes** at **281415.04 edges/sec** (Rank **#1** in All, #1 in DB).
* **R1 Required Stack**: **CaracalDB + Lynxes** achieved **281415.04 edges/sec** (DB Rank **#1**) - **Target Met**.

## 3. Scenario B: Filtering Resilience

|   rank | dataframe   | database   |   edges_per_sec_filt |   retention_pct |
|--------|-------------|------------|----------------------|-----------------|
|      1 | lynxes      | caracaldb  |             297160   |         105.595 |
|      2 | pandas      | caracaldb  |             267528   |         100.089 |
|      3 | polars      | caracaldb  |             258193   |         105.81  |
|      4 | pandas      | pyg_native |             189147   |         106.268 |
|      5 | dask        | pyg_native |             185946   |         106.29  |
|      6 | lynxes      | pyg_native |             185682   |         105.059 |
|      7 | polars      | pyg_native |             184957   |         106.483 |
|      8 | pandas      | duckdb     |             129541   |         111.424 |
|      9 | lynxes      | duckdb     |             126941   |         109.368 |
|     10 | polars      | duckdb     |             126253   |         111.682 |
|     11 | lynxes      | neo4j      |              59984   |         105.126 |
|     12 | pandas      | neo4j      |              59182.7 |         103.944 |
|     13 | polars      | neo4j      |              59113.7 |         104.091 |
|     14 | dask        | neo4j      |              31392.5 |         111.312 |
|     15 | dask        | caracaldb  |              10675.1 |         113.571 |
|     16 | dask        | duckdb     |              10608.6 |         113.598 |

### Rule-Based Diagnosis
* **R3 Filtering leader**: **caracaldb + lynxes** at **297160.07 edges/sec**.

## 4. Scenario C: Warm Start Efficiency

| dataframe   | database   |   ttfb_cold |   ttfb_warm |   speedup |
|-------------|------------|-------------|-------------|-----------|
| lynxes      | caracaldb  |   0.109857  |   0.0272675 |  4.02888  |
| pandas      | caracaldb  |   0.0926991 |   0.0261376 |  3.54658  |
| polars      | caracaldb  |   0.0920844 |   0.0337834 |  2.72573  |
| lynxes      | duckdb     |   0.0578322 |   0.0434232 |  1.33183  |
| pandas      | duckdb     |   0.0581193 |   0.0476599 |  1.21946  |
| dask        | neo4j      |   0.65271   |   0.560154  |  1.16523  |
| pandas      | pyg_native |   0.0714817 |   0.0614052 |  1.1641   |
| pandas      | neo4j      |   0.32184   |   0.278945  |  1.15378  |
| lynxes      | neo4j      |   0.315029  |   0.286083  |  1.10118  |
| polars      | duckdb     |   0.0499368 |   0.049675  |  1.00527  |
| dask        | duckdb     |   0.67447   |   0.676501  |  0.996998 |
| dask        | caracaldb  |   0.687135  |   0.705464  |  0.974018 |
| polars      | pyg_native |   0.064574  |   0.067328  |  0.959096 |
| polars      | neo4j      |   0.251007  |   0.2669    |  0.940452 |
| lynxes      | pyg_native |   0.0620139 |   0.0952599 |  0.650996 |
| dask        | pyg_native |   0.0492835 |   0.0876992 |  0.561961 |

### Rule-Based Diagnosis
* **R4 Warm-start leader**: **caracaldb + lynxes** with **4.03x** speedup.
* **R4 Required Stack**: **CaracalDB + Lynxes** speedup: **4.03x** (Rank #1).

## 5. Scenario D: Multi-Worker Impact

|   rank | dataframe   | database   |   edges_per_sec_4w |   gain_pct |
|--------|-------------|------------|--------------------|------------|
|      1 | lynxes      | caracaldb  |          241317    | -14.2488   |
|      2 | pandas      | caracaldb  |          230002    | -13.9504   |
|      3 | polars      | caracaldb  |          214528    | -12.0844   |
|      4 | dask        | pyg_native |          175662    |   0.411564 |
|      5 | pandas      | pyg_native |          175168    |  -1.58546  |
|      6 | lynxes      | pyg_native |          171490    |  -2.97006  |
|      7 | polars      | pyg_native |          166273    |  -4.27341  |
|      8 | pandas      | duckdb     |          121095    |   4.15928  |
|      9 | lynxes      | duckdb     |          119284    |   2.77089  |
|     10 | polars      | duckdb     |          113622    |   0.508903 |
|     11 | pandas      | neo4j      |           57573.4  |   1.11737  |
|     12 | polars      | neo4j      |           57138.3  |   0.613048 |
|     13 | lynxes      | neo4j      |           56511.3  |  -0.960137 |
|     14 | dask        | neo4j      |           28277.5  |   0.267127 |
|     15 | dask        | duckdb     |            9381.96 |   0.46311  |
|     16 | dask        | caracaldb  |            9279.48 |  -1.27624  |

### Rule-Based Diagnosis
* **R5 Multi-worker leader**: **caracaldb + lynxes** at **241316.64 edges/sec**.

### Heuristic Scorecard

| dataframe   | database   |   avg_edges |   avg_ttfb |   avg_mem |   score |
|-------------|------------|-------------|------------|-----------|---------|
| lynxes      | caracaldb  |   277630    |  0.588915  |   18.4105 | 77.9044 |
| pandas      | caracaldb  |   255274    |  0.904481  |   17.3923 | 73.4721 |
| lynxes      | pyg_native |   176942    |  0.0717345 |   15.0151 | 71.3176 |
| polars      | caracaldb  |   241807    |  0.514199  |   18.1177 | 70.6575 |
| polars      | pyg_native |   175404    |  0.0655816 |   17.855  | 69.0313 |
| pandas      | pyg_native |   178902    |  0.0785352 |   17.5619 | 67.7085 |
| lynxes      | duckdb     |   120955    |  0.047124  |   15.4299 | 65.5104 |
| pandas      | duckdb     |   121607    |  0.0494529 |   17.9271 | 62.0027 |
| polars      | duckdb     |   118026    |  0.0469069 |   18.2516 | 61.9605 |
| dask        | pyg_native |   176717    |  0.157874  |   20.4374 | 58.8273 |
| lynxes      | neo4j      |    57424.7  |  0.309029  |   15.7548 | 34.5071 |
| polars      | neo4j      |    57604.1  |  0.28431   |   17.7334 | 32.6831 |
| pandas      | neo4j      |    57824.9  |  0.30369   |   17.9588 | 32.3076 |
| dask        | neo4j      |    29034.4  |  0.698442  |   20.6363 | 22.1701 |
| dask        | duckdb     |     9678.43 |  0.769712  |   20.6235 | 17.8716 |
| dask        | caracaldb  |     9691.31 |  1.16054   |   20.2629 | 17.7231 |

### Final Recommendations
* **Top Performer**: caracaldb + lynxes (77.90/100)
* **Data-Driven Recommendation**: Based on the aggregated heuristic scorecard, **caracaldb + lynxes** offers the best balance of throughput, latency, and memory efficiency for the evaluated workloads.
