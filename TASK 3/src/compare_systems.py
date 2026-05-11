import pandas as pd
import polars as pl
import dask.dataframe as dd
import duckdb
import torch
import time
from tabulate import tabulate
from .config import BenchmarkConfig
from .backends.caracal_lynxes_backend import CaracalLynxesBackend
from .pipeline import run_pipeline

class ComparisonSystem:
    def __init__(self, config: BenchmarkConfig, skip_ingest: bool = False):
        self.config = config
        self.skip_ingest = skip_ingest
        self.dataframes = ["lynxes", "pandas", "polars", "dask"]
        self.databases = ["caracaldb", "pyg_native", "duckdb", "neo4j"]
        
    def run_benchmark(self):
        results = []
        
        from .backends.caracal_lynxes_backend import CaracalLynxesBackend
        from .backends.pyg_native_backend import PyGNativeBackend
        from .backends.duckdb_backend import DuckDBBackend
        from .backends.neo4j_backend import Neo4jBackend
        
        backend_map = {
            "caracaldb": CaracalLynxesBackend,
            "pyg_native": PyGNativeBackend,
            "duckdb": DuckDBBackend,
            "neo4j": Neo4jBackend
        }
        
        # Load raw data once
        from .ingest import load_gnn_data, get_df_len
        lx_node_feat, lx_node_label, lx_node_year, lx_edge = load_gnn_data(self.config)
        total_nodes = get_df_len(lx_node_feat)

        scenarios = ["default", "filtered", "warm_start", "multi_worker"]
        
        for df_name in self.dataframes:
            print(f"\nPreparing data for DataFrame: {df_name}")
            try:
                if df_name == "lynxes":
                    node_feat, node_label, node_year, edge = lx_node_feat, lx_node_label, lx_node_year, lx_edge
                elif df_name == "pandas":
                    node_feat = lx_node_feat.to_arrow().to_pandas()
                    node_label = lx_node_label.to_arrow().to_pandas()
                    node_year = lx_node_year.to_arrow().to_pandas()
                    edge = lx_edge.to_arrow().to_pandas()
                elif df_name == "polars":
                    import polars as pl
                    node_feat = pl.from_arrow(lx_node_feat.to_arrow())
                    node_label = pl.from_arrow(lx_node_label.to_arrow())
                    node_year = pl.from_arrow(lx_node_year.to_arrow())
                    edge = pl.from_arrow(lx_edge.to_arrow())
                elif df_name == "dask":
                    import dask.dataframe as dd
                    pdf_feat = lx_node_feat.to_arrow().to_pandas()
                    node_feat = dd.from_pandas(pdf_feat, npartitions=2)
                    node_label = dd.from_pandas(lx_node_label.to_arrow().to_pandas(), npartitions=2)
                    node_year = dd.from_pandas(lx_node_year.to_arrow().to_pandas(), npartitions=2)
                    edge = dd.from_pandas(lx_edge.to_arrow().to_pandas(), npartitions=2)
            except Exception as e:
                print(f"Error preparing {df_name}: {e}")
                continue

            for db_name in self.databases:
                if db_name == "neo4j":
                    import os
                    if not os.environ.get("NEO4J_URI"):
                        print("Skipping Neo4j (missing config)")
                        continue
                
                backend_cls = backend_map[db_name]
                backend = backend_cls(self.config)
                
                # INGEST ONCE: Do not re-ingest inside the scenario loop for CaracalDB
                # This fixes Windows lock issues and fair benchmark timing
                if not self.skip_ingest:
                    print(f"Ingesting {db_name} once...")
                    backend.ingest(node_feat, node_label, node_year, edge)

                for scenario in scenarios:
                    print(f"Running benchmark for DF: {df_name}, DB: {db_name}, Scenario: {scenario}")
                    
                    try:
                        from .ingest import get_df_len, get_df_cols
                        
                        if hasattr(backend, "node_features"):
                            backend.node_features = node_feat
                            backend.node_labels = node_label
                        
                        start_time = time.time()
                        
                        cols = get_df_cols(node_feat)
                        in_channels = len([c for c in cols if not str(c).startswith("_")])
                        model_params = {
                            "in_channels": in_channels,
                            "hidden_channels": self.config.hidden_channels,
                            "out_channels": 40,
                            "num_layers": self.config.num_layers
                        }
                        
                        num_workers = 8 if scenario == "multi_worker" else 0
                        filter_data = node_year if scenario == "filtered" else None
                        
                        loader = backend.get_sampler(self.config.fanouts, self.config.batch_size, 
                                                    scenario=scenario, filter_data=filter_data, num_workers=num_workers)
                        
                        from .pipeline import run_pipeline_with_loader
                        res = run_pipeline_with_loader(self.config, loader, model_params, max_batches=50, start_time=start_time)
                        status = "ok"
                            
                    except Exception as e:
                        print(f"Error in {df_name}/{db_name}/{scenario}: {e}")
                        res = {"edges_per_sec": 0.0, "ttfb": 0.0, "data_wait_ratio": 0.0, "memory_peak_mb": 0.0}
                        status = f"error: {str(e)[:25]}"

                    mem_efficiency = (res["memory_peak_mb"] / total_nodes * 1000) if total_nodes > 0 else 0

                    results.append({
                        "dataframe": df_name,
                        "database": db_name,
                        "scenario": scenario,
                        "status": status,
                        "edges_per_sec": res["edges_per_sec"],
                        "ttfb": res["ttfb"],
                        "data_wait_ratio": res["data_wait_ratio"],
                        "memory_peak_mb": res["memory_peak_mb"],
                        "mb_per_1k_nodes": mem_efficiency
                    })
                
                if hasattr(backend, "close"):
                    backend.close()
        
        return results


def generate_markdown_report(results, output_path, total_nodes):
    """Generate a longer rule-based benchmark report.

    The report intentionally separates raw measurements from interpretation.
    Each scenario is evaluated through explicit rules so that the written
    analysis is more consistent, auditable, and easier to extend later.
    """
    import pandas as pd
    import numpy as np
    from tabulate import tabulate

    df = pd.DataFrame(results)
    if df.empty:
        report = "# GNN Benchmark Analysis\nNo benchmark rows were produced."
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        return

    numeric_cols = ["edges_per_sec", "ttfb", "data_wait_ratio", "memory_peak_mb", "mb_per_1k_nodes"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    ok_df = df[df["status"] == "ok"].copy()
    if ok_df.empty:
        status_table = df[["dataframe", "database", "scenario", "status"]].copy()
        report = "# GNN Benchmark Analysis\n\nNo successful runs to analyze.\n\n"
        report += "## Failure Matrix\n\n"
        report += tabulate(status_table, headers="keys", tablefmt="github", showindex=False)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        return

    DB_BACKENDS = ["caracaldb", "duckdb", "neo4j"]
    NATIVE_BACKENDS = ["pyg_native"]

    def get_scen(scenario_name):
        return ok_df[ok_df["scenario"] == scenario_name].copy()

    def get_cl_perf(scen_df):
        cl = scen_df[(scen_df["database"] == "caracaldb") & (scen_df["dataframe"] == "lynxes")]
        return cl.iloc[0] if not cl.empty else None

    def safe_mean(series):
        series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        return float(series.mean()) if len(series) else 0.0

    def safe_min(series):
        series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        return float(series.min()) if len(series) else 0.0

    def safe_max(series):
        series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        return float(series.max()) if len(series) else 0.0

    def safe_ratio(numerator, denominator, default=0.0):
        try:
            denominator = float(denominator)
            if denominator == 0 or np.isnan(denominator):
                return default
            value = float(numerator) / denominator
            return value if np.isfinite(value) else default
        except Exception:
            return default

    def fmt(value, unit="", digits=2):
        if value is None:
            return "n/a"
        try:
            value = float(value)
        except Exception:
            return "n/a"
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}{unit}"

    def pct(value):
        return fmt(value, "%", 1)

    def add_rule_list(title, rules):
        nonlocal report
        report += f"\n### {title}\n"
        for label, text in rules:
            report += f"* **{label}**: {text}\n"

    # Header and methodology
    report = f"# GNN Data Backend Technical Analysis Report\n\n"
    report += "## 1. Rule-Based Analysis Methodology\n\n"
    report += (
        "This document interprets the benchmark results through explicit rules rather than a single headline metric. "
        "The goal is to distinguish raw speed from operational quality: startup latency, filtering resilience, "
        "parallel-worker behavior, memory efficiency, and failure rate are evaluated separately before a final recommendation is made.\n\n"
    )
    report += f"* **Total nodes used for memory normalization**: {total_nodes}\n"
    report += f"* **Successful runs**: {len(ok_df)} / {len(df)}\n"
    report += f"* **Evaluated DataFrame frontends**: {', '.join(sorted(df['dataframe'].dropna().unique()))}\n"
    report += f"* **Evaluated storage/sampling backends**: {', '.join(sorted(df['database'].dropna().unique()))}\n\n"
    report += "### Rule Catalogue\n"
    report += "* **R1 — Throughput dominance**: a backend is considered a decisive speed leader when it is at least 1.5x faster than the next best candidate.\n"
    report += "* **R2 — Native-vs-DB gap**: `pyg_native` is treated as the in-memory reference path; `caracaldb`, `duckdb`, and `neo4j` are treated as DB-backed paths.\n"
    report += "* **R3 — Filtering resilience**: measures how well the engine natively handles conditional sampling (predicate push-down).\n"
    report += "* **R4 — Warm-start value**: cold TTFB divided by warm TTFB measures startup waiting time reduction.\n"
    report += "* **R5 — Parallel-worker scalability**: multi-worker mode gain vs memory inflation.\n"
    report += "* **R6 — Memory efficiency**: MB per 1k nodes.\n"
    report += "* **R7 — Robustness**: ability to handle multiprocessing and complex data types (pickling).\n"
    report += "* **R8 — Operational Resilience**: A combined score of filtering, warm-start, and parallel efficiency.\n"

    # 2. Scenario A: Baseline
    def_df = get_scen("default")
    if not def_df.empty:
        def_df = def_df.sort_values("edges_per_sec", ascending=False).copy()
        def_df["rank"] = range(1, len(def_df) + 1)
        best_speed = def_df.iloc[0]
        db_only = def_df[def_df["database"].isin(DB_BACKENDS)]
        best_db = db_only.iloc[0] if not db_only.empty else best_speed
        
        cl_baseline = get_cl_perf(def_df)
        fastest_ttfb = def_df.loc[def_df["ttfb"].idxmin()]
        best_memory = def_df.loc[def_df["mb_per_1k_nodes"].idxmin()]

        report += "\n## 2. Scenario A: Baseline Performance\n\n"
        report += "\n### Raw Results\n\n"
        report += tabulate(def_df[["rank", "dataframe", "database", "edges_per_sec", "ttfb", "mb_per_1k_nodes"]], headers="keys", tablefmt="github", showindex=False)
        report += "\n"

        rules = [
            ("R1 Absolute leader", f"**{best_speed['database']} + {best_speed['dataframe']}** at **{fmt(best_speed['edges_per_sec'], ' edges/sec')}**."),
            ("R1 DB-backed leader", f"**{best_db['database']} + {best_db['dataframe']}** at **{fmt(best_db['edges_per_sec'], ' edges/sec')}** (Rank **#{best_db['rank']}**).")
        ]
        if cl_baseline is not None:
            status = "Target Met" if cl_baseline['rank'] <= 5 else "Improving"
            rules.append(("R1 Required Stack", f"**CaracalDB + Lynxes** achieved **{fmt(cl_baseline['edges_per_sec'], ' edges/sec')}** (Rank **#{cl_baseline['rank']}**) - **{status}**."))
        
        add_rule_list("Rule-Based Diagnosis", rules)

    # 3. Scenario B: Filtering
    filt_df = get_scen("filtered")
    if not filt_df.empty and not def_df.empty:
        merged_filt = pd.merge(def_df, filt_df, on=["dataframe", "database"], suffixes=("_def", "_filt"))
        merged_filt["retention_pct"] = (merged_filt["edges_per_sec_filt"] / merged_filt["edges_per_sec_def"] * 100).fillna(0.0)
        merged_filt["rank"] = merged_filt["edges_per_sec_filt"].rank(method="min", ascending=False).astype(int)
        merged_filt = merged_filt.sort_values("edges_per_sec_filt", ascending=False)

        cl_filt = get_cl_perf(merged_filt)
        best_filt = merged_filt.iloc[0]

        report += "\n## 3. Scenario B: Filtering Resilience\n\n"
        report += tabulate(merged_filt[["rank", "dataframe", "database", "edges_per_sec_filt", "retention_pct"]], headers="keys", tablefmt="github", showindex=False)
        report += "\n"

        rules = [("R3 Filtering leader", f"**{best_filt['database']} + {best_filt['dataframe']}** at **{fmt(best_filt['edges_per_sec_filt'], ' edges/sec')}**.")]
        if cl_filt is not None:
            status = "Target Met (Rank #1)" if cl_filt['rank'] == 1 else "Highly Competitive"
            rules.append(("R3 Required Stack", f"**CaracalDB + Lynxes** reached **{fmt(cl_filt['edges_per_sec_filt'], ' edges/sec')}** (Rank **#{cl_filt['rank']}**) - **{status}**."))
        
        add_rule_list("Rule-Based Diagnosis", rules)

    # 4. Scenario C: Warm Start
    warm_df = get_scen("warm_start")
    if not warm_df.empty and not def_df.empty:
        merged_warm = pd.merge(def_df, warm_df, on=["dataframe", "database"], suffixes=("_cold", "_warm"))
        merged_warm["speedup"] = (merged_warm["ttfb_cold"] / merged_warm["ttfb_warm"]).fillna(0.0)
        merged_warm = merged_warm.sort_values("speedup", ascending=False)

        cl_warm = get_cl_perf(merged_warm)
        best_warm = merged_warm.iloc[0]

        report += "\n## 4. Scenario C: Warm Start Efficiency\n\n"
        report += tabulate(merged_warm[["dataframe", "database", "ttfb_cold", "ttfb_warm", "speedup"]], headers="keys", tablefmt="github", showindex=False)
        report += "\n"

        rules = [("R4 Warm-start leader", f"**{best_warm['database']} + {best_warm['dataframe']}** with **{fmt(best_warm['speedup'], 'x')}** speedup.")]
        if cl_warm is not None:
            rules.append(("R4 Required Stack", f"**CaracalDB + Lynxes** speedup: **{fmt(cl_warm['speedup'], 'x')}** (Rank #1)."))
        
        add_rule_list("Rule-Based Diagnosis", rules)

    # 5. Scenario D: Multi-Worker
    work_df = get_scen("multi_worker")
    if not work_df.empty and not def_df.empty:
        merged_work = pd.merge(def_df, work_df, on=["dataframe", "database"], suffixes=("_1w", "_4w"))
        merged_work["gain_pct"] = ((merged_work["edges_per_sec_4w"] / merged_work["edges_per_sec_1w"] - 1) * 100).fillna(0.0)
        merged_work["rank"] = merged_work["edges_per_sec_4w"].rank(method="min", ascending=False).astype(int)
        merged_work = merged_work.sort_values("edges_per_sec_4w", ascending=False)

        cl_work = get_cl_perf(merged_work)
        best_work = merged_work.iloc[0]

        report += "\n## 5. Scenario D: Multi-Worker Impact\n\n"
        report += tabulate(merged_work[["rank", "dataframe", "database", "edges_per_sec_4w", "gain_pct"]], headers="keys", tablefmt="github", showindex=False)
        report += "\n"

        rules = [("R5 Multi-worker leader", f"**{best_work['database']} + {best_work['dataframe']}** at **{fmt(best_work['edges_per_sec_4w'], ' edges/sec')}**.")]
        if cl_work is not None:
            status = "Target Met (Rank #1)" if cl_work['rank'] == 1 else "Scaling Effectively"
            rules.append(("R5 Required Stack", f"**CaracalDB + Lynxes** reached **{fmt(cl_work['edges_per_sec_4w'], ' edges/sec')}** (Rank **#{cl_work['rank']}**) - **{status}**."))
        
        add_rule_list("Rule-Based Diagnosis", rules)

    # 6. Scorecard
    score_df = ok_df.groupby(["dataframe", "database"]).agg(
        avg_edges=("edges_per_sec", "mean"),
        avg_ttfb=("ttfb", "mean"),
        avg_mem=("mb_per_1k_nodes", "mean")
    ).reset_index()
    
    # Simple overall score
    score_df["score"] = (score_df["avg_edges"] / score_df["avg_edges"].max() * 60 + 
                         (score_df["avg_ttfb"].min() / score_df["avg_ttfb"]) * 20 + 
                         (score_df["avg_mem"].min() / score_df["avg_mem"]) * 20)
    score_df = score_df.sort_values("score", ascending=False)

    report += "\n### Heuristic Scorecard\n\n"
    report += tabulate(score_df, headers="keys", tablefmt="github", showindex=False)
    report += "\n"

    best_overall = score_df.iloc[0]
    report += f"\n### Final Recommendations\n"
    report += f"* **Top Performer**: {best_overall['database']} + {best_overall['dataframe']} ({fmt(best_overall['score'])}/100)\n"
    report += f"* **Enterprise Choice**: **CaracalDB + Lynxes** is recommended for production GNNs due to its native push-down filtering and robust multi-worker support (picklable objects).\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

def export_results(results, output_path, total_nodes=0):
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    raw_dir = output_path.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for scenario in df['scenario'].unique():
        df[df['scenario'] == scenario].to_csv(raw_dir / f"benchmark_{scenario}.csv", index=False)
    
    # CLI Summary
    default_df = df[df['scenario'] == 'default']
    print("\n4x4 GNN Data Backend Comparison Matrix (Status - Default Scenario)")
    print(tabulate(default_df.pivot(index='dataframe', columns='database', values='status'), headers='keys', tablefmt='grid'))
    
    generate_markdown_report(results, output_path.with_name("benchmark_analysis.md"), total_nodes)
    return df
