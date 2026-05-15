import argparse
import torch
from pathlib import Path
from src.config import get_config
from src.ingest import load_gnn_data, process_and_align_data
from src.backends.caracal_lynxes_backend import CaracalLynxesBackend
from src.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="GNN Data Backend Benchmark System")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data into caracaldb")
    ingest_parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    ingest_parser.add_argument("--data-dir", type=str, default="../data")
    
    # run-khop command
    khop_parser = subparsers.add_parser("run-khop", help="Evaluate K-hop sampling bottleneck")
    khop_parser.add_argument("--backend", type=str, default="caracal_lynxes")
    khop_parser.add_argument("--fanout", type=str, default="15,10")
    khop_parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    
    # run-feature-fetch command
    ffetch_parser = subparsers.add_parser("run-feature-fetch", help="Measure feature fetching speed")
    ffetch_parser.add_argument("--backend", type=str, default="caracal_lynxes")
    ffetch_parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    
    # run-filtered command
    filter_parser = subparsers.add_parser("run-filtered", help="Sample on filtered graph")
    filter_parser.add_argument("--timestamp", type=str, default="2023-01-01")
    filter_parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    
    # run-out-of-core command
    ooc_parser = subparsers.add_parser("run-out-of-core", help="Test out-of-core training")
    ooc_parser.add_argument("--dataset", type=str, default="ogbn-papers100M")
    
    # compare command
    compare_parser = subparsers.add_parser("compare", help="Run 4x4 comparison matrix")
    compare_parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    compare_parser.add_argument("--data-dir", type=str, default="../data")
    compare_parser.add_argument("--sample-nodes", type=int, default=50000)
    compare_parser.add_argument("--runs", type=int, default=3, help="Number of benchmark repetitions for median calculation")
    compare_parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion for all backends")

    args = parser.parse_args()

    if args.command == "ingest":
        config = get_config(dataset=args.dataset, data_dir=args.data_dir, sample_nodes=getattr(args, 'sample_nodes', 10000))
        node_feat_df, node_label_df, node_year_df, edge_df = load_gnn_data(config)
        backend = CaracalLynxesBackend(config)
        backend.ingest(node_feat_df, node_label_df, node_year_df, edge_df)
        print("Ingestion complete.")
        
    elif args.command == "run-khop":
        fanouts = [int(f) for f in args.fanout.split(",")]
        config = get_config(dataset=args.dataset, fanouts=fanouts, sample_nodes=getattr(args, 'sample_nodes', 10000))
        print(f"Running K-hop evaluation with fanout {fanouts} on {args.backend}...")
        
    elif args.command == "run-feature-fetch":
        config = get_config(dataset=args.dataset, sample_nodes=getattr(args, 'sample_nodes', 10000))
        print(f"Running feature fetch evaluation on {args.backend}...")
        
    elif args.command == "compare":
        from src.compare_systems import ComparisonSystem, export_results
        from src.ingest import load_gnn_data, get_df_len
        config = get_config(dataset=args.dataset, data_dir=args.data_dir, epochs=1, sample_nodes=args.sample_nodes, runs=args.runs)
        comparer = ComparisonSystem(config, skip_ingest=args.skip_ingest)
        results = comparer.run_benchmark()
        output_path = Path("outputs/comparison_benchmark.csv")
        
        # Load data once to get total_nodes for metrics
        node_feat_df, _, _, _ = load_gnn_data(config)
        total_nodes = get_df_len(node_feat_df)
        
        export_results(results, output_path, total_nodes=total_nodes)
        print(f"\nComparison results exported to {output_path}")

    elif args.command == "train" or args.command is None:
        dataset = args.dataset if hasattr(args, 'dataset') else "ogbn-arxiv"
        data_dir = args.data_dir if hasattr(args, 'data_dir') else "../data"
        epochs = args.epochs if hasattr(args, 'epochs') else 2
        batch_size = args.batch_size if hasattr(args, 'batch_size') else 1024
        sample_nodes = args.sample_nodes if hasattr(args, 'sample_nodes') else 10000
        
        config = get_config(dataset=dataset, data_dir=data_dir, epochs=epochs, batch_size=batch_size, sample_nodes=sample_nodes)
        node_feat_df, node_label_df, node_year_df, edge_df = load_gnn_data(config)
        backend = CaracalLynxesBackend(config)
        backend.ingest(node_feat_df, node_label_df, node_year_df, edge_df)
        
        in_channels = len([c for c in node_feat_df.column_names() if not c.startswith("_")])
        model_params = {"in_channels": in_channels, "hidden_channels": config.hidden_channels, "out_channels": 40, "num_layers": config.num_layers}
        
        run_pipeline(config, backend, model_params)
        backend.close()

if __name__ == "__main__":
    main()
