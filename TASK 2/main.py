from __future__ import annotations

import argparse
from pathlib import Path

from src.compare_configs import run_config_comparison
from src.config import BenchmarkConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph ecosystem benchmark for Task 2.")
    parser.add_argument("command", nargs="?", default="compare-configs", choices=["compare-configs"])
    parser.add_argument("--dataset", choices=["toy", "multihoprag"], default=None)
    parser.add_argument("--documents", type=Path, default=None)
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--multihoprag-questions", type=Path, default=None)
    parser.add_argument("--multihoprag-corpus", type=Path, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--configs", default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--relation-depth", type=int, default=None)
    parser.add_argument("--embedding-dimension", type=int, default=None)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> BenchmarkConfig:
    base = BenchmarkConfig()
    configs = tuple(item.strip() for item in args.configs.split(",")) if args.configs else base.configs
    dataset = args.dataset or base.dataset
    embedding_dimension = args.embedding_dimension or (256 if dataset == "multihoprag" else base.embedding_dimension)
    return BenchmarkConfig(
        dataset=dataset,
        documents_dir=args.documents or base.documents_dir,
        questions_csv=args.questions or base.questions_csv,
        multihoprag_questions_json=args.multihoprag_questions or base.multihoprag_questions_json,
        multihoprag_corpus_json=args.multihoprag_corpus or base.multihoprag_corpus_json,
        processed_dir=base.processed_dir,
        output_dir=base.output_dir,
        caracal_db_path=base.caracal_db_path,
        vector_store_dir=base.vector_store_dir,
        chunk_size_tokens=base.chunk_size_tokens,
        chunk_overlap_tokens=base.chunk_overlap_tokens,
        embedding_dimension=embedding_dimension,
        top_k_semantic_candidates=args.top_k or base.top_k_semantic_candidates,
        relation_depth=args.relation_depth or base.relation_depth,
        repeats=args.repeats or base.repeats,
        max_questions=args.max_questions if args.max_questions is not None else base.max_questions,
        max_documents=args.max_documents if args.max_documents is not None else base.max_documents,
        configs=configs,
    )


def main() -> None:
    args = parse_args()
    config = make_config(args)
    results = run_config_comparison(config)
    for result in results:
        total = "-" if result.total_seconds is None else f"{result.total_seconds:.6f}s"
        print(
            f"{result.config_id}: {result.status} total={total} "
            f"semantic={result.semantic_entry_mode} relation={result.relation_expand_mode}"
        )
    print(f"Outputs: {config.output_dir}")


if __name__ == "__main__":
    main()
