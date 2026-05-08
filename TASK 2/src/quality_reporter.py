from __future__ import annotations

import pandas as pd
from pathlib import Path

def generate_quality_summary(evaluation_csv: Path, output_path: Path):
    if not evaluation_csv.exists():
        return

    df = pd.read_csv(evaluation_csv)
    
    # Calculate averages per config
    summary = df.groupby('config_id').agg({
        'context_recall': 'mean',
        'answer_grounding_score': 'mean',
        'answer_token_f1': 'mean',
        'latency_seconds': 'mean',
        'answer_contains_gold': 'mean'
    }).reset_index()
    
    # Rename columns for readability
    summary.columns = ['Configuration', 'Context Recall (Accuracy)', 'Answer Grounding', 'Answer Similarity (F1)', 'Avg Latency (s)', 'Direct Match Rate']
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# RAG Quality & Accuracy Summary\n\n")
        f.write("This report summarizes how well each configuration retrieves the correct context and generates accurate answers.\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n## Metric Definitions\n")
        f.write("- **Context Recall**: Probability that the retrieved documents contain the actual answer.\n")
        f.write("- **Answer Grounding**: Whether the LLM's answer is strictly based on the provided context (no hallucination).\n")
        f.write("- **Answer Similarity (F1)**: Overlap between the generated answer and the gold standard answer.\n")
        f.write("- **Direct Match Rate**: Percentage of answers that contain the exact key phrase from the gold data.\n")

if __name__ == "__main__":
    generate_quality_summary(Path("outputs/evaluation_report.csv"), Path("outputs/quality_summary.md"))
