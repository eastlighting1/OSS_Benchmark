# RAG Quality & Accuracy Summary

This report summarizes how well each configuration retrieves the correct context and generates accurate answers.

| Configuration             |   Context Recall (Accuracy) |   Answer Grounding |   Answer Similarity (F1) |   Avg Latency (s) |   Direct Match Rate |
|:--------------------------|----------------------------:|-------------------:|-------------------------:|------------------:|--------------------:|
| caracal-external-semantic |                    0.675    |                  1 |                 0.5      |         0.74272   |                 0.5 |
| caracal-only              |                    0.641667 |                  1 |                 0.585714 |         0.698825  |                 0.6 |
| neo4j-external-semantic   |                    0.791667 |                  1 |                 0.75     |         0.104358  |                 0.7 |
| neo4j-only                |                    0.766667 |                  1 |                 0.7      |         0.0798419 |                 0.7 |

## Metric Definitions
- **Context Recall**: Probability that the retrieved documents contain the actual answer.
- **Answer Grounding**: Whether the LLM's answer is strictly based on the provided context (no hallucination).
- **Answer Similarity (F1)**: Overlap between the generated answer and the gold standard answer.
- **Direct Match Rate**: Percentage of answers that contain the exact key phrase from the gold data.
