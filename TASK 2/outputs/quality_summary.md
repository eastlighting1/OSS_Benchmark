# RAG Quality & Accuracy Summary

This report summarizes how well each configuration retrieves the correct context and generates accurate answers.

| Configuration             |   Context Recall (Accuracy) |   Answer Grounding |   Answer Similarity (F1) |   Avg Latency (s) |   Direct Match Rate |
|:--------------------------|----------------------------:|-------------------:|-------------------------:|------------------:|--------------------:|
| caracal-external-semantic |                    1        |                  1 |                 0.451429 |         0.0743499 |                 0.4 |
| caracal-only              |                    1        |                  1 |                 0.46     |         0.103749  |                 0.4 |
| neo4j-external-semantic   |                    0.933333 |                  1 |                 0.5      |         0.0449459 |                 0.2 |
| neo4j-only                |                    0.933333 |                  1 |                 0.4      |         0.0398564 |                 0.2 |

## Metric Definitions
- **Context Recall**: Probability that the retrieved documents contain the actual answer.
- **Answer Grounding**: Whether the LLM's answer is strictly based on the provided context (no hallucination).
- **Answer Similarity (F1)**: Overlap between the generated answer and the gold standard answer.
- **Direct Match Rate**: Percentage of answers that contain the exact key phrase from the gold data.
