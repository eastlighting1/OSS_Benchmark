# RAG Quality & Accuracy Summary

This report summarizes how well each configuration retrieves the correct context and generates accurate answers.

| Configuration             |   Context Recall (Accuracy) |   Answer Grounding |   Answer Similarity (F1) |   Avg Latency (s) |   Direct Match Rate |
|:--------------------------|----------------------------:|-------------------:|-------------------------:|------------------:|--------------------:|
| caracal-external-semantic |                    0.786667 |                  1 |                     0.34 |           0.30383 |                0.34 |
| caracal-only              |                    0.761667 |                  1 |                     0.32 |           0.88369 |                0.32 |
| neo4j-external-semantic   |                    0.508333 |                  1 |                     0.3  |          21.0749  |                0.3  |
| neo4j-only                |                    0.478333 |                  1 |                     0.29 |          21.5315  |                0.28 |

## Metric Definitions
- **Context Recall**: Probability that the retrieved documents contain the actual answer.
- **Answer Grounding**: Whether the LLM's answer is strictly based on the provided context (no hallucination).
- **Answer Similarity (F1)**: Overlap between the generated answer and the gold standard answer.
- **Direct Match Rate**: Percentage of answers that contain the exact key phrase from the gold data.
