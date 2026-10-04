# Evaluation Report

## Summary
- Dataset: microsoft/ms_marco
- Split: validation
- Queries evaluated: 29
- Status: complete
- Model: sentence-transformers/all-MiniLM-L6-v2
- Top-k: 5

## Metrics
| Metric | Value |
| --- | ---: |
| Context Precision | 0.6603 |
| Recall@5 | 0.8966 |
| MRR@5 | 0.5902 |

## Scope
- Collection: msmarco_qwen3_embedding_0_6b
- Indexed passages: 100,000
- Validation queries checked: 9,696
- Queries with indexed references: 29
- Selected reference contexts in index: 27
- Selected reference contexts checked: 10,705

## Notes
- The evaluation is a partial-index, filtered assessment: it includes only queries with at least one exact normalized selected reference passage present in the collection.
- Context precision was computed using the RAGAS `NonLLMContextPrecisionWithReference` method against selected MS MARCO reference passage texts.
- Context recall is unavailable because this setup does not include an LLM judge or passage-ID qrels for full semantic coverage.
