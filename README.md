# Improving Domain Specific QA: Inter IIT Tech Meet 11.0, 2023

## Retriever

| Approach | Top-1 Accuracy | Top-5 Accuracy |
| -- | -- | -- |
| Cosine Similarity b/w MPNet (`multi-qa-mpnet-base`) Embeddings | 63.57 | 88.4 |
| BM25 Scores | 58.57 | 79.82 |
| MPNet Embeddings + BM25 | **70.67** | **89.63** |

> Accuracies on SQuAD-V2 dev set with theme information

## Reader

| Architecture | F1 | EM |
| -- | -- | -- |
| `BERT-base` | 74.67 | 71.15 |
| `ELECTRA-base` | 81.71 | 77.60 |
| `DeBERTa-V3-base` | **87.41** | **83.92** |

> F1 and EM on SQuAD-V2 dev set

## Domain Adaptation

### Retriever

| Approach | Top-1 Accuracy | Top-5 Accuracy |
| -- | -- | -- |
| `multi-qa-mpnet-base` | **63.57** | **88.4** |
| GPL (`multi-qa-mpnet-base`) | 66.5 | 86.4 |
| LaPraDoR (checkpoint not trained on SQuADV2 Retrieval) | 51.2 | 79.9 |

### Reader

| Approach | F1 | EM |
| -- | -- | -- |
| `BERT-base` zero shot | 74.67 | 71.15 |
| CAQA (Synthetic - QAGen-T5-base) | 72.42 | 68.91 |
| CAQA (No Synthetic Data) | 76.27 | 72.87 |
| QADA (4 epochs) | 76.50 | 73.23 |

| Approach | F1 | EM |
| -- | -- | -- |
| `DeBERTa-V3-base` zero shot | 87.41 | 83.92 |
| CAQA (Synthetic - QAGen-T5-base) | 86.12 | 82.68 |
| CAQA (No Synthetic Data) | 88.93 | 85.07 |