| Model | Dim | Encode s | Questions | Hit@1 | Hit@3 | Hit@5 | MRR | MRR 95% CI |
|---|---|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-base | 768 | 65.7 | 442 | 0.790 | 0.930 | 0.957 | 0.865 | 0.84–0.89 |
| BAAI/bge-m3 | 1024 | 229.7 | 442 | 0.792 | 0.932 | 0.955 | 0.863 | 0.84–0.89 |
| intfloat/multilingual-e5-small | 384 | 26.0 | 442 | 0.796 | 0.914 | 0.939 | 0.861 | 0.83–0.89 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 384 | 7.2 | 442 | 0.333 | 0.514 | 0.572 | 0.448 | 0.41–0.49 |

1096 chunks (paragraph chunking, ≤250 words).
