| Model | Dim | Encode s | Questions | Hit@1 | Hit@3 | Hit@5 | MRR | MRR 95% CI |
|---|---|---|---|---|---|---|---|---|
| BAAI/bge-m3 | 1024 | 3.1 | 80 | 0.838 | 0.950 | 0.988 | 0.895 | 0.84–0.95 |
| intfloat/multilingual-e5-small | 384 | 0.4 | 80 | 0.825 | 0.938 | 0.963 | 0.884 | 0.82–0.94 |
| intfloat/multilingual-e5-base | 768 | 0.7 | 80 | 0.812 | 0.950 | 0.975 | 0.880 | 0.82–0.93 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 384 | 0.2 | 80 | 0.613 | 0.787 | 0.825 | 0.719 | 0.64–0.80 |

130 chunks (paragraph chunking, ≤250 words).
