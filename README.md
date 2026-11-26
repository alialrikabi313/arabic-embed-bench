# arabic-embed-bench

A small CLI that answers one question: **which embedding model retrieves best on *your* Arabic documents?** Give it a corpus and questions with evidence quotes; it reports Hit@k, MRR with bootstrap confidence intervals, and encoding time per model.

[![tests](https://github.com/alialrikabi313/arabic-embed-bench/actions/workflows/test.yml/badge.svg)](https://github.com/alialrikabi313/arabic-embed-bench/actions/workflows/test.yml)

> **TL;DR** — `e5-small` matches `bge-m3` on Arabic retrieval within error at ~9× less encode time; a paraphrase model is far worse. Run the same comparison on your own documents.

## Why not just read MTEB?

Leaderboards average over tasks and languages. Retrieval quality on your own Arabic text (MSA, dialect-heavy, religious, legal…) can rank models differently, and a model's required prompt prefixes silently change results — `multilingual-e5` needs `query: ` / `passage: `, and omitting them degrades accuracy without any error. This tool bakes the prefixes into `config.MODEL_SPECS` and a test guards them.

## Example result

Arabic Wikipedia sample from [arabic-rag-bench](https://github.com/alialrikabi313/arabic-rag-bench) — 227 articles, 1,096 paragraph chunks, 442 LLM-generated questions:

| Model | Dim | Encode s | Hit@1 | Hit@3 | Hit@5 | MRR | MRR 95% CI |
|---|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-base | 768 | 66 | 0.790 | 0.930 | 0.957 | 0.865 | 0.84–0.89 |
| BAAI/bge-m3 | 1024 | 230 | 0.792 | 0.932 | 0.955 | 0.863 | 0.84–0.89 |
| intfloat/multilingual-e5-small | 384 | 26 | 0.796 | 0.914 | 0.939 | 0.861 | 0.83–0.89 |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 7 | 0.333 | 0.514 | 0.572 | 0.448 | 0.41–0.49 |

**Reading it honestly:** the top three are statistically indistinguishable (MRR intervals overlap almost entirely; Hit@1 differs by 0.006). The practical finding is that `e5-small` (384-d) matches `bge-m3` here at roughly **9× less encoding time**, while the paraphrase MiniLM model is far worse for retrieval — it is trained for sentence similarity, not question→passage search. Encode seconds are wall-clock for chunks + questions on a laptop that was busy with other jobs: trust the ratios, not the absolute values.

## Use

```bash
pip install "arabic-embed-bench[models] @ git+https://github.com/alialrikabi313/arabic-embed-bench"   # PyPI release pending
arabic-embed-bench --corpus corpus.jsonl --questions questions.jsonl --out results.md
arabic-embed-bench ... --models BAAI/bge-m3 intfloat/multilingual-e5-small --device cpu
```

Input (JSONL):

- `corpus.jsonl`: `{"pageid": 1, "title": "...", "text": "..."}`
- `questions.jsonl`: `{"pageid": 1, "question": "...", "quote": "verbatim sentence from that document containing the answer"}`

A chunk counts as correct if it comes from the right document and contains the middle word-window of the quote (the middle, so a chunk boundary cutting the quote's edge doesn't unfairly penalise a model). Chunking is fixed to paragraphs so the model is the only variable.

Add a model by adding its prompt prefixes to `config.MODEL_SPECS`; unknown models exit with an error rather than running unprefixed.

## Test

```bash
pytest tests
```

## Limits

- Small sample → wide intervals; use your own larger question set before deciding.
- Questions are LLM-generated, which can favour models that paraphrase like the generator.
- e5 models truncate at 512 tokens; long paragraphs (≤250 words) may be cut for them.
