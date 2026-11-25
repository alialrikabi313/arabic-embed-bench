# arabic-embed-bench

A small CLI that answers one question: **which embedding model retrieves best on *your* Arabic documents?** Give it a corpus and questions with evidence quotes; it reports Hit@k, MRR with bootstrap confidence intervals, and encoding time per model.

## Why not just read MTEB?

Leaderboards average over tasks and languages. Retrieval quality on your own Arabic text (MSA, dialect-heavy, religious, legal…) can rank models differently, and a model's required prompt prefixes silently change results — `multilingual-e5` needs `query: ` / `passage: `, and omitting them degrades accuracy without any error. This tool bakes the prefixes into `config.MODEL_SPECS` and a test guards them.

## Example result

Arabic Wikipedia sample from [arabic-rag-bench](https://github.com/alialrikabi313/arabic-rag-bench) — 40 articles, 130 paragraph chunks, 80 LLM-generated questions:

| Model | Dim | Encode s | Hit@1 | Hit@3 | Hit@5 | MRR | MRR 95% CI |
|---|---|---|---|---|---|---|---|
| BAAI/bge-m3 | 1024 | 3.1 | 0.838 | 0.950 | 0.988 | 0.895 | 0.84–0.95 |
| intfloat/multilingual-e5-small | 384 | 0.4 | 0.825 | 0.938 | 0.963 | 0.884 | 0.82–0.94 |
| intfloat/multilingual-e5-base | 768 | 0.7 | 0.812 | 0.950 | 0.975 | 0.880 | 0.82–0.93 |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 0.2 | 0.613 | 0.787 | 0.825 | 0.719 | 0.64–0.80 |

**Reading it honestly:** the top three are statistically indistinguishable at this sample size (overlapping CIs). The practical finding is that `e5-small` (384-d, ~8× faster to encode) matches `bge-m3` here, while the paraphrase MiniLM model is clearly worse for retrieval. Encode times are on a GPU and cover chunks + questions; treat them as relative.

## Use

```bash
pip install -r requirements.txt
python embed_bench.py --corpus corpus.jsonl --questions questions.jsonl --out results.md
python embed_bench.py ... --models BAAI/bge-m3 intfloat/multilingual-e5-small --device cpu
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
