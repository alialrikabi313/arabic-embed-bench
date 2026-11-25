# -*- coding: utf-8 -*-
"""مقارنة نماذج التضمين العربية على وثائقك وأسئلتك.

يقرأ: corpus.jsonl (pageid, title, text) و questions.jsonl (pageid, question, quote).
ينتج: جدول Markdown بـ Hit@k و MRR وفترات ثقة bootstrap وزمن التضمين لكل نموذج.
الخطوات: تقطيع بالفقرات ← تضمين المقاطع والأسئلة (مع بادئات النموذج) ← بحث cosine
← موضع أول مقطع صحيح ← مقاييس.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from config import (
    BOOTSTRAP_ROUNDS,
    ENCODE_BATCH,
    EVIDENCE_WINDOW_WORDS,
    MAX_CHUNK_WORDS,
    MODEL_SPECS,
    TOP_KS,
)

# عناوين ويكيبيديا النصّية: == عنوان ==
_HEADING_RE = re.compile(r"^=+\s*.+?\s*=+$")


def paragraph_chunks(text: str) -> list[str]:
    # الفقرة وحدة طبيعية؛ تُدمج الصغيرة حتى MAX، والفقرة الأطول من MAX تبقى كما هي
    paras = [p.strip() for p in text.split("\n") if p.strip() and not _HEADING_RE.match(p.strip())]
    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for p in paras:
        w = len(p.split())
        if cur and cur_words + w > MAX_CHUNK_WORDS:
            chunks.append(" ".join(cur))
            cur, cur_words = [], 0
        cur.append(p)
        cur_words += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def evidence_window(quote: str) -> str:
    words = quote.split()
    mid = len(words) // 2
    half = EVIDENCE_WINDOW_WORDS // 2
    return " ".join(words[max(0, mid - half) : mid + half])


def first_relevant_ranks(
    q_vecs: np.ndarray,
    c_vecs: np.ndarray,
    chunk_page: list[int],
    chunk_texts: list[str],
    questions: list[dict],
) -> list[int]:
    """موضع أول مقطع صحيح (من 1) لكل سؤال؛ الأسئلة بلا مقطع صحيح تُتجاهل."""
    order = np.argsort(-(q_vecs @ c_vecs.T), axis=1)
    norm_texts = [" ".join(t.split()) for t in chunk_texts]
    ranks = []
    for qi, q in enumerate(questions):
        window = evidence_window(q["quote"])
        relevant = np.array([p == q["pageid"] and window in t for p, t in zip(chunk_page, norm_texts)])
        if not relevant.any():
            continue
        ranks.append(int(np.nonzero(relevant[order[qi]])[0][0]) + 1)
    return ranks


def summarize(ranks: list[int], rng: np.random.Generator) -> dict:
    r = np.array(ranks)
    idx = rng.integers(0, len(r), size=(BOOTSTRAP_ROUNDS, len(r)))
    boot_mrr = (1.0 / r[idx]).mean(axis=1)
    return {
        "scored": len(r),
        **{f"hit@{k}": float((r <= k).mean()) for k in TOP_KS},
        "mrr": float((1.0 / r).mean()),
        "mrr_ci": (float(np.percentile(boot_mrr, 2.5)), float(np.percentile(boot_mrr, 97.5))),
    }


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"ملف غير موجود: {path}")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_model(name: str, device: str | None, chunk_texts, chunk_page, questions, rng) -> dict:
    # استيراد كسول: المكتبة ثقيلة ولا حاجة لها لباقي الدوال
    from sentence_transformers import SentenceTransformer

    spec = MODEL_SPECS[name]
    model = SentenceTransformer(name, device=device)
    t0 = time.time()
    c_vecs = model.encode(
        [spec["passage"] + t for t in chunk_texts], batch_size=ENCODE_BATCH, normalize_embeddings=True
    )
    q_vecs = model.encode(
        [spec["query"] + q["question"] for q in questions], batch_size=ENCODE_BATCH, normalize_embeddings=True
    )
    seconds = time.time() - t0
    ranks = first_relevant_ranks(q_vecs, c_vecs, chunk_page, chunk_texts, questions)
    return {"model": name, "dim": int(c_vecs.shape[1]), "seconds": round(seconds, 1), **summarize(ranks, rng)}


def to_markdown(results: list[dict], n_chunks: int) -> str:
    ks = [f"Hit@{k}" for k in TOP_KS]
    lines = [
        f"| Model | Dim | Encode s | Questions | {' | '.join(ks)} | MRR | MRR 95% CI |",
        "|---|---|---|---|" + "---|" * len(ks) + "---|---|",
    ]
    for r in sorted(results, key=lambda r: -r["mrr"]):
        cells = " | ".join(f"{r[f'hit@{k}']:.3f}" for k in TOP_KS)
        lines.append(
            f"| {r['model']} | {r['dim']} | {r['seconds']} | {r['scored']} | {cells} "
            f"| {r['mrr']:.3f} | {r['mrr_ci'][0]:.2f}–{r['mrr_ci'][1]:.2f} |"
        )
    return "\n".join(lines) + f"\n\n{n_chunks} chunks (paragraph chunking, ≤{MAX_CHUNK_WORDS} words).\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=list(MODEL_SPECS))
    ap.add_argument("--device", default=None, help="cuda|cpu (افتراضي: تلقائي)")
    ap.add_argument("--out", type=Path, default=Path("results.md"))
    args = ap.parse_args()

    unknown = [m for m in args.models if m not in MODEL_SPECS]
    if unknown:
        sys.exit(f"نماذج غير معرّفة في config.MODEL_SPECS: {unknown}\nأضف بادئاتها هناك أولاً.")

    questions = load_jsonl(args.questions)
    wanted = {q["pageid"] for q in questions}
    docs = [d for d in load_jsonl(args.corpus) if d["pageid"] in wanted]
    chunk_texts, chunk_page = [], []
    for d in docs:
        chunks = paragraph_chunks(d["text"])
        chunk_texts += chunks
        chunk_page += [d["pageid"]] * len(chunks)
    print(f"◄ {len(docs)} وثيقة ← {len(chunk_texts)} مقطع · {len(questions)} سؤال")

    rng = np.random.default_rng(0)
    results = []
    for name in args.models:
        print(f"← {name} …", flush=True)
        results.append(run_model(name, args.device, chunk_texts, chunk_page, questions, rng))
        print(f"✓ MRR {results[-1]['mrr']:.3f}", flush=True)

    args.out.write_text(to_markdown(results, len(chunk_texts)), encoding="utf-8")
    print(f"✓ {args.out}")


if __name__ == "__main__":
    main()
