# -*- coding: utf-8 -*-
"""اختبارات. الحالات من عيّنة arabic-rag-bench الحقيقية؛ اختبار التكامل يُتخطّى إن غابت."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import embed_bench as eb
from config import MODEL_SPECS

_DATA = Path(__file__).resolve().parent.parent.parent / "arabic-rag-bench-data"


def test_e5_models_keep_query_passage_prefixes():
    # بدون «query: / passage: » تنخفض دقة e5 بصمت؛ لا يجوز أن يُحذف أحدهما
    for name, spec in MODEL_SPECS.items():
        if "e5" in name:
            assert spec == {"query": "query: ", "passage": "passage: "}


def test_evidence_window_is_from_the_middle_of_a_real_short_quote():
    # اقتباس حقيقي قصير (8 كلمات) من مقالة «كتيبة إس إس بانزر الثقيلة 102»:
    # الوسط يستبعد الطرفين («في» و«للإصلاح.») اللذين قد يشطرهما حدّ المقطع
    quote = "في سبتمبر 1944 تم سحبها إلى ألمانيا للإصلاح."
    assert eb.evidence_window(quote) == "سبتمبر 1944 تم سحبها إلى ألمانيا"


def test_unknown_model_exits_with_message(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--corpus", "a", "--questions", "b", "--models", "no/such-model"])
    with pytest.raises(SystemExit) as e:
        eb.main()
    assert "no/such-model" in str(e.value)


def test_paragraph_chunks_drop_wikipedia_headings_and_respect_cap():
    text = "== مراجع ==\n" + "\n".join(["كلمة " * 100] * 4)
    chunks = eb.paragraph_chunks(text)
    assert all("==" not in c for c in chunks)
    assert all(len(c.split()) <= eb.MAX_CHUNK_WORDS for c in chunks)


def test_real_data_ranks_are_valid_with_bge_m3():
    q_path, c_path = _DATA / "questions.jsonl", _DATA / "wiki_sample.jsonl"
    if not (q_path.exists() and c_path.exists()):
        pytest.skip("بيانات arabic-rag-bench غير موجودة")
    questions = [json.loads(l) for l in q_path.open(encoding="utf-8")][:16]
    wanted = {q["pageid"] for q in questions}
    docs = [d for d in map(json.loads, c_path.open(encoding="utf-8")) if d["pageid"] in wanted]
    texts, pages = [], []
    for d in docs:
        c = eb.paragraph_chunks(d["text"])
        texts += c
        pages += [d["pageid"]] * len(c)

    out = eb.run_model("BAAI/bge-m3", None, texts, pages, questions, np.random.default_rng(0))
    assert out["scored"] >= 10
    assert out["mrr"] > 0.5 and 0 <= out["hit@1"] <= out["hit@5"] <= 1
    assert out["mrr_ci"][0] <= out["mrr"] <= out["mrr_ci"][1]
