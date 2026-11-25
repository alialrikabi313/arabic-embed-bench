# -*- coding: utf-8 -*-
"""الإعدادات: النماذج المقارَنة وبادئاتها. عدّل هنا فقط عند إضافة نموذج أو تغيير القياس."""

# بادئة الاستعلام/المستند لكل نموذج. e5 تُدرَّب على «query: / passage: » وبدونها
# تنخفض الدقة بصمت؛ bge-m3 والباقي لا يحتاجان بادئة
MODEL_SPECS: dict[str, dict[str, str]] = {
    "BAAI/bge-m3": {"query": "", "passage": ""},
    "intfloat/multilingual-e5-base": {"query": "query: ", "passage": "passage: "},
    "intfloat/multilingual-e5-small": {"query": "query: ", "passage": "passage: "},
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {"query": "", "passage": ""},
}

# كل المقارنة بالتقطيع بالفقرات (الأفضل في arabic-rag-bench)، فيُعزل أثر النموذج وحده
MAX_CHUNK_WORDS = 250

# الدليل الصحيح = مقطع من نفس الوثيقة يحوي نافذة الكلمات الوسطى من الاقتباس
EVIDENCE_WINDOW_WORDS = 6

TOP_KS = (1, 3, 5)
ENCODE_BATCH = 32
BOOTSTRAP_ROUNDS = 2000
