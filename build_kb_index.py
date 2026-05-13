"""
Build TF-IDF retrieval index from disease_knowledge.json.
Outputs: flutter_app/assets/data/kb_retrieval_index.json

Each disease has structured fields (symptoms, treatment, prevention, causes,
fungicides, severity, overview) — we create a focused Q&A entry per field
so different questions about the same disease return different answers.

Run with: python build_kb_index.py
"""

import json
import math
import re
import os

# ── Paths ────────────────────────────────────────────────────────────────────
KNOWLEDGE_FILE = os.path.join(
    os.path.dirname(__file__),
    "flutter_app", "assets", "data", "disease_knowledge.json"
)
OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__),
    "flutter_app", "assets", "data", "kb_retrieval_index.json"
)

# ── Stopwords (must include "me") ────────────────────────────────────────────
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'need', 'dare',
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it',
    'its', 'they', 'their', 'this', 'that', 'these', 'those',
    'what', 'which', 'who', 'how', 'about', 'if', 'so', 'up', 'out',
    'as', 'into', 'than', 'then', 'just', 'more', 'also', 'not',
    'no', 'very', 'much', 'some', 'any', 'all',
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", ' ', text)
    return [t for t in text.split() if t and t not in STOPWORDS]


# ── Per-aspect question templates and intent-signal tokens ───────────────────
# For every disease, we generate ONE entry per aspect.
# `questions` = phrasings users might type
# `boost_tokens` = intent signal added to the vector so vague queries still hit
ASPECT_TEMPLATES = {
    'overview': {
        'questions': [
            "What is {d}?",
            "Tell me about {d}",
            "Describe {d}",
            "Explain {d}",
        ],
        'boost_tokens': ['overview', 'about', 'describe', 'explain', 'info'],
    },
    'symptoms': {
        'questions': [
            "What are the symptoms of {d}?",
            "How do I identify {d}?",
            "What does {d} look like?",
            "Signs of {d}",
        ],
        'boost_tokens': ['symptom', 'symptoms', 'sign', 'signs', 'identify',
                         'look', 'appear', 'lesion', 'spot'],
    },
    'treatment': {
        'questions': [
            "How do I treat {d}?",
            "How to control {d}",
            "How to manage {d}",
            "Treatment for {d}",
            "How do I cure {d}?",
        ],
        'boost_tokens': ['treat', 'treatment', 'cure', 'control', 'manage',
                         'remedy', 'fix', 'handle', 'stop'],
    },
    'prevention': {
        'questions': [
            "How can I prevent {d}?",
            "How to avoid {d}",
            "Prevention of {d}",
            "How to stop {d} from spreading",
        ],
        'boost_tokens': ['prevent', 'prevention', 'avoid', 'stop', 'reduce',
                         'rotation', 'resistant'],
    },
    'causes': {
        'questions': [
            "What causes {d}?",
            "Why does {d} happen?",
            "What is the cause of {d}?",
        ],
        'boost_tokens': ['cause', 'causes', 'reason', 'why', 'fungus',
                         'pathogen', 'origin'],
    },
    'fungicides': {
        'questions': [
            "What fungicides work for {d}?",
            "Best fungicide for {d}",
            "Which chemical kills {d}?",
            "What spray works against {d}?",
        ],
        'boost_tokens': ['fungicide', 'fungicides', 'chemical', 'spray',
                         'product', 'azoxystrobin', 'pyraclostrobin'],
    },
    'severity': {
        'questions': [
            "How serious is {d}?",
            "How bad is {d}?",
            "Does {d} cause yield loss?",
            "Impact of {d}",
        ],
        'boost_tokens': ['serious', 'severity', 'bad', 'damage', 'yield',
                         'loss', 'impact'],
    },
}


def build_index():
    with open(KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
        kb = json.load(f)

    entries = []

    # ── Disease-aspect entries ──────────────────────────────────────────────
    for disease in kb.get('diseases', []):
        name = disease['name']
        display_name = name.replace('_', ' ')

        for aspect, template in ASPECT_TEMPLATES.items():
            answer = disease.get(aspect)
            if not answer:
                continue
            # Create one entry per question phrasing, each with the same
            # focused per-aspect answer.
            for q_template in template['questions']:
                question = q_template.format(d=display_name)
                tokens = tokenize(question)
                tokens += tokenize(display_name)  # disease name (always)
                tokens += template['boost_tokens']  # intent signal
                entries.append({
                    'question': question,
                    'answer': answer,
                    'tokens': tokens,
                })

    # ── Generic FAQ entries ─────────────────────────────────────────────────
    for faq in kb.get('faq', []):
        question = faq['question']
        answer = faq['answer']
        # Index question text + lightweight intent boost based on question
        tokens = tokenize(question)
        # Heuristic intent boost on FAQ entries
        ql = question.lower()
        if any(w in ql for w in ('treat', 'control', 'manage', 'cure')):
            tokens += ['treat', 'treatment', 'control', 'manage']
        if any(w in ql for w in ('prevent', 'avoid', 'stop')):
            tokens += ['prevent', 'prevention', 'avoid']
        if any(w in ql for w in ('when', 'time', 'stage', 'often')):
            tokens += ['when', 'timing', 'stage', 'schedule']
        if any(w in ql for w in ('fungicide', 'chemical', 'spray', 'product')):
            tokens += ['fungicide', 'chemical', 'spray', 'product']
        if 'scout' in ql or 'identify' in ql or 'find' in ql:
            tokens += ['scout', 'identify', 'detect', 'find']
        entries.append({
            'question': question,
            'answer': answer,
            'tokens': tokens,
        })

    # ── Build IDF ────────────────────────────────────────────────────────────
    n_docs = len(entries)
    df: dict[str, int] = {}
    for entry in entries:
        for term in set(entry['tokens']):
            df[term] = df.get(term, 0) + 1

    idf: dict[str, float] = {
        term: math.log((n_docs + 1) / (freq + 1)) + 1.0
        for term, freq in df.items()
    }

    # ── Build TF-IDF vectors (L2 normalized) ─────────────────────────────────
    def make_vec(tokens: list[str]) -> dict[str, float]:
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec: dict[str, float] = {}
        n = len(tokens)
        for term, count in tf.items():
            if term in idf:
                vec[term] = (count / n) * idf[term]
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            vec = {k: v / norm for k, v in vec.items()}
        return vec

    index_entries = []
    for entry in entries:
        vec = make_vec(entry['tokens'])
        index_entries.append({
            'question': entry['question'],
            'answer': entry['answer'],
            'vec': {k: round(v, 6) for k, v in vec.items()},
        })

    # ── Write output ─────────────────────────────────────────────────────────
    output = {
        'idf': {k: round(v, 6) for k, v in idf.items()},
        'entries': index_entries,
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, separators=(',', ':'))

    size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"[OK] Index built: {len(index_entries)} entries, "
          f"{len(idf)} vocab tokens, {size_kb:.1f} KB")
    print(f"   Output: {OUTPUT_FILE}")
    print(f"   'me' in IDF: {'me' in idf}")


if __name__ == '__main__':
    build_index()
