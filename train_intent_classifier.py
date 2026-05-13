"""
Train a tiny offline intent classifier for the CornDoctor chatbot.

The output is a JSON multinomial Naive Bayes model that Flutter can run
directly without TensorFlow or network access:

    python train_intent_classifier.py

Output:
    flutter_app/assets/models/intent_classifier.json
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict


ROOT = os.path.dirname(__file__)
OUTPUT_FILE = os.path.join(
    ROOT, "flutter_app", "assets", "models", "intent_classifier.json"
)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "i", "me",
    "my", "we", "our", "you", "your", "it", "its", "this", "that",
    "these", "those", "about", "so", "just", "more", "also", "very",
}

EXAMPLES = {
    "overview": [
        "what is blight",
        "tell me about common rust",
        "explain gray leaf spot",
        "describe this disease",
        "what is northern corn leaf blight",
        "give me information about rust",
        "what does this diagnosis mean",
        "overview of corn disease",
    ],
    "symptoms": [
        "what are the symptoms",
        "what does blight look like",
        "how do i identify common rust",
        "signs of gray leaf spot",
        "why are there brown spots on my corn",
        "what are these lesions",
        "does rust make pustules",
        "how can i recognize the disease",
        "what leaf marks should i look for",
    ],
    "treatment": [
        "how do i treat blight",
        "what should i do",
        "how can i fix this",
        "how do i control gray leaf spot",
        "what is the treatment",
        "how do i manage rust",
        "can i cure this disease",
        "what action should i take",
        "help my corn is infected",
    ],
    "prevention": [
        "how can i prevent this",
        "how do i stop blight from spreading",
        "how to avoid common rust",
        "prevention tips",
        "how do i protect healthy corn",
        "what can prevent gray leaf spot",
        "how do i reduce future disease",
        "should i rotate crops",
        "what resistant hybrids help",
    ],
    "causes": [
        "what causes blight",
        "why does common rust happen",
        "what fungus causes gray leaf spot",
        "where does this disease come from",
        "why are my leaves infected",
        "what weather causes corn disease",
        "does rain cause this",
        "why did my corn get sick",
    ],
    "fungicides": [
        "what fungicide should i use",
        "what spray works for blight",
        "best chemical for gray leaf spot",
        "which product kills rust",
        "should i spray fungicide",
        "what medicine should i apply",
        "is azoxystrobin good",
        "when should i use quadris",
        "what product should i buy",
    ],
    "severity": [
        "how serious is this",
        "is blight bad",
        "will gray leaf spot reduce yield",
        "how much damage can rust cause",
        "is this dangerous for my crop",
        "what is the yield loss",
        "how severe is the infection",
        "should i be worried",
    ],
    "scouting": [
        "how do i scout for corn diseases",
        "how often should i check the field",
        "what leaves should i inspect",
        "how do i monitor disease",
        "where should i look for symptoms",
        "how many plants should i check",
        "when should i scout",
    ],
    "unknown": [
        "tell me a joke",
        "what is the weather today",
        "who are you",
        "open the camera",
        "how much does corn cost",
        "what is your favorite color",
        "call my friend",
        "play music",
    ],
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [token for token in text.split() if token and token not in STOPWORDS]


def train() -> dict:
    vocabulary: set[str] = set()
    class_counts: dict[str, int] = {}
    token_counts: dict[str, Counter[str]] = defaultdict(Counter)
    token_totals: dict[str, int] = defaultdict(int)

    total_docs = 0
    for intent, examples in EXAMPLES.items():
        class_counts[intent] = len(examples)
        total_docs += len(examples)
        for example in examples:
            tokens = tokenize(example)
            vocabulary.update(tokens)
            token_counts[intent].update(tokens)
            token_totals[intent] += len(tokens)

    vocab = sorted(vocabulary)
    vocab_size = len(vocab)
    alpha = 1.0

    priors = {
        intent: math.log(count / total_docs)
        for intent, count in sorted(class_counts.items())
    }

    token_log_probs: dict[str, dict[str, float]] = {}
    unknown_log_probs: dict[str, float] = {}
    for intent in sorted(class_counts):
        denominator = token_totals[intent] + alpha * vocab_size
        unknown_log_probs[intent] = math.log(alpha / denominator)
        token_log_probs[intent] = {
            token: round(
                math.log((token_counts[intent][token] + alpha) / denominator),
                6,
            )
            for token in vocab
        }

    return {
        "modelType": "multinomial_naive_bayes",
        "version": 1,
        "minConfidence": 0.26,
        "intents": sorted(class_counts),
        "vocabulary": vocab,
        "classLogPriors": {k: round(v, 6) for k, v in priors.items()},
        "tokenLogProbabilities": token_log_probs,
        "unknownTokenLogProbabilities": {
            k: round(v, 6) for k, v in unknown_log_probs.items()
        },
    }


def main() -> None:
    model = train()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(model, f, separators=(",", ":"))
    print(
        f"[OK] Wrote {OUTPUT_FILE} with "
        f"{len(model['intents'])} intents and {len(model['vocabulary'])} tokens"
    )


if __name__ == "__main__":
    main()
