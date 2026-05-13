import 'dart:convert';
import 'dart:math';
import 'package:flutter/services.dart';

class RetrievalResult {
  final String answer;
  final String matchedQuestion;
  final double score;

  const RetrievalResult({
    required this.answer,
    required this.matchedQuestion,
    required this.score,
  });
}

class RetrievalService {
  static const double kMinScore = 0.50;

  // TF-IDF index loaded from asset
  Map<String, dynamic> _index = {};
  bool _loaded = false;

  // Stopwords — must include "me" to prevent false positives
  static const Set<String> _stopwords = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'need', 'dare',
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it',
    'its', 'they', 'their', 'this', 'that', 'these', 'those',
    'what', 'which', 'who', 'how', 'about', 'if', 'so', 'up', 'out',
    'as', 'into', 'than', 'then', 'just', 'more', 'also', 'not',
    'no', 'very', 'much', 'some', 'any', 'all',
  };

  Future<void> load() async {
    if (_loaded) return;
    final raw = await rootBundle.loadString('assets/data/kb_retrieval_index.json');
    _index = json.decode(raw) as Map<String, dynamic>;
    _loaded = true;
  }

  List<String> _tokenize(String text) {
    return text
        .toLowerCase()
        .replaceAll(RegExp(r"[^a-z0-9\s]"), ' ')
        .split(RegExp(r'\s+'))
        .where((t) => t.isNotEmpty && !_stopwords.contains(t))
        .toList();
  }

  Map<String, double> _buildQueryVector(List<String> tokens) {
    final idf = Map<String, double>.from(
      (_index['idf'] as Map<String, dynamic>).map(
        (k, v) => MapEntry(k, (v as num).toDouble()),
      ),
    );

    // Term frequency
    final tf = <String, int>{};
    for (final t in tokens) {
      tf[t] = (tf[t] ?? 0) + 1;
    }

    final vec = <String, double>{};
    tf.forEach((term, count) {
      if (idf.containsKey(term)) {
        vec[term] = count / tokens.length * idf[term]!;
      }
    });

    // L2 normalize
    final norm = sqrt(vec.values.fold(0.0, (s, v) => s + v * v));
    if (norm > 0) {
      vec.updateAll((k, v) => v / norm);
    }
    return vec;
  }

  double _cosineSimilarity(Map<String, double> qVec, Map<String, double> dVec) {
    double dot = 0.0;
    for (final entry in qVec.entries) {
      final dVal = dVec[entry.key];
      if (dVal != null) dot += entry.value * dVal;
    }
    return dot; // Both already L2-normalized
  }

  RetrievalResult query(String queryText, {String? diseaseHint}) {
    if (!_loaded) {
      return const RetrievalResult(
        answer: '',
        matchedQuestion: '',
        score: 0.0,
      );
    }

    String effectiveQuery = queryText;
    final lower = queryText.toLowerCase().trim();

    // Disease-hint boost — pronoun references or short action verbs only
    if (diseaseHint != null && diseaseHint.isNotEmpty) {
      final hasPronounRef =
          RegExp(r'\b(it|this|that)\b').hasMatch(lower) ||
          lower.contains('the disease') ||
          lower.contains('my corn');
      final isActionVerb =
          RegExp(r'\b(fix|treat|help|handle|cure|stop|prevent|manage|control)\b')
              .hasMatch(lower) &&
          _tokenize(queryText).length <= 2;

      if (hasPronounRef || isActionVerb) {
        effectiveQuery =
            '$queryText ${diseaseHint.toLowerCase().replaceAll('_', ' ')}';
      }
    }

    final tokens = _tokenize(effectiveQuery);
    if (tokens.isEmpty) {
      return const RetrievalResult(answer: '', matchedQuestion: '', score: 0.0);
    }

    final qVec = _buildQueryVector(tokens);
    if (qVec.isEmpty) {
      return const RetrievalResult(answer: '', matchedQuestion: '', score: 0.0);
    }

    final entries = _index['entries'] as List<dynamic>;

    double bestScore = 0.0;
    String bestAnswer = '';
    String bestQuestion = '';

    for (final entry in entries) {
      final dVec = Map<String, double>.from(
        (entry['vec'] as Map<String, dynamic>).map(
          (k, v) => MapEntry(k, (v as num).toDouble()),
        ),
      );
      final score = _cosineSimilarity(qVec, dVec);
      if (score > bestScore) {
        bestScore = score;
        bestAnswer = entry['answer'] as String;
        bestQuestion = entry['question'] as String;
      }
    }

    // Single-token guard: slightly higher threshold for single-token queries
    // (just enough to filter out cases like "tell me a joke" → ['tell'] which
    // scores around 0.45, while still allowing "what is blight" → ['blight'])
    final singleToken = qVec.length == 1;
    final effectiveThreshold = singleToken ? 0.55 : kMinScore;

    if (bestScore < effectiveThreshold) {
      return const RetrievalResult(answer: '', matchedQuestion: '', score: 0.0);
    }

    return RetrievalResult(
      answer: bestAnswer,
      matchedQuestion: bestQuestion,
      score: bestScore,
    );
  }
}
