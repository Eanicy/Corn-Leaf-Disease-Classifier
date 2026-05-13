import 'dart:convert';
import 'dart:math';
import 'package:flutter/services.dart';

class IntentPrediction {
  final String intent;
  final double confidence;

  const IntentPrediction({
    required this.intent,
    required this.confidence,
  });

  bool get isConfident => intent != 'unknown' && confidence >= 0.26;
}

class IntentClassifier {
  static const String _modelFileName =
      'assets/models/intent_classifier.json';

  Map<String, dynamic> _model = {};
  bool _loaded = false;

  static const Set<String> _stopwords = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'need', 'i', 'me',
    'my', 'we', 'our', 'you', 'your', 'it', 'its', 'this', 'that',
    'these', 'those', 'about', 'so', 'just', 'more', 'also', 'very',
  };

  Future<void> load() async {
    if (_loaded) return;
    final raw = await rootBundle.loadString(_modelFileName);
    _model = json.decode(raw) as Map<String, dynamic>;
    _loaded = true;
  }

  IntentPrediction classify(String text) {
    if (!_loaded) {
      return const IntentPrediction(intent: 'unknown', confidence: 0.0);
    }

    final tokens = _tokenize(text);
    if (tokens.isEmpty) {
      return const IntentPrediction(intent: 'unknown', confidence: 0.0);
    }

    final intents = List<String>.from(_model['intents'] as List<dynamic>);
    final priors = Map<String, double>.from(
      (_model['classLogPriors'] as Map<String, dynamic>).map(
        (k, v) => MapEntry(k, (v as num).toDouble()),
      ),
    );
    final unknownLogProbabilities = Map<String, double>.from(
      (_model['unknownTokenLogProbabilities'] as Map<String, dynamic>).map(
        (k, v) => MapEntry(k, (v as num).toDouble()),
      ),
    );
    final tokenLogProbabilities =
        _model['tokenLogProbabilities'] as Map<String, dynamic>;

    final scores = <String, double>{};
    for (final intent in intents) {
      var score = priors[intent] ?? double.negativeInfinity;
      final intentTokenProbs = Map<String, double>.from(
        (tokenLogProbabilities[intent] as Map<String, dynamic>).map(
          (k, v) => MapEntry(k, (v as num).toDouble()),
        ),
      );
      final unknownTokenScore = unknownLogProbabilities[intent] ?? -12.0;

      for (final token in tokens) {
        score += intentTokenProbs[token] ?? unknownTokenScore;
      }
      scores[intent] = score;
    }

    final probabilities = _softmax(scores);
    var bestIntent = 'unknown';
    var bestConfidence = 0.0;
    for (final entry in probabilities.entries) {
      if (entry.value > bestConfidence) {
        bestIntent = entry.key;
        bestConfidence = entry.value;
      }
    }

    final minConfidence = (_model['minConfidence'] as num?)?.toDouble() ?? 0.26;
    if (bestConfidence < minConfidence) {
      return IntentPrediction(
        intent: 'unknown',
        confidence: bestConfidence,
      );
    }

    return IntentPrediction(
      intent: bestIntent,
      confidence: bestConfidence,
    );
  }

  List<String> _tokenize(String text) {
    return text
        .toLowerCase()
        .replaceAll(RegExp(r"[^a-z0-9\s]"), ' ')
        .split(RegExp(r'\s+'))
        .where((t) => t.isNotEmpty && !_stopwords.contains(t))
        .toList();
  }

  Map<String, double> _softmax(Map<String, double> scores) {
    final maxScore = scores.values.reduce(max);
    final expScores = <String, double>{};
    var total = 0.0;

    for (final entry in scores.entries) {
      final value = exp(entry.value - maxScore);
      expScores[entry.key] = value;
      total += value;
    }

    return expScores.map((k, v) => MapEntry(k, v / total));
  }
}
