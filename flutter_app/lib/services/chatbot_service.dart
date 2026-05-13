import 'dart:convert';
import 'package:flutter/services.dart';
import 'intent_classifier.dart';
import 'retrieval_service.dart';

class ChatbotService {
  final RetrievalService _retrieval = RetrievalService();
  final IntentClassifier _intentClassifier = IntentClassifier();
  Map<String, dynamic> _knowledgeBase = {};
  String? _lastDisease;
  String? _lastIntent;
  bool _initialized = false;

  Future<void> initialize() async {
    if (_initialized) return;
    await Future.wait([
      _retrieval.load(),
      _intentClassifier.load(),
      _loadKnowledgeBase(),
    ]);
    _initialized = true;
  }

  Future<void> _loadKnowledgeBase() async {
    final raw = await rootBundle.loadString('assets/data/disease_knowledge.json');
    _knowledgeBase = json.decode(raw) as Map<String, dynamic>;
  }

  /// Returns a response string for the given user query.
  /// [diseaseContext] is the detected disease from the classification screen.
  Future<String> getResponse(String query, {String? diseaseContext}) async {
    if (!_initialized) await initialize();

    final resolvedDisease = _resolveDisease(query, diseaseContext);
    final prediction = _intentClassifier.classify(query);

    if (prediction.isConfident) {
      final answer = _answerFromKnowledgeBase(
        prediction.intent,
        disease: resolvedDisease,
      );
      if (answer != null && answer.isNotEmpty) {
        _lastIntent = prediction.intent;
        _lastDisease = resolvedDisease ?? _lastDisease;
        return answer;
      }
    }

    final expanded = _expandQuery(query);
    final result = _retrieval.query(expanded, diseaseHint: resolvedDisease);

    if (result.score == 0.0 || result.answer.isEmpty) {
      return "I'm not sure about that. Try asking about corn disease symptoms, "
          "treatment, fungicides, or prevention strategies.";
    }

    _lastDisease = resolvedDisease ?? _lastDisease;
    return result.answer;
  }

  String? _answerFromKnowledgeBase(String intent, {String? disease}) {
    if (intent == 'scouting') {
      return _findFaqAnswer('scout');
    }

    final diseaseEntry = _findDiseaseEntry(disease);
    if (diseaseEntry == null) {
      return _answerGeneralIntent(intent);
    }

    final answer = diseaseEntry[intent];
    return answer is String ? answer : null;
  }

  String? _answerGeneralIntent(String intent) {
    switch (intent) {
      case 'prevention':
        return _findFaqAnswer('prevent');
      case 'causes':
        return _findFaqAnswer('causes');
      case 'fungicides':
        return _findFaqAnswer('fungicides');
      case 'symptoms':
        return _findFaqAnswer('identify');
      case 'treatment':
        return _findFaqAnswer('fungicides');
      default:
        return null;
    }
  }

  String? _findFaqAnswer(String needle) {
    final faq = _knowledgeBase['faq'];
    if (faq is! List) return null;

    for (final item in faq) {
      if (item is! Map<String, dynamic>) continue;
      final question = (item['question'] as String? ?? '').toLowerCase();
      if (question.contains(needle)) {
        return item['answer'] as String?;
      }
    }
    return null;
  }

  Map<String, dynamic>? _findDiseaseEntry(String? disease) {
    if (disease == null || disease.isEmpty || disease.contains('Unknown')) {
      return null;
    }

    final diseases = _knowledgeBase['diseases'];
    if (diseases is! List) return null;

    final normalized = _normalizeDiseaseName(disease);
    for (final item in diseases) {
      if (item is! Map<String, dynamic>) continue;
      if (_normalizeDiseaseName(item['name'] as String? ?? '') == normalized) {
        return item;
      }
    }
    return null;
  }

  String? _resolveDisease(String query, String? diseaseContext) {
    final detectedFromQuery = _detectDiseaseMention(query);
    if (detectedFromQuery != null) return detectedFromQuery;

    if (diseaseContext != null &&
        diseaseContext.isNotEmpty &&
        !diseaseContext.contains('Unknown')) {
      return diseaseContext;
    }

    final lower = query.toLowerCase();
    final hasFollowUpReference =
        RegExp(r'\b(it|this|that|disease|infection|problem)\b').hasMatch(lower) ||
        query.trim().split(RegExp(r'\s+')).length <= 3;

    if (hasFollowUpReference) return _lastDisease;
    return null;
  }

  String? _detectDiseaseMention(String query) {
    final q = query.toLowerCase();
    if (q.contains('gray leaf') || q.contains('grey leaf')) {
      return 'Gray_Leaf_Spot';
    }
    if (q.contains('common rust') || RegExp(r'\brust\b').hasMatch(q)) {
      return 'Common_Rust';
    }
    if (q.contains('blight') || q.contains('northern leaf')) {
      return 'Blight';
    }
    if (q.contains('healthy')) {
      return 'Healthy';
    }
    return null;
  }

  String _normalizeDiseaseName(String value) {
    return value
        .toLowerCase()
        .replaceAll('_', ' ')
        .replaceAll(RegExp(r'[^a-z0-9]+'), ' ')
        .trim();
  }

  /// Expands query with intent-signal words to improve retrieval fallback.
  String _expandQuery(String query) {
    final q = query.toLowerCase();
    final extra = <String>[];

    if (_lastIntent != null && query.trim().split(RegExp(r'\s+')).length <= 3) {
      extra.add(_lastIntent!);
    }

    if (q.contains('what to do') ||
        q.contains('what should i do') ||
        q.contains('what do i do') ||
        q.contains('how to fix') ||
        q.contains('how to handle') ||
        q.contains('what can i do') ||
        RegExp(r'\bdo\b.{0,20}\bfor\b').hasMatch(q)) {
      extra.add('treat treatment remedy control manage');
    }

    if (q.contains('when') ||
        q.contains('how often') ||
        q.contains('what time') ||
        q.contains('what stage')) {
      extra.add('timing when spray stage schedule apply');
    }

    if (extra.isEmpty) return query;
    return '$query ${extra.join(' ')}';
  }

  /// Returns a list of suggested questions, optionally scoped to a disease.
  List<String> getSuggestedQuestions({String? disease}) {
    if (disease != null && disease.isNotEmpty && disease != 'Healthy') {
      final d = disease.replaceAll('_', ' ');
      return [
        'What are the symptoms of $d?',
        'How do I treat $d?',
        'What fungicides work for $d?',
        'How can I prevent $d?',
      ];
    }
    return [
      'What causes corn leaf diseases?',
      'How can I prevent corn diseases?',
      'When should I apply fungicides?',
      'What fungicides work best?',
      'How do I scout for corn diseases?',
    ];
  }
}
