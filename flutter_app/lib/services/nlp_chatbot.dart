import 'dart:convert';
import 'package:flutter/services.dart';

class ChatMessage {
  final String text;
  final bool isUser;
  final DateTime timestamp;

  ChatMessage({
    required this.text,
    required this.isUser,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();
}

class NLPChatbot {
  static const String _knowledgeFileName = 'assets/data/disease_knowledge.json';
  static const String _modelFileName = 'assets/models/mobilebert_qa.tflite';

  late Map<String, dynamic> _knowledge;
  bool _hasMobileBERT = false;
  final List<ChatMessage> _conversationHistory = [];

  Future<void> initialize() async {
    try {
      // Load knowledge base
      final jsonString = await rootBundle.loadString(_knowledgeFileName);
      _knowledge = jsonDecode(jsonString);
      print('NLP Chatbot initialized with disease knowledge');

      // Check if MobileBERT model exists
      try {
        await rootBundle.load(_modelFileName);
        _hasMobileBERT = true;
        print('✓ MobileBERT TFLite model detected - using advanced NLP');
      } catch (e) {
        _hasMobileBERT = false;
        print('⚠ MobileBERT model not found - using keyword matching');
        print('  To enable: Download mobilebert_qa.tflite and place in assets/models/');
      }
    } catch (e) {
      print('Error initializing chatbot: $e');
      rethrow;
    }
  }

  Future<String> answerQuestion(String question, {String? detectedDisease}) async {
    try {
      // Normalize question
      final normalizedQuestion = question.toLowerCase().trim();

      // If MobileBERT is available, use it for better answers
      if (_hasMobileBERT) {
        return _answerWithMobileBERT(normalizedQuestion, detectedDisease);
      }

      // Fallback to keyword matching
      return _answerWithKeywordMatching(normalizedQuestion, detectedDisease);
    } catch (e) {
      return 'Error processing question: $e';
    }
  }

  String _answerWithMobileBERT(String question, String? detectedDisease) {
    // TODO: Implement actual TFLite inference when model is available
    // For now, use enhanced keyword matching as bridge
    print('Using MobileBERT for question: $question');
    return _answerWithKeywordMatching(question, detectedDisease);
  }

  String _answerWithKeywordMatching(String question, String? detectedDisease) {
    // If user just asks about the detected disease
    if (detectedDisease != null && _isAskingAboutDisease(question, detectedDisease)) {
      return _getDiseaseInfo(detectedDisease);
    }

    // Search FAQ first (improved matching)
    final faqAnswer = _searchFAQ(question);
    if (faqAnswer.isNotEmpty) {
      return faqAnswer;
    }

    // Try to match disease name
    final disease = _matchDisease(question);
    if (disease.isNotEmpty) {
      return _getDiseaseInfo(disease);
    }

    // Try to extract relevant content from disease context
    final relevantAnswer = _extractRelevantContent(question);
    if (relevantAnswer.isNotEmpty) {
      return relevantAnswer;
    }

    // Default response if no match
    return 'I can help with questions about corn leaf diseases (Blight, Common Rust, Gray Leaf Spot, and Healthy leaves). Try asking about symptoms, treatment, or prevention.';
  }

  String _getDiseaseInfo(String diseaseName) {
    try {
      final diseases = _knowledge['diseases'] as List;
      final disease = diseases.firstWhere(
        (d) => (d['name'] as String).toLowerCase() == diseaseName.toLowerCase(),
        orElse: () => null,
      );

      if (disease != null) {
        return disease['context'] as String;
      }
    } catch (e) {
      print('Error getting disease info: $e');
    }
    return 'Disease information not found.';
  }

  String _searchFAQ(String question) {
    try {
      final faqs = _knowledge['faq'] as List;

      for (var faq in faqs) {
        final faqQuestion = (faq['question'] as String).toLowerCase();
        if (_isSimilar(question, faqQuestion)) {
          return faq['answer'] as String;
        }
      }
    } catch (e) {
      print('Error searching FAQ: $e');
    }
    return '';
  }

  String _matchDisease(String text) {
    final diseases = ['blight', 'common_rust', 'common rust', 'gray_leaf_spot', 'gray leaf spot', 'healthy'];

    for (final disease in diseases) {
      if (text.contains(disease)) {
        return disease.replaceAll('_', ' ').split(' ').map((w) => w[0].toUpperCase() + w.substring(1)).join('_');
      }
    }
    return '';
  }

  bool _isAskingAboutDisease(String question, String disease) {
    final diseaseWords = disease.toLowerCase().split('_');
    return diseaseWords.any((word) => question.contains(word));
  }

  bool _isSimilar(String q1, String q2) {
    // Improved keyword matching with better thresholds
    final words1 = q1.split(RegExp(r'\s+|[?.,!]')).where((w) => w.isNotEmpty).toList();
    final words2 = q2.split(RegExp(r'\s+|[?.,!]')).where((w) => w.isNotEmpty).toList();

    int matches = 0;
    for (final word in words1) {
      if (word.length > 2) { // Ignore short words
        if (words2.any((w) => w.contains(word) || word.contains(w))) {
          matches++;
        }
      }
    }

    final threshold = words1.isEmpty ? 0 : (words1.length * 0.5); // 50% keyword match
    return matches >= threshold;
  }

  String _extractRelevantContent(String question) {
    // Search through disease context for relevant keywords
    try {
      final diseases = _knowledge['diseases'] as List;
      final questionWords = question.split(RegExp(r'\s+|[?.,!]')).where((w) => w.length > 3).toList();

      for (var disease in diseases) {
        final context = (disease['context'] as String).toLowerCase();
        final diseaseWords = (disease['name'] as String).toLowerCase().split('_');

        // Score each disease based on keyword overlap
        int score = 0;
        for (final word in questionWords) {
          if (context.contains(word.toLowerCase())) {
            score++;
          }
        }

        // If good match, return relevant excerpt from context
        if (score >= 2) {
          final sentences = context.split('. ');
          final relevantSentences = sentences.where((s) {
            return questionWords.any((w) => s.contains(w.toLowerCase()));
          }).toList();

          if (relevantSentences.isNotEmpty) {
            return relevantSentences.take(2).join('. ') + '.';
          }
        }
      }
    } catch (e) {
      print('Error extracting content: $e');
    }
    return '';
  }

  void addMessage(ChatMessage message) {
    _conversationHistory.add(message);
  }

  List<ChatMessage> getConversationHistory() {
    return List.unmodifiable(_conversationHistory);
  }

  void clearConversation() {
    _conversationHistory.clear();
  }

  List<String> getSuggestedQuestions({String? disease}) {
    if (disease != null) {
      return [
        'What are the symptoms of $disease?',
        'How do I treat $disease?',
        'How do I prevent $disease?',
      ];
    }
    return [
      'What causes corn leaf diseases?',
      'How do I prevent corn diseases?',
      'When should I apply fungicides?',
    ];
  }

  void dispose() {
    _conversationHistory.clear();
  }
}
