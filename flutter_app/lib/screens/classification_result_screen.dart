import 'dart:io';
import 'package:flutter/material.dart';
import '../services/image_classifier.dart';
import 'chatbot_screen.dart';

class ClassificationResultScreen extends StatelessWidget {
  final ClassificationResult result;
  final File imageFile;

  const ClassificationResultScreen({
    Key? key,
    required this.result,
    required this.imageFile,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Classification Result'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            // Image preview
            Container(
              width: double.infinity,
              height: 300,
              color: Colors.grey[200],
              child: Image.file(
                imageFile,
                fit: BoxFit.cover,
              ),
            ),
            // Result section
            Padding(
              padding: const EdgeInsets.all(24.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Main result
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.green[50],
                      border: Border.all(color: Colors.green, width: 2),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Detected Disease:',
                          style: TextStyle(
                            fontSize: 14,
                            color: Colors.grey,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          result.label,
                          style: TextStyle(
                            fontSize: 28,
                            fontWeight: FontWeight.bold,
                            color: Colors.green[800],
                          ),
                        ),
                        const SizedBox(height: 16),
                        ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: LinearProgressIndicator(
                            value: result.confidence,
                            minHeight: 8,
                            backgroundColor: Colors.grey[300],
                            valueColor: AlwaysStoppedAnimation<Color>(
                              Colors.green[600]!,
                            ),
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Confidence: ${(result.confidence * 100).toStringAsFixed(2)}%',
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 32),
                  // All predictions
                  const Text(
                    'All Predictions:',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 16),
                  ...result.allConfidences.entries
                      .toList()
                      .asMap()
                      .entries
                      .map((entry) {
                        final index = entry.key;
                        final className = entry.value.key;
                        final confidence = entry.value.value;

                        return _buildPredictionRow(
                          label: className,
                          confidence: confidence,
                          isTop: index == 0,
                        );
                      })
                      .toList(),
                  const SizedBox(height: 32),
                  // Disease info
                  _buildDiseaseInfo(result.label),
                  const SizedBox(height: 32),
                  // Action buttons
                  SizedBox(
                    width: double.infinity,
                    height: 50,
                    child: ElevatedButton.icon(
                      onPressed: () {
                        Navigator.of(context).push(
                          MaterialPageRoute(
                            builder: (_) => ChatbotScreen(detectedDisease: result.label),
                          ),
                        );
                      },
                      icon: const Icon(Icons.chat),
                      label: const Text(
                        'Ask Chatbot',
                        style: TextStyle(fontSize: 16),
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.blue[600],
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  SizedBox(
                    width: double.infinity,
                    height: 50,
                    child: ElevatedButton(
                      onPressed: () {
                        Navigator.of(context).pop();
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.green,
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      child: const Text(
                        'Classify Another Image',
                        style: TextStyle(fontSize: 16),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPredictionRow({
    required String label,
    required double confidence,
    required bool isTop,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        children: [
          Expanded(
            flex: 2,
            child: Text(
              label,
              style: TextStyle(
                fontSize: 14,
                fontWeight: isTop ? FontWeight.bold : FontWeight.normal,
              ),
            ),
          ),
          Expanded(
            flex: 3,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: confidence,
                backgroundColor: Colors.grey[200],
                valueColor: AlwaysStoppedAnimation<Color>(
                  isTop ? Colors.green : Colors.grey,
                ),
              ),
            ),
          ),
          SizedBox(
            width: 70,
            child: Text(
              '${(confidence * 100).toStringAsFixed(1)}%',
              textAlign: TextAlign.right,
              style: TextStyle(
                fontSize: 12,
                fontWeight: isTop ? FontWeight.bold : FontWeight.normal,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDiseaseInfo(String disease) {
    // Handle "Unknown" classification
    if (disease.contains('Unknown')) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.orange[50],
          border: Border.all(color: Colors.orange[300]!, width: 1),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Unable to Classify',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: Colors.orange[800],
              ),
            ),
            const SizedBox(height: 12),
            _buildInfoItem(
              'Reason:',
              'The model is not confident this is a corn leaf. Please ensure:',
            ),
            const SizedBox(height: 8),
            const Padding(
              padding: EdgeInsets.only(left: 8),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('• Image shows a corn plant leaf'),
                  Text('• Lighting is adequate'),
                  Text('• Leaf is clearly visible'),
                  Text('• No other objects in frame'),
                ],
              ),
            ),
          ],
        ),
      );
    }

    final diseaseInfo = {
      'Blight': {
        'description': 'Northern Corn Leaf Blight',
        'symptoms': 'Long, narrow, elliptical lesions on leaves',
        'treatment': 'Use resistant varieties, apply fungicides, remove infected leaves',
      },
      'Common_Rust': {
        'description': 'Common Rust',
        'symptoms': 'Small, brown, circular pustules on leaves',
        'treatment': 'Plant resistant hybrids, apply fungicides if needed',
      },
      'Gray_Leaf_Spot': {
        'description': 'Gray Leaf Spot',
        'symptoms': 'Rectangular gray-brown lesions with dark borders',
        'treatment': 'Rotate crops, use resistant varieties, apply fungicides',
      },
      'Healthy': {
        'description': 'Healthy Corn',
        'symptoms': 'No visible disease symptoms',
        'treatment': 'Continue regular maintenance and monitoring',
      },
    };

    final info = diseaseInfo[disease] ?? diseaseInfo['Healthy']!;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.blue[50],
        border: Border.all(color: Colors.blue[300]!, width: 1),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            info['description']!,
            style: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 12),
          _buildInfoItem('Symptoms:', info['symptoms']!),
          const SizedBox(height: 8),
          _buildInfoItem('Treatment:', info['treatment']!),
        ],
      ),
    );
  }

  Widget _buildInfoItem(String title, String content) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: const TextStyle(
            fontWeight: FontWeight.w600,
            fontSize: 12,
            color: Colors.grey,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          content,
          style: const TextStyle(fontSize: 14),
        ),
      ],
    );
  }
}
