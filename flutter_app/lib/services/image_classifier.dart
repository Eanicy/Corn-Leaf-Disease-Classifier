import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:convert';
import 'package:flutter/services.dart';
import 'dart:math' as math;

class ClassificationResult {
  final String label;
  final double confidence;
  final Map<String, double> allConfidences;

  ClassificationResult({
    required this.label,
    required this.confidence,
    required this.allConfidences,
  });
}

class ImageClassifier {
  static const String _modelFileName = 'assets/models/corn_disease_classifier.tflite';
  static const String _labelsFileName = 'assets/models/labels.json';
  static const int _inputSize = 224;

  late Interpreter _interpreter;
  List<String> _labels = [];
  late List<int> _inputShape;
  late List<int> _outputShape;

  Future<void> initialize() async {
    try {
      // Load model
      final options = InterpreterOptions()..threads = 4;
      _interpreter = await Interpreter.fromAsset(
        _modelFileName,
        options: options,
      );

      _inputShape = _interpreter.getInputTensor(0).shape;
      _outputShape = _interpreter.getOutputTensor(0).shape;

      // Load labels
      final labelsData = await rootBundle.loadString(_labelsFileName);
      final json = jsonDecode(labelsData);
      _labels = List<String>.from(json['labels']);

      print('Model initialized successfully');
      print('Input shape: $_inputShape');
      print('Output shape: $_outputShape');
      print('Labels: $_labels');
    } catch (e) {
      print('Error initializing model: $e');
      rethrow;
    }
  }

  Future<ClassificationResult> classify(File imageFile) async {
    try {
      // Read and decode image
      final bytes = await imageFile.readAsBytes();
      img.Image? image = img.decodeImage(bytes);

      if (image == null) {
        throw Exception('Failed to decode image');
      }

      // Resize to model input size
      image = img.copyResize(image, width: _inputSize, height: _inputSize);

      // Normalize and prepare input
      final input = _prepareInput(image);

      // Run inference
      final output = List.filled(1 * _outputShape[1], 0.0).reshape([1, _outputShape[1]]);
      _interpreter.run(input, output);

      // Process output
      final confidences = _processOutput(output);

      // Find max confidence class
      int maxIndex = 0;
      double maxConfidence = confidences[0];

      for (int i = 1; i < confidences.length; i++) {
        if (confidences[i] > maxConfidence) {
          maxConfidence = confidences[i];
          maxIndex = i;
        }
      }

      // Confidence threshold: reject if confidence is too low
      // This prevents the model from guessing on non-corn-leaf images
      const double confidenceThreshold = 0.60; // 60% confidence threshold

      if (maxConfidence < confidenceThreshold) {
        return ClassificationResult(
          label: 'Unknown - Please capture a corn leaf',
          confidence: 0.0,
          allConfidences: Map.fromEntries(
            _labels.asMap().entries.map(
              (e) => MapEntry(e.value, confidences[e.key]),
            ),
          ),
        );
      }

      return ClassificationResult(
        label: _labels[maxIndex],
        confidence: maxConfidence,
        allConfidences: Map.fromEntries(
          _labels.asMap().entries.map(
            (e) => MapEntry(e.value, confidences[e.key]),
          ),
        ),
      );
    } catch (e) {
      print('Error during classification: $e');
      rethrow;
    }
  }

  List<List<List<List<double>>>> _prepareInput(img.Image image) {
    final List<List<List<List<double>>>> input = List.generate(
      1,
      (i) => List.generate(
        _inputSize,
        (j) => List.generate(
          _inputSize,
          (k) => List.filled(3, 0.0),
        ),
      ),
    );

    for (int y = 0; y < _inputSize; y++) {
      for (int x = 0; x < _inputSize; x++) {
        final pixel = image.getPixelSafe(x, y);
        final r = pixel.r / 255.0;
        final g = pixel.g / 255.0;
        final b = pixel.b / 255.0;

        // ImageNet normalization
        input[0][y][x][0] = (r - 0.485) / 0.229;
        input[0][y][x][1] = (g - 0.456) / 0.224;
        input[0][y][x][2] = (b - 0.406) / 0.225;
      }
    }

    return input;
  }

  List<double> _processOutput(List<dynamic> output) {
    List<double> probabilities = [];

    if (output is List && output.isNotEmpty) {
      final outputList = output[0];
      if (outputList is List) {
        probabilities = List<double>.from(outputList.map((x) => x is double ? x : (x as num).toDouble()));
      }
    }

    // Apply softmax if not already done
    return _softmax(probabilities);
  }

  List<double> _softmax(List<double> logits) {
    // Temperature scaling for calibration (lower = sharper/more confident)
    const double temperature = 0.6;

    final maxLogit = logits.reduce((a, b) => a > b ? a : b);
    final scaledLogits = logits.map((x) => (x - maxLogit) / temperature).toList();
    final expValues = scaledLogits.map((x) => math.exp(x)).toList();
    final sumExp = expValues.fold(0.0, (a, b) => a + b);
    return expValues.map<double>((x) => x / sumExp).toList();
  }

  void dispose() {
    _interpreter.close();
  }
}
