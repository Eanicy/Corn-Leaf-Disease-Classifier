import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/image_classifier.dart';
import 'classification_result_screen.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({Key? key}) : super(key: key);

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  final ImagePicker _imagePicker = ImagePicker();
  bool _isClassifying = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('CornDoctor'),
        centerTitle: true,
        elevation: 0,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.eco,
                size: 100,
                color: Colors.green[700],
              ),
              const SizedBox(height: 32),
              const Text(
                'Diagnose Corn Leaf Health',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 48),
              _buildActionButton(
                icon: Icons.camera_alt,
                label: 'Take Photo',
                onPressed: _isClassifying ? null : _classifyFromCamera,
              ),
              const SizedBox(height: 16),
              _buildActionButton(
                icon: Icons.image,
                label: 'Choose from Gallery',
                onPressed: _isClassifying ? null : _classifyFromGallery,
              ),
              const SizedBox(height: 48),
              if (_isClassifying)
                const CircularProgressIndicator()
              else
                const Text(
                  'Tap a button to classify a corn leaf image',
                  style: TextStyle(
                    color: Colors.grey,
                    fontSize: 14,
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildActionButton({
    required IconData icon,
    required String label,
    required VoidCallback? onPressed,
  }) {
    return SizedBox(
      width: double.infinity,
      height: 60,
      child: ElevatedButton.icon(
        onPressed: onPressed,
        icon: Icon(icon, size: 24),
        label: Text(
          label,
          style: const TextStyle(fontSize: 18),
        ),
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.green,
          foregroundColor: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          disabledBackgroundColor: Colors.grey[300],
        ),
      ),
    );
  }

  Future<void> _classifyFromCamera() async {
    final pickedFile = await _imagePicker.pickImage(
      source: ImageSource.camera,
      imageQuality: 90,
    );

    if (pickedFile != null) {
      await _classifyImage(File(pickedFile.path));
    }
  }

  Future<void> _classifyFromGallery() async {
    final pickedFile = await _imagePicker.pickImage(
      source: ImageSource.gallery,
      imageQuality: 90,
    );

    if (pickedFile != null) {
      await _classifyImage(File(pickedFile.path));
    }
  }

  Future<void> _classifyImage(File imageFile) async {
    setState(() {
      _isClassifying = true;
    });

    try {
      final classifier = ImageClassifier();
      await classifier.initialize();
      final result = await classifier.classify(imageFile);

      if (mounted) {
        Navigator.of(context).push(
          MaterialPageRoute(
            builder: (_) => ClassificationResultScreen(
              result: result,
              imageFile: imageFile,
            ),
          ),
        );
      }

      classifier.dispose();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Classification failed: $e')),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isClassifying = false;
        });
      }
    }
  }
}
