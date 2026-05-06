import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'services/image_classifier.dart';
import 'screens/camera_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final classifier = ImageClassifier();
  await classifier.initialize();

  runApp(
    ChangeNotifierProvider(
      create: (_) => ImageClassifierProvider(classifier),
      child: const MyApp(),
    ),
  );
}

class ImageClassifierProvider extends ChangeNotifier {
  final ImageClassifier classifier;

  ImageClassifierProvider(this.classifier);

  @override
  void dispose() {
    classifier.dispose();
    super.dispose();
  }
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'CornDoctor',
      theme: ThemeData(
        primarySwatch: Colors.green,
        useMaterial3: true,
      ),
      home: const CameraScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}
