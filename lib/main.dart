import 'package:face_match_app/services.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

void main() {
  runApp(const FaceMatchApp());
}

class FaceMatchApp extends StatefulWidget {
  const FaceMatchApp({super.key});

  @override
  _FaceMatchAppState createState() => _FaceMatchAppState();
}

class _FaceMatchAppState extends State<FaceMatchApp> {
  final FaceNetService _faceNetService = FaceNetService();
  String? _imagePath1;
  String? _imagePath2;
  double? _similarity;

  Future<void> _pickImage(int imageNumber) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.camera);

    if (pickedFile != null) {
      setState(() {
        if (imageNumber == 1) {
          _imagePath1 = pickedFile.path;
        } else {
          _imagePath2 = pickedFile.path;
        }
      });
    }
  }

  Future<void> _compareFaces() async {
    if (_imagePath1 != null && _imagePath2 != null) {
      final input1 = Services.preprocessImage(_imagePath1!);
      final input2 = Services.preprocessImage(_imagePath2!);

      final emb1 = _faceNetService.getEmbeddings(input1);
      final emb2 = _faceNetService.getEmbeddings(input2);

      final similarity = FaceNetService.calculateSimilarity(emb1, emb2);

      setState(() {
        _similarity = similarity;
      });
    }
  }

  @override
  void dispose() {
    _faceNetService.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('Face Match')),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton(
                onPressed: () => _pickImage(1),
                child: const Text('Select Image 1'),
              ),
              ElevatedButton(
                onPressed: () => _pickImage(2),
                child: const Text('Select Image 2'),
              ),
              ElevatedButton(
                onPressed: _compareFaces,
                child: const Text('Compare Faces'),
              ),
              if (_similarity != null)
                Text(
                  'Similarity: ${(_similarity! * 100).toStringAsFixed(2)}%',
                  style: const TextStyle(fontSize: 20),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
