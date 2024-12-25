import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class Services {
  static Uint8List preprocessImage(String imagePath) {
    final image = File(imagePath).readAsBytesSync();
    final decodedImage = img.decodeImage(image)!;

    // Resize the image to 160x160 (required by FaceNet)
    final resizedImage = img.copyResize(decodedImage, width: 160, height: 160);

    // Normalize the image to [-1, 1]
    final imageBytes = resizedImage.getBytes();
    final normalizedBytes = Float32List(imageBytes.length ~/ 4);
    for (int i = 0; i < imageBytes.length; i += 4) {
      normalizedBytes[i ~/ 4] = (imageBytes[i] - 127.5) / 127.5;
    }

    return normalizedBytes.buffer.asUint8List();
  }
}

class FaceNetService {
  late Interpreter _interpreter;

  FaceNetService() {
    _loadModel();
  }

  Future<void> _loadModel() async {
    _interpreter = await Interpreter.fromAsset('assets/models/facenet.tflite');
  }

  List<double> getEmbeddings(Uint8List input) {
    final inputShape = _interpreter.getInputTensor(0).shape;
    final outputShape = _interpreter.getOutputTensor(0).shape;

    final inputBuffer = input.buffer.asFloat32List();
    final outputBuffer = List<double>.filled(outputShape[1], 0);

    _interpreter.run(inputBuffer, outputBuffer);
    return outputBuffer;
  }

  void close() {
    _interpreter.close();
  }

  static double calculateSimilarity(List<double> emb1, List<double> emb2) {
    double dotProduct = 0.0;
    double magnitude1 = 0.0;
    double magnitude2 = 0.0;

    for (int i = 0; i < emb1.length; i++) {
      dotProduct += emb1[i] * emb2[i];
      magnitude1 += emb1[i] * emb1[i];
      magnitude2 += emb2[i] * emb2[i];
    }

    magnitude1 = sqrt(magnitude1);
    magnitude2 = sqrt(magnitude2);

    return dotProduct / (magnitude1 * magnitude2);
  }
}
