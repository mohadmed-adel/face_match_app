import 'dart:developer' as dev;
import 'dart:math';
import 'dart:typed_data';

import 'package:tflite_flutter/tflite_flutter.dart';

class FaceNetService {
  late Interpreter _interpreter;

  FaceNetService() {
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      dev.log("Loading model...");
      _interpreter =
          await Interpreter.fromAsset('assets/models/facenet.tflite');
      dev.log("Model loaded successfully.");
    } catch (e) {
      dev.log("Error loading model: $e");
    }
  }

  List<double> getEmbeddings(Float32List input) {
    try {
      final inputShape = _interpreter.getInputTensor(0).shape;
      final outputShape = _interpreter.getOutputTensor(0).shape;

      dev.log('Input shape: $inputShape');
      dev.log('Output shape: $outputShape');

      // Ensure the input size matches the expected shape (1, 160, 160, 3)
      if (inputShape.length != 4 ||
          inputShape[1] != 160 ||
          inputShape[2] != 160 ||
          inputShape[3] != 3) {
        throw Exception(
            "Input shape does not match expected size (1x160x160x3). Actual shape: $inputShape");
      }

      // Reshape the input buffer into [1, 160, 160, 3]
      final reshapedInput = List.generate(
        1,
        (_) => List.generate(
          160,
          (y) => List.generate(
            160,
            (x) => List.generate(
              3,
              (c) => input[(y * 160 + x) * 3 + c].toDouble(),
            ),
          ),
        ),
      );

      // Create an output buffer matching the model's output shape
      final outputBuffer = List.generate(
          outputShape[0], (_) => List<double>.filled(outputShape[1], 0.0));

      dev.log("Running inference...");
      _interpreter.run(reshapedInput, outputBuffer);
      dev.log("Inference completed.");

      // Flatten the output buffer to a single list of embeddings
      dev.log(
          "Output buffer shape before inference: ${outputBuffer.length} x ${outputBuffer[0].length}");
      return outputBuffer[0];
    } catch (e) {
      dev.log("Error during inference: $e");
      return [];
    }
  }

  void close() {
    _interpreter.close();
    dev.log("Interpreter closed.");
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
