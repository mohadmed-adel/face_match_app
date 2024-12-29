import 'dart:io';
import 'dart:typed_data';

import 'package:image/image.dart' as img;

class Services {
  static Float32List preprocessImage(String imagePath) {
    final image = File(imagePath).readAsBytesSync();
    final decodedImage = img.decodeImage(image)!;

    // Resize the image to 160x160
    final resizedImage = img.copyResize(decodedImage, width: 160, height: 160);

    // Normalize the image to [-1, 1]
    final normalizedBytes = Float32List(160 * 160 * 3);
    int index = 0;
    for (int y = 0; y < 160; y++) {
      for (int x = 0; x < 160; x++) {
        final pixel = resizedImage.getPixel(x, y);

        // Extract the RGB components
        final r = pixel.r;
        final g = pixel.g;
        final b = pixel.b;

        // Normalize each component to [-1, 1]
        normalizedBytes[index++] = (r - 127.5) / 127.5;
        normalizedBytes[index++] = (g - 127.5) / 127.5;
        normalizedBytes[index++] = (b - 127.5) / 127.5;
      }
    }

    // Add batch dimension: [1, 160, 160, 3]
    final normalizedBytesWithBatch = Float32List(1 * 160 * 160 * 3);
    normalizedBytesWithBatch.setAll(0, normalizedBytes);
    return normalizedBytesWithBatch;
  }
}
