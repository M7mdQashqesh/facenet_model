# FaceNet Model

A FaceNet facial recognition model with conversion to TensorFlow Lite format for mobile and edge device deployment.

## Project Overview

This project contains a pre-trained FaceNet model and utilities for:
- Loading and inspecting the frozen TensorFlow graph
- Converting the model to TensorFlow Lite (.tflite) format for lightweight inference
- Examining model tensors and operations

## Files

- **20180402-114759.pb** - Pre-trained FaceNet frozen graph model (original TensorFlow format)
- **facenet.tflite** - Converted TensorFlow Lite model for mobile/edge deployment
- **convert_to_tflite.py** - Script to convert the frozen graph to TensorFlow Lite format
- **list_tensors.py** - Utility script to inspect all operations and tensors in the model

## Model Details

### Input Tensors
- **input:0** - Input image tensor (shape: [1, 160, 160, 3])
- **phase_train:0** - Training phase flag (boolean)
- **batch_size:0** - Batch size (int32)

### Output Tensors
- **embeddings:0** - Face embedding vector (output representation)

The model expects:
- Image input: 160×160 RGB images normalized to float32
- Output: 128-dimensional face embedding (L2 normalized)

## Usage

### Prerequisites

```bash
pip install tensorflow
```

### Convert Model to TensorFlow Lite

Run the conversion script:

```bash
python convert_to_tflite.py
```

This generates an optimized `facenet.tflite` file suitable for:
- Mobile devices (iOS/Android)
- Edge devices (Raspberry Pi, Jetson Nano)
- Embedded systems
- IoT applications

### Inspect Model Structure

To view all operations in the model:

```bash
python list_tensors.py
```

This outputs all operation names defined in the frozen graph, useful for understanding the model architecture.

## Model Information

- **Architecture:** FaceNet (VGGFaceNet-based)
- **Input Size:** 160×160 pixels
- **Embedding Dimension:** 128
- **Format:** TensorFlow protobuf (.pb) / TensorFlow Lite (.tflite)
- **Use Case:** Face recognition, face verification, facial similarity comparison

## Applications

- Face recognition and identification
- Face verification and authentication
- Facial similarity search
- Clustering faces by identity
- De-duplication of face images

## Notes

- The model uses inference mode during conversion (phase_train=False)
- Optimization is applied during TensorFlow Lite conversion
- Input images should be preprocessed (aligned, normalized to [-1, 1] or [0, 1])
- The output embeddings can be compared using L2 distance or cosine similarity

## License

Ensure compliance with FaceNet model licensing and usage terms.
