# Hardhat Detection - YOLOv8

Object detection model to detect hardhats in construction sites.

## Performance Benchmark Table

| Metric | FP32 Base Model | FP16 Quantized (ONNX) |
|--------|----------------|----------------------|
| Model Size | 6.2 MB | 6.2 MB |
| mAP50 | 0.715 | ~0.700 |
| Inference Speed | 92.02ms | 115.46ms |
| FPS | 10.9 | 8.7 |

## Model Weights
- FP32 weights: https://drive.google.com/drive/folders/1Q13GJkK4wTjYjfZZG-4isNE__BIgxB8Q?usp=sharing
- FP16 ONNX weights: https://drive.google.com/drive/folders/1Q13GJkK4wTjYjfZZG-4isNE__BIgxB8Q?usp=sharing

## How to Run
```bash
python live_inference.py
```

## Dataset
COCO128 dataset used for training (128 images, 80 classes).

## Tech Stack
- YOLOv8n
- ONNX Runtime
- OpenCV
- Python
