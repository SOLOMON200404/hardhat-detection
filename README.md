# Hardhat Detection - YOLOv8n

Real-world industrial safety system to detect hardhat compliance on construction sites.

## Problem Statement
Detecting missing PPE (hardhats) on construction sites using computer vision to prevent workplace accidents.

## Dataset
- **Source:** Roboflow Universe - Hardhat Dataset
- **Total Images:** 2,146 (exceeds 500 minimum requirement)
- **Classes:** hardhat, no_hardhat
- **Format:** YOLOv8

## Performance Benchmark Table

| Metric | FP32 Base Model | FP16 Quantized (ONNX) |
|--------|----------------|----------------------|
| Model Size | 6.2 MB | 6.2 MB |
| mAP50 | 0.974 | ~0.965 |
| mAP50-95 | 0.784 | ~0.770 |
| Inference Speed | 25.89ms | 93.11ms |
| FPS | 38.6 | 10.7 |

## Model Weights
- FP32 + FP16 weights: https://drive.google.com/drive/folders/1uFeP-fS8MLRO-uX1_mrX4r3eOQP2un8P?usp=sharing

## How to Run
```bash
pip install onnxruntime opencv-python
python live_inference.py
```

## Tech Stack
- YOLOv8n
- ONNX Runtime (FP16)
- OpenCV
- Python

## Results
- Hardhat detection: 96.8% mAP50
- No-hardhat detection: 98.0% mAP50
