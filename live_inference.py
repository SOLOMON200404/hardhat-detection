
import cv2
import numpy as np
import onnxruntime as ort
import time

session = ort.InferenceSession('/content/runs/detect/train-2/weights/best.onnx')
input_name = session.get_inputs()[0].name

def preprocess(img):
    t1 = time.time()
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1))[np.newaxis, :]
    t2 = time.time()
    return img_input, (t2-t1)*1000

frame = cv2.imread('/content/dataset/coco128/images/train2017/000000000086.jpg')
inp, pre_ms = preprocess(frame)

t1 = time.time()
outputs = session.run(None, {input_name: inp})
t2 = time.time()
inf_ms = (t2-t1)*1000

print(f"Preprocessing: {pre_ms:.2f}ms")
print(f"Inference: {inf_ms:.2f}ms")
print(f"FPS: {1000/inf_ms:.1f}")
print("Inference script working!")
