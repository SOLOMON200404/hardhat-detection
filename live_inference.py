
import cv2
import numpy as np
import onnxruntime as ort
import time

session = ort.InferenceSession("best.onnx")
input_name = session.get_inputs()[0].name
CLASSES = ["hardhat", "no_hardhat"]

def preprocess(img):
    t1 = time.time()
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2, 0, 1))[np.newaxis, :]
    return img_input, (time.time()-t1)*1000

def postprocess(outputs, conf=0.5):
    t1 = time.time()
    preds = outputs[0][0]
    boxes = []
    for pred in preds.T:
        confidence = pred[4:].max()
        if confidence > conf:
            x, y, w, h = pred[:4]
            boxes.append((int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2),
                         confidence, pred[4:].argmax()))
    return boxes, (time.time()-t1)*1000

cap = cv2.VideoCapture(0)
print("Press q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    inp, pre_ms = preprocess(frame)
    t1 = time.time()
    outputs = session.run(None, {input_name: inp})
    inf_ms = (time.time()-t1)*1000
    fps = 1000/inf_ms

    boxes, nms_ms = postprocess(outputs)
    h, w = frame.shape[:2]

    for (x1,y1,x2,y2,conf,cls_id) in boxes:
        x1,y1,x2,y2 = int(x1*w/640),int(y1*h/640),int(x2*w/640),int(y2*h/640)
        color = (0,255,0) if cls_id==0 else (0,0,255)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"{CLASSES[int(cls_id)]} {conf:.2f}",
                   (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.putText(frame, f"Inference: {inf_ms:.1f}ms", (10,70),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.putText(frame, f"Preprocess: {pre_ms:.1f}ms", (10,110),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.putText(frame, f"NMS: {nms_ms:.1f}ms", (10,150),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Hardhat Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
