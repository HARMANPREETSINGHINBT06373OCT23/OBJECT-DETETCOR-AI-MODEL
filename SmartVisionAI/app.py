from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2

app = Flask(__name__)
model = YOLO("yolov8n.pt")

def detect_yolo():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            conf = box.conf[0]

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (204, 0, 255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("YOLOv8 Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start_detection():
    detect_yolo()
    return "Detection ended"

if __name__ == "__main__":
    app.run(debug=True)
