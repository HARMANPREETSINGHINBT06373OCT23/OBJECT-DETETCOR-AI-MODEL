from flask import Flask, render_template, request
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/start', methods=['POST'])
def start_detection():
    detect_objects()
    return "Detection complete. Close camera window and return."

def detect_objects():
    net = cv2.dnn.readNetFromCaffe(
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt",
        "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel"
    )

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
               "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)

        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                object_roi = frame[startY:endY, startX:endX]
                color_name = get_dominant_color(object_roi)

                width = endX - startX
                height = endY - startY

                text = f"{label} | {color_name} | {width}x{height}px"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, text, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_dominant_color(image):
    if image.size == 0:
        return "Unknown"
    image = cv2.resize(image, (50, 50))
    pixels = image.reshape(-1, 3)
    avg_color = np.mean(pixels, axis=0)

    # Simple color mapping
    r, g, b = avg_color
    if r > 150 and g < 100 and b < 100:
        return "Red"
    elif g > 150 and r < 100 and b < 100:
        return "Green"
    elif b > 150 and r < 100 and g < 100:
        return "Blue"
    elif r > 200 and g > 200 and b < 100:
        return "Yellow"
    elif r > 200 and g > 200 and b > 200:
        return "White"
    elif r < 50 and g < 50 and b < 50:
        return "Black"
    else:
        return "Mixed"

if __name__ == '__main__':
    app.run(debug=True)
