import cv2
from ultralytics import YOLO

# Load medium model (better accuracy than nano/small)
model = YOLO("yolov8m.pt")

# Load image
img = cv2.imread("traffic.jpeg")

# Run prediction on CPU
results = model.predict(
    img,
    imgsz=960,      # lower than 1280 for CPU speed
    conf=0.35,      # confidence threshold
    iou=0.5,        # NMS threshold
    device="cpu"    # IMPORTANT for your setup
)

# Vehicle classes in COCO dataset
vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

vehicle_count = 0

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls in vehicle_classes:
            vehicle_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{vehicle_classes[cls]} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

# Display total count
cv2.putText(img, f"Total Vehicles: {vehicle_count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0,0,255),
            3)

print("Total Vehicles Detected:", vehicle_count)

cv2.imshow("Vehicle Detection (CPU)", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
