import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use a custom-trained model for specific products

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=30)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 detection
    results = model(frame)[0]

    # Prepare detections for Deep SORT
    detections = []
    class_names = []
    for r in results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        conf = float(r.conf[0])
        cls = int(r.cls[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
        class_names.append(cls)

    # Run Deep SORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracked objects
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Get class ID from last detection (track.det_class is optional)
        det_class = getattr(track, "det_class", None)
        class_name = model.names.get(det_class, "Object") if det_class is not None else "Object"

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_name} ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Product Detection and Tracking", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
