import cv2                      # OpenCV: webcam + drawing
from ultralytics import YOLO    # YOLOv8

model = YOLO("yolov8n.pt")      # n = nano, smallest model; downloads ~6 MB on first run
cam = cv2.VideoCapture(0)       # 0 = the laptop's built-in webcam

while True:                     # (1) loop forever; we break out ourselves
    ok, frame = cam.read()      # ok = did we get a frame, frame = the image
    if not ok:                # (2)
        break
    result = model(frame, verbose=False)[0]   # run YOLO on this one frame

    for box in result.boxes:            # (3) every detection YOLO found
        x1, y1, x2, y2 = map(int, box.xyxy[0])   # the box's corners, as whole pixels
        label = model.names[int(box.cls[0])]     # class number -> name, e.g. 0 -> "person"
        conf = float(box.conf[0])
        if conf < 0.55:
            continue          # skip this box, move on to the next one                # confidence, 0 to 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:0.4f}", (x1, y1 - 8),   # (4) 4 decimals
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("LOD", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):        # press q to quit
        break

cam.release()
cv2.destroyAllWindows()
