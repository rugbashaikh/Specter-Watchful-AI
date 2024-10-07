from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('best200.pt')

# Open the webcam
cap = cv2.VideoCapture(1)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Run the YOLO model on the frame
    results = model(frame)

    # Get the number of detections
    num_detections = 0
    for r in results:
        num_detections += len(r.boxes)

    # Add the count to the image
    cv2.putText(frame, f"Total people: {num_detections}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # # Display the frame
    # cv2.imshow("Object Detection", frame)


    # Iterate over the detection results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = int(box.cls[0])
            confidence = box.conf[0].item()

            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{model.names[class_id]} {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow("Object Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()