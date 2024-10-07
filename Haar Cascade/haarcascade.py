import cv2

face_classifier = cv2.CascadeClassifier('HC-classifier.xml')

def face_detector(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
    if len(faces) == 0:
        return img, 0 
    else:
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        return img, len(faces) 

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Failed to open camera.")
    exit()

default_screen_res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)

    frame_with_faces, count = face_detector(frame, scaleFactor=1.10)

    cv2.putText(frame_with_faces, f'Number of Faces: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    screen_res = default_screen_res  

    aspect_ratio = screen_res[0] / frame_with_faces.shape[1]
    new_height = int(frame_with_faces.shape[0] * aspect_ratio)
    frame_with_faces_resized = cv2.resize(frame_with_faces, (screen_res[0], new_height))

    cv2.imshow('Our Face Extractor', frame_with_faces_resized)

    if cv2.waitKey(1) & 0xFF == ord('2'): 
        break

cap.release()
cv2.destroyAllWindows()