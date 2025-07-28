import cv2
import mediapipe as mp
import facemeshmodule as fmm
import time

cap = cv2.VideoCapture('sources/facemesh1.mp4')  
#cap = cv2.VideoCapture(0)  # Use webcam for real-time detection
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()
ptime=0
detector = fmm.FaceMeshDetector()
while True:
    ret , frame = cap.read()
    if not ret:
        print("End of video or cannot read the frame.")
        break
    frame, faces = detector.findFaceMesh(frame)
    if faces:
        print(faces[0])
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Face Mesh',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()