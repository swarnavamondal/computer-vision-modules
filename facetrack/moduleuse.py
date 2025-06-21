import cv2
import mediapipe as mp
import time
import facetrackingmodule as ftm

ptime = 0
# cap = cv2.VideoCapture('sources/facevid1.mp4')
cap = cv2.VideoCapture(0)  

detector = ftm.FaceDetector()

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()
while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video or cannot read the frame.")
        break
    frame, bboxes = detector.findFaces(frame)
    print(bboxes)
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()