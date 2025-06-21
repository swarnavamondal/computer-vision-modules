import cv2
import mediapipe as mp
import posemodule as pmt


cap = cv2.VideoCapture('posevid2.mp4')
#cap = cv2.VideoCapture(0)
detector = pmt.poseDetector()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Frame not captured properly.")
        break

    frame = detector.findPose(frame)
    lmList = detector.findPosition(frame,draw=False)

    if len(lmList) !=0:
        print(lmList[14])  
        cv2.circle(frame, (lmList[14][1], lmList[14][2]), 10, (100, 200, 255), cv2.FILLED)
    else:
        print("No landmarks detected or incomplete list.")
    cv2.imshow("Pose Detection", frame)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

