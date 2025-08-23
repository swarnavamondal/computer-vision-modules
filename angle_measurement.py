import cv2
import mediapipe as mp
import posetrack.posemodule as pmt
import numpy as np

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


#cap = cv2.VideoCapture('sources/posevid2.mp4')
cap = cv2.VideoCapture(0)
detector = pmt.poseDetector()

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()
while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video or cannot read the frame.")
        break
    frame = detector.findPose(frame)
    lmList = detector.findPosition(frame,draw=False)

    if len(lmList) !=0:
        print(lmList[22],lmList[16],lmList[20])  
        cv2.circle(frame, (lmList[12][1], lmList[12][2]), 10, (100, 200, 255), cv2.FILLED)
        cv2.circle(frame, (lmList[14][1], lmList[14][2]), 10, (100, 200, 255), cv2.FILLED)
        cv2.circle(frame, (lmList[16][1], lmList[16][2]), 10, (100, 200, 255), cv2.FILLED)
        leftshoulder = [lmList[12][1], lmList[12][2]]
        leftelbow = [lmList[14][1], lmList[14][2]]
        leftwrist = [lmList[16][1], lmList[16][2]]
        angle = calculate_angle(leftshoulder, leftelbow, leftwrist)
        cv2.putText(frame, f'Angle: {int(angle)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        print("No landmarks detected or incomplete list.")
    cv2.imshow("Pose Detection", frame) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

'''while cap.isOpened():
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
cv2.destroyAllWindows()'''

