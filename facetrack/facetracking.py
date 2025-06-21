import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('facevid1.mp4')

pTime = 0
while True:
    ret, frame = cap.read()
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow("face detection",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 