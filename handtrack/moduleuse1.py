import cv2
import mediapipe as mp
import time
import handtrackingmodule as htm

pTime = 0
cap = cv2.VideoCapture(0)  
detector = htm.handDetector()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame from camera")
        break

    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if lmList:
        print(lmList[0])

    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime - pTime > 0 else 0
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




