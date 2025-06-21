import cv2
import mediapipe as mp

cap = cv2.VideoCapture('posevid2.mp4')
#cap = cv2.VideoCapture(0)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
while True:
    ret, frame = cap.read()
    imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark ):
            h, w, c = frame.shape
            #print(id,lm)
            cx, cy = int(lm.x*w) , int(lm.y*h)
            cv2.circle(frame,(cx,cy),5,(100,200,0),cv2.FILLED)

    cv2.imshow("frame",frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
