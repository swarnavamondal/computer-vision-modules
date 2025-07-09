import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('sources/facemesh2.mp4')  
ptime=0

mpDraw  = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=3)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    ret , frame = cap.read()
    if not ret:
        print("End of video or cannot read the frame.")
        break
    
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_TESSELATION,
                                  drawSpec,drawSpec)
            for id,lm in enumerate(faceLms.landmark):
                #print(lm)
                ih, iw, ic = frame.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                print(id,x,y)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Face Mesh',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()