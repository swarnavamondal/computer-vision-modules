import cv2
import mediapipe as mp
import time
ptime = 0

cap = cv2.VideoCapture('sources/facevid1.mp4')

mpFaceDetection = mp.solutions.face_detection
mpDraw  = mp.solutions.drawing_utils
faceDetection  = mpFaceDetection.FaceDetection()

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    ret, frame = cap.read()

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    

    if results.detections:
        for id, dectection in enumerate(results.detections):
            bboxC = dectection.location_data.relative_bounding_box
            h , w, c = frame.shape
            bbox = int(bboxC.xmin *w) , int(bboxC.ymin *h),\
                   int(bboxC.width *w) , int(bboxC.height *h)
            cv2.rectangle(frame,bbox, (255,0,255), 2)
            cv2.putText(frame, f'accuracy: {int(dectection.score[0]*100)}%', (bbox[0],bbox[1]),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if not ret:
        print("End of video or cannot read the frame.")
        break
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
