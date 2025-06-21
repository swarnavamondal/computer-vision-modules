import cv2
import mediapipe as mp
import time

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.min_detection_confidence = min_detection_confidence
        self.mpFaceDetection = mp.solutions.face_detection 
        self.faceDetection = self.mpFaceDetection.FaceDetection(min_detection_confidence)
        self.mpDraw = mp.solutions.drawing_utils 

    def findFaces(self,frame, draw = True):

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxes =[]

        if self.results.detections:
            for id, dectection in enumerate(self.results.detections):
                bboxC = dectection.location_data.relative_bounding_box
                h , w, c = frame.shape
                bbox = int(bboxC.xmin *w) , int(bboxC.ymin *h),\
                   int(bboxC.width *w) , int(bboxC.height *h)
                bboxes.append([id, bbox, dectection.score])
                if draw:
                    frame = self.fancyDraw(frame, bbox)
                    cv2.putText(frame, f'accuracy: {int(dectection.score[0]*100)}%', (bbox[0],bbox[1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame, bboxes

    

    def fancyDraw(self, img, bbox, l=30, t=6, rt= 1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        # cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # Top Left  x,y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # Top Right  x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)
        # Bottom Left  x,y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right  x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img


def main():
    ptime = 0
    # cap = cv2.VideoCapture('sources/facevid1.mp4')
    cap = cv2.VideoCapture(0)  

    detector = FaceDetector()

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
if __name__ == "__main__":
    main()