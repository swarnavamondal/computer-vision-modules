import cv2
import mediapipe as mp

class poseDetector():
    def __init__(self, mode=False, smooth=True,
                 detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils


    def findPose(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return frame

    def findPosition(self, frame, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            h, w, c = frame.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (100, 200, 0), cv2.FILLED)
        return lmList

def main():
    source = 'posevid1.mp4'
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"Error opening video source: {source}")
        return

    detector = poseDetector()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Video ended or frame not read correctly.")
            break

        # Optional: Resize frame if needed
        # frame = cv2.resize(frame, (640, 480))

        frame = detector.findPose(frame)
        lmList = detector.findPosition(frame, draw=False)

        if len(lmList) !=0:  
            print(lmList[14])
            cv2.circle(frame, (lmList[14][1], lmList[14][2]), 10, (100, 200, 255), cv2.FILLED)

        cv2.imshow("Pose Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
