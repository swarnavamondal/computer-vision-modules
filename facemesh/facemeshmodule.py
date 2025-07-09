import cv2
import mediapipe as mp
import time

class FaceMeshDetector:
    def __init__(self, staticMode = False, max_faces=3,min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.staticMode = staticMode
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_faces = max_faces
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode,max_num_faces=self.max_faces,
                                                min_detection_confidence=self.min_detection_confidence,
                                                min_tracking_confidence=self.min_tracking_confidence)

        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, frame,Draw=True):
        self.imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            
            for faceLms in self.results.multi_face_landmarks:
                if Draw:
                    self.mpDraw.draw_landmarks(frame, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION,
                                           self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = frame.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    #cv2.putText(frame, f'FPS: {int(id)}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                    #print(id, x, y)
                    face.append([x,y])
                faces.append(face)   
        return frame, faces



def main():
    cap = cv2.VideoCapture('sources/facemesh1.mp4')  
    #cap = cv2.VideoCapture(0)  # Use webcam for real-time detection
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()
    ptime=0
    detector = FaceMeshDetector()
    while True:
        ret , frame = cap.read()
        if not ret:
            print("End of video or cannot read the frame.")
            break
        frame, faces = detector.findFaceMesh(frame)
        if faces:
            print(faces[0])
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('Face Mesh',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()