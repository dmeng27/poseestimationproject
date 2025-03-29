import cv2
import mediapipe as mp
import time

from PoseEstimationMain import id_storage


class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon  # Fixed parameter name (was detectCon)
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,  # "upBody" (0, 1, or 2)
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img


    def findPosition(self, img, draw=True):
        id_storage = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                id_storage.append([id, lm.x, lm.y])
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return id_storage
def main():
    cap = cv2.VideoCapture("videos/1.mp4")
    previousTime = 0
    detector = poseDetector(
        mode=False,
        upBody=1,  # model_complexity (0, 1, or 2)
        smooth=True,
        detectionCon=0.5,  # Note: this should be between 0.0 and 1.0
        trackCon=0.5  # Note: this should be between 0.0 and 1.0
    )
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        id_storage = detector.findPosition(img)
        print(id_storage)
        if not success:
            break

        with open('items_1.txt', 'w') as file:
            for item in id_storage:
                line = ','.join(str(x) for x in item)
                file.write(line + '\n')

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 3)
        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
