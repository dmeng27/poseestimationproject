import cv2
import mediapipe as mp
import time

# init
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture("videos/2.mp4")  # Relative path

previousTime = 0
while True:
    success, img = cap.read()  # read() returns a tuple (success, image)
    if not success:
        break  # Exit loop if video ends

    # pose detection
    # img before in bgr we have to convert to rgb
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    # check landmarks
    # print(results.pose_landmarks)
    # list init
    id_storage = []
    if results.pose_landmarks:
        # draw landmarks
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        # iterate through all landmarks

        for id, lm in enumerate(results.pose_landmarks.landmark):
            # height, width, channel
            h, w, c = img.shape
            # identify landmarks
            # print(id, lm)
            # Store id and landmark coordinates
            id_storage.append((id, lm.x, lm.y, lm.z, lm.visibility))

            # pixel values
            cx, cy = int(lm.x * w), int(lm.y * h)
            # circle check
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        #write to file after processing all landmarks in this frame
        with open('items.txt', 'w') as file:
            for item in id_storage:
                #convert each tuple to a string with comma-separated values
                line = ','.join(str(x) for x in item)
                file.write(line + '\n')

    # fps counter
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    # video playback
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()