import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)  # video object

mpHands = mp.solutions.hands
hands = mpHands.Hands()  # initialize a mediapipe hand object - only support RGB colors
mpDraw = mp.solutions.drawing_utils

cTime = 0
pTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:      # detects hand(s)
        for handsLms in results.multi_hand_landmarks: 
            for id, lm in enumerate(handsLms.landmark):
                # print(id, lm)  # show the landmark position (in decimasl) and its id
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h) # position in pixels (multiplying width and height)
                print(id, cx, cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255),cv2.FILLED) # make a circle in the landmark position


            mpDraw.draw_landmarks(img, handsLms, mpHands.HAND_CONNECTIONS) # draw landmarks and connections

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255, 0, 255),3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
