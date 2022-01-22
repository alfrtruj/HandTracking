import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)  # video object

mpHands = mp.solutions.hands
hands = mpHands.Hands()  # initialize a mediapipe hand object - only support RGB colors
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:      # detects hand(s)
        for handsLms in results.multi_hand_landmarks: 
            mpDraw.draw_landmarks(img, handsLms) # draw landmarks


    cv2.imshow("Image", img)
    cv2.waitKey(1)
