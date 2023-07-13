import signal
import cv2
import numpy as np
import time
import os
import handTrackingModule as htm
import math
import pyautogui

##########################
wCam, hCam = 1080, 720
frameR = 200  # Frame Reduction
smoothening = 8
#########################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

prevThumb, prstThumb = 0, 0
prev = 0
prevLength = 0

detector = htm.handDetector(maxHands=1)
wScr, hScr = pyautogui.size()
#print(wScr, hScr)

# Define the font and size of the buttons text
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1

# Define button positions and sizes
run_button_x = 200
run_button_y = 70
run_button_width = 350
run_button_height = 100

stop_button_x = 600
stop_button_y = 70
stop_button_width = 150
stop_button_height = 100

# Define colors
button_color = (255, 0, 0)  # Blue
text_color = (255, 255, 255)  # White

def run_another_file():
    cap.release()
    cv2.destroyAllWindows()
    # Specify the path to the Python file you want to run
    another_file_path = "FAW.py"
    os.system("python " + another_file_path)
    os.kill(os.getpid(), signal.SIGTERM)

while True:
    # Find hand Landmarks
    success, img = cap.read()
    
    # Flip the axis horizontally
    img = cv2.flip(img, 1)
    
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
    
    # Check if the run button is clicked
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        cx, cy = 0, 0
        # Check if the index and middle finger are close enough (within 40 pixels)
        if abs(x1 - x2) < 40 and abs(y1 - y2) < 40:
            # Calculate the center of index and middle coordinates 
            cx = (x1 + x2) // 2 
            cy = (y1 + y2) // 2 
        if run_button_x < cx < run_button_x + run_button_width and run_button_y < cy < run_button_y + run_button_height:
            print('Opening Finger Air Writing ...')
            run_another_file()

    # Check if the stop button is clicked
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        cx, cy = 0, 0
        # Check if the index and middle finger are close enough (within 40 pixels)
        if abs(x1 - x2) < 40 and abs(y1 - y2) < 40:
            # Calculate the center of index and middle coordinates 
            cx = (x1 + x2) // 2 
            cy = (y1 + y2) // 2 
        length, img, lineInfo = detector.findDistance(8, 12, img)
        if stop_button_x < cx < stop_button_x + stop_button_width and stop_button_y < cy < stop_button_y + stop_button_height:
            print('Closing Virtual Mouse ...')
            break
    
    # Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x0, y0 = lmList[4][1:]
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        x3, y3 = lmList[16][1:]
        x4, y4 = lmList[20][1:]

        # Check which fingers are up
        fingers = detector.fingersUp()

       # if index, middle, and ring fingers are up then volume is increased
        if fingers == [0, 1, 1, 1, 0]:
            pyautogui.press('volumeup')
            cv2.circle(img, (x1, y1), 15, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (255, 255, 255), cv2.FILLED)
            cv2.putText(img, "Volume Up", (500, 55), font, fontScale, (255, 0, 0), 3, cv2.LINE_AA)
            
        # if index, middle, ring, little fingers are up then volume is decreased
        elif fingers == [0, 1, 1, 1, 1]:
            pyautogui.press('volumedown')
            cv2.circle(img, (x1, y1), 15, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (x4, y4), 15, (255, 255, 255), cv2.FILLED)
            cv2.putText(img, "Volume Down", (500, 55), font, fontScale, (255, 0, 0), 3, cv2.LINE_AA)
        
        # Only Index Finger: Moving Mode
        elif fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 0:
            # Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            # Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Move Mouse
            pyautogui.moveTo(clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.putText(img, "Moving Mouse", (460, 55), font, fontScale, (255, 0, 0), 3, cv2.LINE_AA) # white text 
            plocX, plocY = clocX, clocY

        # Both Index and middle fingers are up: Clicking Mode
        elif fingers[1] == 1 and fingers[2] == 1:
            # Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # Click mouse if distance short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                cv2.putText(img, "Left Click", (500, 55), font, fontScale, (255, 0, 0), 3, cv2.LINE_AA) # white text 
                pyautogui.click()

        # Thumb and index finger: Right Click
        elif fingers[0] == 1 and fingers[1] == 1:
            # Find distance between fingers
            length, img, lineInfo = detector.findDistance(4, 8, img)
            if length < 40:
                cv2.putText(img, "Right Click", (500, 55), font, fontScale, (255, 0, 0), 3, cv2.LINE_AA) # white text 
                pyautogui.rightClick()


    # Draw run button
    cv2.rectangle(img, (run_button_x, run_button_y), (run_button_x + run_button_width, run_button_y + run_button_height), button_color, cv2.FILLED)
    cv2.putText(img, "Finger Air Writing", (run_button_x + 40, run_button_y + 55), font, fontScale, text_color, 2, cv2.LINE_AA)

    # Draw stop button
    cv2.rectangle(img, (stop_button_x, stop_button_y), (stop_button_x + stop_button_width, stop_button_y + stop_button_height), button_color, cv2.FILLED)
    cv2.putText(img, "Close", (stop_button_x + 40, stop_button_y + 55), font, fontScale, text_color, 2, cv2.LINE_AA)

    # Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Display
    cv2.imshow('Virtual Mouse', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
