import cv2, os
import numpy as np
import subprocess
import pickle
from keras.models import load_model
import tensorflow as tf
from keras.backend import set_session
from Speak import speak
from handTrackingModule import handDetector
import pyautogui

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config = config)
set_session(session)


def get_char(mask):
    mask = mask[:, :, 0]
    mask = cv2.dilate(mask, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    contours, hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=lambda x: cv2.contourArea(x))
    x, y, w, h = cv2.boundingRect(contour)
    return mask[y: y+h, x: x+w]

# def predict_char(mask):
#     mask = cv2.resize(mask, (28,28))
#     mask = mask / 255.
#     mask = mask.reshape((1,-1)).astype('float32')
#     res = model.predict(mask)
#     for k, v in dict_file.items():
#         if v == np.argmax(res.astype(int)):
#             return k

import pandas as pd

DATASET_FILENAME = 'D:\MiniProject\DataSet from Kaggle\A_Z Handwritten Data\A_Z Handwritten Data.csv'
df = pd.read_csv(DATASET_FILENAME)

# def predict_char(mask):
#     mask = cv2.resize(mask, (28,28))
#     mask = mask.reshape((1,-1)).astype('float32')
#     _, indices = model.kneighbors(mask, n_neighbors=1)
#     row_idx = indices[0][0]
#     return dict_file[df.iloc[row_idx, 0]]

# def predict_char(mask):
#     mask = cv2.resize(mask, (28,28))
#     mask = mask.reshape((1,-1)).astype('float32')
#     pred_class = model.predict(mask)
#     return dict_file[pred_class[0]]

#CNN
def predict_char(mask):
    # mask = cv2.imread(mask, cv2.IMREAD_COLOR)
    mask = cv2.resize(mask, (28,28))
    # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = np.reshape(mask, (-1,28,28,1)).astype(np.float32)
    mask /= 255.0
    cv2.imshow('framename', mask)
    pred_class = model.predict(mask)
    
    return dict_file[np.argmax(pred_class)]


def distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def get_screen_resolution():
    output = subprocess.Popen('xrandr | grep "\*" | cut -d" " -f4', shell=True, stdout=subprocess.PIPE).communicate()[0]
    resolution = output.split()[0].split(b'x')
    return int(resolution[0]), int(resolution[1])

def translate(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)

detector = handDetector(maxHands=1, detectionCon=0.5, trackCon=0.5)

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
model = load_model('cnn_model.h5')
# a_file = open("D:\\MiniProject\\model_loader.pkl", "rb")
# dict_file = pickle.load(a_file)

# print(dict_file)
# a_file.close()
dict_file = {}
for i in range(26):
    dict_file[i] = chr(ord('A') + i)
dict_file[26] = 'back'
dict_file[27] = 'space'
print(dict_file)

# Load the model



# load the saved KNN model
# filename = 'knn_model.sav'
# model = pickle.load(open(filename, 'rb'))

# from joblib import dump, load
# filename = 'rfc_model.sav'
# rfc = load(filename)
    


NEXT_C = 0
NEXT_D = 0
NEXT_B = False
CORRECT = 0

BUFFER_SIZE = 10
BUFFER_IDX = 0
BUFFER_TOTAL_X = 0
BUFFER_TOTAL_Y = 0
buffer_x_coords = [0] * BUFFER_SIZE
buffer_y_coords = [0] * BUFFER_SIZE

canvas = None
random_color = (128, 84, 244)

STRING = ""


# continue the sentence when we come after certain duration.
option = pyautogui.confirm('Select an option', buttons =['1>> Start Fresh', '2>> Continue from the previous Save'])
if option == '1>> Start Fresh':
    STRING = ""
else:
    if os.path.exists('newfile.txt'):
        with open('newfile.txt', 'r') as f:
            STRING = f.read()


cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    image_rows, image_cols = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros_like(frame)

    frame = cv2.flip(frame, 1)
    frame = detector.findHands(frame)
    landmark_list, bbox = detector.findPosition(frame, handNo=0, draw=False)

    if len(landmark_list) > 0:
        finger_x = landmark_list[8][1]
        finger_y = landmark_list[8][2]
        cv2.circle(frame, (finger_x, finger_y), 8, (0, 255, 0), -1)

        hand_w = max(landmark_list, key=lambda x: x[1])[1] - min(landmark_list, key=lambda x: x[1])[1]
        hand_h = max(landmark_list, key=lambda x: x[2])[2] - min(landmark_list, key=lambda x: x[2])[2]

        BUFFER_TOTAL_X -= buffer_x_coords[BUFFER_IDX]
        BUFFER_TOTAL_Y -= buffer_y_coords[BUFFER_IDX]

        buffer_x_coords[BUFFER_IDX] = finger_x
        buffer_y_coords[BUFFER_IDX] = finger_y

        BUFFER_TOTAL_X += buffer_x_coords[BUFFER_IDX]
        BUFFER_TOTAL_Y += buffer_y_coords[BUFFER_IDX]

        BUFFER_IDX += 1

        if BUFFER_IDX >= BUFFER_SIZE:
            BUFFER_IDX = 0

        finger_x = BUFFER_TOTAL_X // BUFFER_SIZE
        finger_y = BUFFER_TOTAL_Y // BUFFER_SIZE

        color = (255, 0, 0)
        if distance((finger_x / hand_w, finger_y / hand_h),
                    ((landmark_list[0][1] + landmark_list[5][1]) // 2 / hand_w, (landmark_list[0][2] + landmark_list[5][2]) // 2 / hand_h)) > 0.5:
            color = (0, 0, 255)
            if NEXT_D < 4:
                NEXT_D += 1
            else:
                NEXT_C = 0
                NEXT_B = False
                cv2.circle(canvas, (finger_x, finger_y), 5, (255, 255, 255), -1)
        else:
            NEXT_C += 1

        if NEXT_C > 5 and not NEXT_B:
            NEXT_D = 0
            NEXT_B = True
            cut = get_char(canvas)
            char = predict_char(cut)

            # if char == 'back' and len(STRING) > 0:
            #     STRING = STRING[:-1]
            # elif char == 'space':
            #     if len(STRING) == 0:
            #         continue
            #     STRING += ' '
            # else:
            #     if CORRECT == 1:
            #         STRING += ' '
            #         CORRECT = 0
            #     if char != 'back':
            if char != None:
                STRING += char

            canvas = np.zeros_like(frame)
            random_color = list(np.random.choice(range(256), size=3))
            cv2.circle(frame, (int(finger_x), int(finger_y)), 8, (0, 255, 0), -1)

    combined = cv2.addWeighted(np.full_like(frame, random_color), 0.6, frame, 0.4, 0)
    combined = cv2.add(np.uint8(combined * (canvas / 255.)), np.uint8(frame * ((255 - canvas) / 255.)))

    cv2.putText(combined, STRING, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 0), 2)
    cv2.imshow('Virtual Keyboard', combined)
    cv2.imshow('Mask', canvas)
    # if(STRING!=""):print(STRING)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()

option = pyautogui.confirm('Select an option', buttons =['1>> Speak', '2>> Save in txt file'])

if option == '1>> Speak':
    speak(STRING)
elif option == '2>> Save in txt file':
    fileName = "D:\\MiniProject\\newfile.txt"
    with open(fileName, 'w') as f:
        f.write(STRING)
    pyautogui.alert(f'Stored in CWD with filename as {fileName}')
