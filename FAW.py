import signal
import time
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
import numpy as np
from Speak import speak
import pyautogui
import os
from scipy.ndimage.measurements import center_of_mass

def mask_frame(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to binarize the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the contours of the white regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding rectangle of the largest contour
    max_area = 0
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            x, y, w, h = cv2.boundingRect(cnt)

    # Crop the image to the bounding rectangle
    cropped = img[y:y+h, x:x+w]

    # Resize the image to 28x28 pixels
    resized = cv2.resize(cropped, (28, 28))

    # Convert the image to grayscale
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Normalize the pixel values to [0, 1]
    normalized = grayscale / 255.0

    # Find the center of mass of the character
    center_y, center_x = center_of_mass(normalized)

    # Calculate the shift to center the character
    shift_x = int(14 - center_x)
    shift_y = int(14 - center_y)

    # Create the transformation matrix for shifting the character
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

    # Apply the affine transformation to center the character
    centered = cv2.warpAffine(normalized, M, (28, 28))

    # Expand the dimensions to match the model input shape (1, 28, 28, 1)
    masked = np.expand_dims(centered, axis=0)
    masked = np.expand_dims(masked, axis=-1)
    
    # Show masked image
    cv2.imshow("Masked Image 1", grayscale)
    cv2.waitKey(100)

    return masked

def run_another_file():
    cap.release()
    cv2.destroyAllWindows()
    # Specify the path to the Python file you want to run
    another_file_path = "virtual_mouse_hands.py"
    os.system("python " + another_file_path)
    os.kill(os.getpid(), signal.SIGTERM)


# importing cnn model
cnn_model = keras.models.load_model('emnist_mrf_cnn_model.h5')
cnn_digital_model = keras.models.load_model('emnist_digit_cnn_model.h5')

# Map the index to a character using a dictionary 
mapp = {}

with open('dictfile.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split(' ')
        mapp[int(line[0])] = int(line[1])


# Create a video capture object
cap = cv2.VideoCapture(0)

cap.set(3, 1080)
cap.set(4, 720)

# Create a mediapipe hands object
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands = 1)

# Create a mediapipe drawing object
mpDraw = mp.solutions.drawing_utils

# Define the thickness of the index finger tip
index_thickness = 1

# Define a list to store the circles
circles = []

# Define the font and size of the buttons text
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1

color = (255, 255, 255)

# Define the coordinates and size of the buttons
clear_x, clear_y = 10, 75 # top left corner of clear button
back_x, back_y = 140, 75 # top left corner of back button
space_x, space_y = 270, 75 # top left corner of space button
predict_x, predict_y = 400, 75 # top left corner of predict button
speak_x, speak_y = 540, 75 # top left corner of speak button
save_x, save_y = 660, 75 # top right corner of save button
color_x, color_y = 790, 75 # top right corner of save button
mouse_x, mouse_y = 920, 75 # top right corner of mouse button
ad_x, ad_y = 1050, 75 # top right corner of mouse button
button_w, button_h = 130, 75 # width and height of buttons

# Define a string variable to display on the screen
text = ""

is_digit = False

# continue the sentence when we come after certain duration.
option = pyautogui.confirm('Select an option', buttons =['1>> Start Fresh', '2>> Continue from the previous Save of txt file', '3>> Continue from the previous Save of docx file'])
if option == '1>> Start Fresh':
    text = ""
elif option == '2>> Continue from the previous Save of txt file':
    if os.path.exists('newfile.txt'):
        with open('newfile.txt', 'r') as f:
            text = f.read()
elif option == '3>> Continue from the previous Save of docx file':
    if os.path.exists('newfile.docx'):
        with open('newfile.docx', 'r') as f:
            text = f.read()

# Define thickness of circle
thickness = 1

canvas = None

chosen_mouse_operations = False

while cap.isOpened():
    # Read a frame from the video capture
    success, img = cap.read()

    # Flip the axis horizontally
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    # Convert the normalized coordinates to pixel coordinates 
    h , w , c = img.shape

    # Convert the image to RGB format
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image with the hands object
    results = hands.process(imgRGB)
    
    # Show mode which it is running in.
    mode = 'Alpha' if(not is_digit) else 'Digit'
    cv2.putText(img, mode, (850, 200), font, fontScale, (255, 200, 100), thickness, cv2.LINE_AA)

    # Check if there are any hand landmarks detected
    if results.multi_hand_landmarks:
        # Loop through each hand
        for handLms in results.multi_hand_landmarks:
            # Draw the hand landmarks on the image 
            mpDraw.draw_landmarks(img , handLms , mpHands.HAND_CONNECTIONS)

            # Get the index finger tip landmark coordinates 
            x1 , y1 = handLms.landmark[8].x , handLms.landmark[8].y

            # Get the middle finger tip landmark coordinates 
            x2 , y2 = handLms.landmark[12].x , handLms.landmark[12].y

            # Get the thumb tip landmark coordinates 
            x3 , y3 = handLms.landmark[4].x , handLms.landmark[4].y

            # Convert the normalized coordinates to pixel coordinates 
            h , w , c = img.shape 
            x1 = int(x1 * w) 
            y1 = int(y1 * h) 
            x2 = int(x2 * w) 
            y2 = int(y2 * h) 
            x3 = int(x3 * w) 
            y3 = int(y3 * h)

            # Draw a circle on the image with the index finger tip as the center with black color 
            cv2.circle(img,(x1 , y1),20,(0,0,0),index_thickness)

            # Check if the index and thumb finger are close enough (within 40 pixels)
            if abs(x1 - x3) < 40 and abs(y1 - y3) < 60:
                # Calculate the center of index and thumb coordinates 
                cx = (x1 + x3) // 2 
                cy = (y1 + y3) // 2 

                # Append the circle information to the list (center , radius) 
                circles.append((cx, cy, 20))

            # Check if the index and middle finger are close enough (within 40 pixels)
            if abs(x1 - x2) < 40 and abs(y1 - y2) < 40:
                # Calculate the center of index and middle coordinates 
                cx = (x1 + x2) // 2 
                cy = (y1 + y2) // 2 

                # Check if the center is within any of the buttons area 
                if back_x <= cx <= back_x + button_w and back_y <= cy <= back_y + button_h:
                    print("Back button pressed")
                    if len(text) > 0: 
                        text = text[:-1] # deletes last written character.
                    
                    time.sleep(1)
                
                elif clear_x <= cx <= clear_x + button_w and clear_y <= cy <= clear_y + button_h:
                    print("Clear button pressed")
                    circles = []
                    canvas = np.zeros_like(img)
                    time.sleep(1)
                    
                elif space_x <= cx <= space_x + button_w and space_y <= cy <= space_y + button_h:
                    print("Space button pressed")
                    text += " " # append "Space" to text variable

                    time.sleep(1)
                
                elif predict_x <= cx <= predict_x + button_w and predict_y <= cy <= predict_y + button_h:
                    print("Predict button pressed")

                    # Call the mask_frame function to get the masked frame 
                    masked = mask_frame(canvas)

                    # Use the model to predict the character 
                    if not is_digit:
                        prediction = cnn_model.predict(masked)
                        
                        print(prediction)
                        
                        # Get the index of the highest probability 
                        index = np.argmax(prediction)
                        
                        # ascii value is stored from the index returned by cnn model.
                        asci_value = mapp[index+1]

                        # using chr and ascii value we can get that character drawn in the frame.
                        char = chr(asci_value)
                        
                        # Append the character to text variable 
                        text += char
                    
                    else:
                        prediction = cnn_digital_model.predict(masked)

                        print(prediction)
                        
                        # Get the index of the highest probability 
                        index = np.argmax(prediction)

                        text += str(index)

                    # so all the circles drawn till now are erased.
                    circles = []

                    print('Loading screen')

                    time.sleep(1)

                    canvas = np.zeros_like(img)

                    time.sleep(1)

                elif speak_x <= cx <= speak_x + button_w and speak_y <= cy <= speak_y + button_h:
                    print("Speak button pressed")

                    # Just speak out the text written from start
                    speak(text)

                    print('Please wait for 3s')

                    time.sleep(1)
                    
                elif save_x <= cx <= save_x + button_w and save_y <= cy <= save_y + button_h:
                    print("Save button pressed")

                    option = pyautogui.confirm('Select an option', buttons =['1>> Save in txt file', '2>> Save in docx file'])

                    if option == '1>> Save in txt file':
                        fileName = "newfile.txt"
                        with open(fileName, 'w') as f:
                            f.write(text)
                        pyautogui.alert(f'Stored in CWD with filename as {fileName}')
                        
                    else:
                        filename = 'newfile.docx'
                        with open(filename, 'w') as f:
                            f.write(text)
                        pyautogui.alert(f'Stored in CWD with filename as {filename}')
                        
                    print('Writing into document.... ')

                    time.sleep(1)
                    
                elif color_x <= cx <= color_x + button_w and color_y <= cy <= color_y + button_h:
                    print("Change Color button pressed")

                    option = pyautogui.confirm('Select an option', buttons =['Blue', 'Green', 'Red', 'White', 'Black', 'Pink', 'Brown', 'Orange', 'Yellow'])

                    if option == 'Blue':
                        color = (255, 0, 0)
                        
                    elif option == 'Green':
                        color = (0, 255, 0)
                        
                    elif option == 'Red':
                        color = (0, 0, 255)
                        
                    elif option == 'White':
                        color = (255, 255, 255)
                        
                    elif option == 'Black':
                        color = (0, 0, 0)
                        
                    elif option == 'Pink':
                        color = (255, 0, 255)
                        
                    elif option == 'Brown':
                        color = (0, 75, 150)
                        
                    elif option == 'Orange':
                        color = (0, 165, 255)
                        
                    elif option == 'Yellow':
                        color = (0, 255, 255)
                        
                    print('Changing color of the pen.... ')

                    time.sleep(1)
                    
                elif mouse_x <= cx <= mouse_x + button_w and mouse_y <= cy <= mouse_y + button_h:
                    print("Change to mouse operations, button is choosen")

                    option = pyautogui.confirm('Select an option', buttons =['1>> Speak', '2>> Save in txt file', '3>> Save in docx file'])

                    if option == '1>> Speak':
                        speak(text)
                    elif option == '2>> Save in txt file':
                        fileName = "newfile.txt"
                        with open(fileName, 'w') as f:
                            f.write(text)
                        pyautogui.alert(f'Stored in CWD with filename as {fileName}')
                    else:
                        filename = 'newfile.docx'
                        with open(filename, 'w') as f:
                            f.write(text)
                        pyautogui.alert(f'Stored in CWD with filename as {filename}')
                        
                        
                    print('Opening Virtual Mouse ...')

                    run_another_file()
                
                elif ad_x <= cx <= ad_x + button_w and ad_y <= cy <= ad_y + button_h:
                    print("Alpha digit button pressed")

                    is_digit = not is_digit
                    
                    time.sleep(1)

                    
    # Loop through the circles list and draw them on the image with white color  
    for i in range(len(circles)):
        circle = circles[i]
        cv2.circle(img,circle[:2],circle[2],color,-1)
        cv2.circle(canvas,circle[:2],circle[2],(255, 255, 255),-1)

        # # Check if any circle is clicked by the index finger tip 
        # if abs(x1 - circle[0]) < circle[2] and abs(y1 - circle[1]) < circle[2]:
        #     print(f"Circle {i} clicked")
            
        #     # Keep the circle information in the list unchanged 
        #     circles[i] = circle


    # Draw three buttons on the image with white color and black text 
    cv2.rectangle(img,(clear_x , clear_y),(clear_x+button_w , clear_y+button_h),(255 ,255 ,255),-1) # clear button 
    cv2.putText(img,"Clear",(clear_x+25 , clear_y+50), font , fontScale,(0 ,0 ,0), thickness,cv2.LINE_AA) # clear text 
    
    # Draw three buttons on the image with white color and black text 
    cv2.rectangle(img,(back_x , back_y),(back_x+button_w , back_y+button_h),(255 ,255 ,255),-1) # back button 
    cv2.putText(img,"Back",(back_x+25 , back_y+50), font , fontScale,(0 ,0 ,0), thickness,cv2.LINE_AA) # back text 

    cv2.rectangle(img,(space_x , space_y),(space_x+button_w , space_y+button_h),(255 ,255 ,255),-1) # space button 
    cv2.putText(img,"Space",(space_x+25 , space_y+50), font , fontScale,(0 ,0 ,0), thickness,cv2.LINE_AA) # space text 

    cv2.rectangle(img,(predict_x,predict_y),(predict_x+button_w,predict_y+button_h),(255,255,255),-1) # predict button 
    cv2.putText(img,"Predict",(predict_x+25,predict_y+50),font,fontScale,(0,0,0),thickness,cv2.LINE_AA) # predict text 
    
    cv2.rectangle(img,(speak_x,speak_y),(speak_x+button_w,speak_y+button_h),(255,255,255),-1) # speak button 
    cv2.putText(img,"Speak",(speak_x+25,speak_y+50),font,fontScale,(0,0,0),thickness,cv2.LINE_AA) # speak text 
    
    cv2.rectangle(img,(save_x,save_y),(save_x+button_w,save_y+button_h),(255,255,255),-1) # save button 
    cv2.putText(img,"Save",(save_x+25,save_y+50),font,fontScale,(0,0,0),thickness,cv2.LINE_AA) # save text 
    
    cv2.rectangle(img,(color_x,color_y),(color_x+button_w,color_y+button_h),(255,255,255),-1) # color button 
    cv2.putText(img,"Color",(color_x+25,color_y+50),font,fontScale,(0,0,0),thickness,cv2.LINE_AA) # color text 
    
    cv2.rectangle(img,(mouse_x,mouse_y),(mouse_x+button_w,mouse_y+button_h),(255,255,255),-1) # mouse button 
    cv2.putText(img,"Mouse",(mouse_x+25,mouse_y+50),font,fontScale,(0,0,0),thickness,cv2.LINE_AA) # mouse text 
    
    cv2.rectangle(img,(ad_x,ad_y),(ad_x+button_w,ad_y+button_h),(255,255,255),-1) # alpha_digit button 
    cv2.putText(img,"Alpha/Digit",(ad_x+25,ad_y+50),font,fontScale,(0,0,0),thickness,cv2.LINE_AA) 

    # Display the text variable on the top of the screen with white color and black background 
    cv2.rectangle(img,(0 ,0),(w ,100),(0 ,0 ,0),-1) # black background 
    cv2.putText(img,text,(10, 25), font , fontScale,(255 ,255 ,255), thickness,cv2.LINE_AA) # white text 

    # Show image on a window 
    cv2.imshow("Image", img)
    cv2.imshow("Maked Image", canvas)

    # Break the loop if user presses ESC key
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

    img.fill(0)

if(not chosen_mouse_operations):
    
    # Release video capture object and destroy all windows 
    cap.release()
    cv2.destroyAllWindows()

    option = pyautogui.confirm('Select an option', buttons =['1>> Speak', '2>> Save in txt file', '3>> Save in docx file'])

    if option == '1>> Speak':
        speak(text)
    elif option == '2>> Save in txt file':
        fileName = "newfile.txt"
        with open(fileName, 'w') as f:
            f.write(text)
        pyautogui.alert(f'Stored in CWD with filename as {fileName}')
    elif option == '3>> Save in docx file':
        filename = 'newfile.docx'
        with open(filename, 'w') as f:
            f.write(text)
        pyautogui.alert(f'Stored in CWD with filename as {filename}')