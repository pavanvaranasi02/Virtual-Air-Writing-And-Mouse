import cv2

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    cv2.imshow('Test', frame)
    cv2.imwrite('test.png', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break