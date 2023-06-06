import pandas as pd
import cv2
import numpy as np

df = pd.read_csv('D:\\MiniProject\\DataSet from Eminist\\emnist-byclass-train.csv')

labels = df.iloc[:, 0]
images = df.iloc[:, 1:]

mapp = {}

with open('D:\\MiniProject\\dictfile.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split(' ')
        mapp[int(line[0])] = int(line[1])
        
def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

for i in range(150, 165):
    img = images.iloc[i, :]
    print(chr(mapp[labels.iloc[i]]))

    try:
        print(img.shape)
    except AttributeError:
        print('Not a numpy array')

    img = np.asarray(img)
    img = img.astype('uint8')
    
    vimg = rotate(img)

    print(img.shape, vimg.shape)

    img = img.reshape([28, 28])

    cv2.imshow('Horizontal Image', img)
    cv2.imshow('Vertical Image', vimg)
    # Break the loop if user presses ESC key
    if cv2.waitKey(0) & 0xff == ord('q'):
        cv2.destroyAllWindows()
