import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

# for video capturing

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
counter = 0

labels = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "yes"]

# first i go for data collection

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        # getting axis and size ie x, y-axis, width & height
        x, y, w, h = hand['bbox']

        # getting the white background of the image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # cropping the image
        imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]
        imgCropShape = imgCrop.shape

        aspectratio = h / w

        # image resizing
        if aspectratio > 1:
            # if the aspect ratio is > 1 then image is wider than its taller
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw= False)
            print(prediction, index)
        else:
            # if the aspect ratio is > 1 then image is taller than its wider
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, labels[index],(x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)

    # # to open cameras
    # cv2.imshow("Image", img)
    # key = cv2.waitKey(1)
    # if key == ord('h'):
    #     counter += 1
    #     cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
    #     print(counter)
