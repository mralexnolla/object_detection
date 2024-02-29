import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# for video capturing

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 300
counter = 0

folder = "Data/Hello"

# first i go for data collection

while True:
    success, img = cap.read()
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
        else:
            # if the aspect ratio is > 1 then image is taller than its wider
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    # to open cameras
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
