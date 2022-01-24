# Detecting face and applying sunglasses with fancy effects automatically 
# This process can be applied to a single static image or real-time video stream

# Algorithm Outline:
# Step 1: facial landmark detection using Dlib
# Step 2: using landmarks to resize and locate sunglasses and apply on face
# Step 3: applying specular reflection on glasses
# Step 4: applying fancy effect on glasses

import numpy as np
import cv2
import dlib


def detectFaceDlib(img, showFaces=True):
    # dlib face detector
    detector = dlib.get_frontal_face_detector()
    faceRects = detector(img, 0)
    print("Number of faces detected: {}".format(len(faceRects)))

    if showFaces:
        imgCopy = img.copy()
        for i, faceRect in enumerate(faceRects):
            cv2.rectangle(imgCopy, (faceRect.left(), int(faceRect.top())), (faceRect.right(), faceRect.bottom()), (255, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow("Detected Faces", imgCopy)

    return faceRects


def detectFacialLandmarks(img, faceRects, landmarkDetector, showLandmarks=True):
    landmarksAll = []

    # Loop over all detected face rectangles
    for i in range(0, len(faceRects)):
        newRect = faceRects[i]

        # For every face rectangle, run landmarkDetector
        landmarks = landmarkDetector(img, newRect)

        # Print number of landmarks
        if i == 0:
            print("Number of landmarks",len(landmarks.parts()))

        # Store landmarks for current face
        landmarksAll.append(landmarks)

    if showLandmarks:
        displayLandmarks(img, landmarksAll)

    return landmarksAll


def displayLandmarks(img, landmarksAll, color=(0, 255, 0), radius=3):
    imgCopy = img.copy()
    for landmarks in landmarksAll:
        for p in landmarks.parts():
            cv2.circle(imgCopy, (p.x, p.y), radius, color, -1)
    cv2.imshow("Facial Landmarks", imgCopy)


def rescaleGlasses(img, landmarks, imgGlasses):
    # resize the glasses to fit the face
    faceWidth = (landmarks.part(16).x - landmarks.part(0).x) * 1.1  # measure face width using the left and right face landmarks
    glassesWidth = imgGlasses.shape[1]
    scaleFactor = faceWidth / glassesWidth
    imgGlassesReScaled = cv2.resize(imgGlasses, None, fx=scaleFactor, fy=scaleFactor)

    imgGlassesRescaledBGR = imgGlassesReScaled[..., 0:3]
    imgGlassesRescaledMask = imgGlassesReScaled[..., -1]
    imgGlassesRescaledMask = cv2.merge((imgGlassesRescaledMask, imgGlassesRescaledMask, imgGlassesRescaledMask)) / 255.0

    return imgGlassesRescaledBGR, imgGlassesRescaledMask


# generate rescaled glasses and masks for all faces detected in image
def generateAllGlasses(img, landmarksAll, imgGlasses):
    imgGlassesRescaledList = [] 
    maskGlassesRescaledList = []
    for landmarks in landmarksAll:
        imgGlassesRescaledBGR, imgGlassesRescaledMask = rescaleGlasses(img, landmarks, imgGlasses)
        imgGlassesRescaledList.append(imgGlassesRescaledBGR)
        maskGlassesRescaledList.append(imgGlassesRescaledMask)
    return imgGlassesRescaledList, maskGlassesRescaledList


# Inputs:
# foreground: uint8
# background: uint8
# bgMask: float (0 ~ 1)
# opacity: float (0 ~ 1)
def alphaBlend(background, foreground, fgMask, opacity): 
    background.astype(float)
    foreground.astype(float)
    blend = background.copy()
    fgMaskWithOpacity = fgMask * opacity
    blend = background * (1 - fgMaskWithOpacity) + foreground * fgMaskWithOpacity
    return blend.astype('uint8')

# imgGlassesRescaledList: rescaled glasses for each detected face
# maskGlassesRescaledList: rescaled glaeeses mask for eacdh detected face
# landmarksAll: landmark locations for all detected faces
def autoApplyGlassesOnFace(img, imgGlassesRescaledList, maskGlassesRescaledList, landmarksAll, opacity):
    glassesRegionList = []
    for glassesBGR, glassesMask, landmarks in zip(imgGlassesRescaledList, maskGlassesRescaledList, landmarksAll):
        eyeCenterLoc = ((landmarks.part(27).x + landmarks.part(28).x) // 2, (landmarks.part(27).y + landmarks.part(28).y) // 2)  
        glassHeight = glassesBGR.shape[0]
        glassWidth = glassesBGR.shape[1]
        upperLeftRowInd = eyeCenterLoc[1] - glassHeight // 2
        upperLeftColInd = eyeCenterLoc[0] - glassWidth // 2

        # blend using the alpha channel of the glasses image
        glassesRegion = np.s_[upperLeftRowInd : upperLeftRowInd+glassHeight, upperLeftColInd : upperLeftColInd+glassWidth]
        img[glassesRegion] = alphaBlend(img[glassesRegion], glassesBGR, glassesMask, opacity)
        glassesRegionList.append(glassesRegion)
    return img, glassesRegionList


if __name__ == '__main__':
    PREDICTOR_PATH = './models/shape_predictor_68_face_landmarks.dat'
    landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

    # Adjust opacity here
    glassesOpacity = 0.7
    spectacularReflectionOpacity = 0.4
    fancyEffectOpacity = 0.4

    # Reading images
    img = cv2.imread('two_faces.jpg')
    img = cv2.resize(img, None, fx = 0.3, fy = 0.3)
    imgGlasses = cv2.imread('sunglass.png', cv2.IMREAD_UNCHANGED)
    cv2.imshow("Input Image", img)    

    specularReflection = cv2.imread('high_contrast_landscape.jpg', 0)
    specularReflection = cv2.resize(specularReflection, (img.shape[1], img.shape[0]))
    cv2.imshow("Spectacular Reflection", specularReflection)

    fancyEffect = cv2.imread('waterdrop.jpg')
    fancyEffect = cv2.resize(fancyEffect, (img.shape[1], img.shape[0]))
    cv2.imshow("Fancy Effect", fancyEffect)

    # Step 1: facial landmark detection using Dlib
    faceRects = detectFaceDlib(img)
    landmarksAll = detectFacialLandmarks(img, faceRects, landmarkDetector)

    # Step 2: using landmarks to resize and locate glasses and apply on face
    imgGlassesRescaledList, maskGlassesRescaledList = generateAllGlasses(img, landmarksAll, imgGlasses)
    facesWithGlasses, glassesRegionList = autoApplyGlassesOnFace(img, imgGlassesRescaledList, maskGlassesRescaledList, landmarksAll, glassesOpacity)
    cv2.imshow('Faces with glasses', facesWithGlasses)

    # Step 3: apply specular reflection
    facesWithGlassesRef = facesWithGlasses.copy()
    specularReflection = cv2.resize(specularReflection, (img.shape[1], img.shape[0]))
    specularReflection_C3 = cv2.merge((specularReflection, specularReflection, specularReflection))
    for maskGlassesRescaled, glassesRegion in zip(maskGlassesRescaledList, glassesRegionList):
        facesWithGlassesRef[glassesRegion] = alphaBlend(facesWithGlassesRef[glassesRegion], specularReflection_C3[glassesRegion], maskGlassesRescaled, spectacularReflectionOpacity)
    cv2.imshow('Glasses With Specular Reflection', facesWithGlassesRef)

    # Step 4: apply fancy effect
    facesWithGlassesFancy = facesWithGlasses.copy()
    for maskGlassesRescaled, glassesRegion in zip(maskGlassesRescaledList, glassesRegionList):
        facesWithGlassesFancy[glassesRegion] = alphaBlend(facesWithGlassesFancy[glassesRegion], fancyEffect[glassesRegion], maskGlassesRescaled, fancyEffectOpacity)
    cv2.imshow('Glasses With Fancy Effect', facesWithGlassesFancy)

    cv2.waitKey()