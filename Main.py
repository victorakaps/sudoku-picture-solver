import cv2
import numpy as np
import tensorflow as tf

# Loading CNN Model
model = tf.keras.models.load_model('Utils/model.h5')


# Preprocessing Image
def preProcess(img):
    # convert image to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # adding blur
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)  # apply adaptive threshold
    return imgThreshold


# get warp perspective
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


# finding the puzzle square
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


# split each cell into different square image(s) i.e. 81 images
def splitBoxes(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes


# get digit of from every square
def getPredection(boxes, model):
    result = []
    for image in boxes:
        # prepare image sqaure
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
        img = cv2.resize(img, (28, 28))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)

        # find its digit
        predictions = model.predict(img)
        classIndex = model.predict_classes(img)
        probabilityValue = np.amax(predictions)

        # append prediction into an array
        if probabilityValue > 0.8:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result


def displayNumbers(img, numbers, color=(0, 255, 0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range(0, 9):
        for y in range(0, 9):
            if numbers[(y*9)+x] != 0:
                cv2.putText(img, str(numbers[(y*9)+x]), (x*secW+int(secW/2)-10, int(
                    (y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2, cv2.LINE_AA)
    return img


# make output stack image
def stackImages(imgArray, scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(
                    imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(
                        imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    return ver


def solve(bo):
    find = find_empty(bo)
    if not find:
        return True
    else:
        row, col = find
    for i in range(1, 10):
        if valid(bo, i, (row, col)):
            bo[row][col] = i
            if solve(bo):
                return True
            bo[row][col] = 0
    return False


def valid(bo, num, pos):
    # Check row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False
    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False
    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if bo[i][j] == num and (i, j) != pos:
                return False
    return True


def find_empty(bo):
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                return (i, j)
    return None


pathImage = "Utils/sudoku.jpg"
heightImg = 450
widthImg = 450


# load image
img = cv2.imread(pathImage)

# resizing it to a square
img = cv2.resize(img, (widthImg, heightImg))

# process image to grayscale and gausian blur
imgThreshold = preProcess(img)

# dummy placholder image(complete black)
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)

imgContours = img.copy()
imgBigContour = img.copy()

# find contours
contours, hierarchy = cv2.findContours(
    imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# draw contours
cv2.drawContours(imgContours, contours, -1, (0, 255, 0),
                 3)

# find the biggest contour i.e. sudoku square itself
biggest, maxArea = biggestContour(contours)
if biggest.size != 0:
    biggest = reorder(biggest)

    # draw sudoku contour
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255),
                     25)

    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [
                      widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgDetectedDigits = imgBlank.copy()
    imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgSolvedDigits = imgBlank.copy()

    boxes = splitBoxes(imgWarpColored)

    numbers = getPredection(boxes, model)

    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0, 0, 1)

    # get solution
    board = np.array_split(numbers, 9)
    try:
        solve(board)
    except:
        pass
    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers = flatList*posArray
    imgSolvedDigits = displayNumbers(imgSolvedDigits, solvedNumbers)

    # bootstrap solution
    pts2 = np.float32(biggest)
    pts1 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [
                      widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(
        imgSolvedDigits, matrix, (widthImg, heightImg))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)

    imageArray = ([img, inv_perspective])
    stackedImage = stackImages(imageArray, 1)
    cv2.imshow('Output', stackedImage)

else:
    print("couldn't find suduoku sqaure")

cv2.waitKey(0)
