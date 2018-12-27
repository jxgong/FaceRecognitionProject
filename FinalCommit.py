import numpy as np
import cv2, os
import tkinter

#From course notes
def readFile(path):
    with open(path, "rt") as f:
        return f.read()

def writeFile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)
#end code from course notes

def encode(contents, faceData = None, shift = 5):
    newstring = ''
    splitter = '741789654' * 4
    for s in contents:
        if s.isalpha():
            fill = 'A' if s.isupper() else 'a'
            newstring += chr(ord(fill) + (ord(s) - ord(fill)+shift)%26)
        else:
            newstring += s
    newstring += splitter
    newstring += convertToString(faceData)
    return newstring

def convertToString(faceData):
    eigenVectors, averageVector, vectors = faceData
    splitter = '7898521987412365' * 3
    valSplitter = '*'
    vectorSplitter = '+'
    result = ''
    for vector in eigenVectors:
        # print(vector)
        for component in vector:
            # print(str(component))
            result += str(component)
            result += valSplitter
        result = result[:-1]
        result += vectorSplitter
    result = result[:-1]
    result += splitter
    for component in averageVector:
        # print(str(component))
        result += str(component)
        result += valSplitter
    result = result[:-1]
    result += splitter
    for vector in vectors:
        for component in vector:
            result += str(component)
            result += valSplitter
        result += vectorSplitter
    result = result[:-1]
    return result

def convertToData(s):
    splitter = '7898521987412365' * 3
    valSplitter = '*'
    vectorSplitter = '+'
    # print(len(s.split(splitter)))
    eigenVectorsString, averageVectorString, vectorsString = s.split(splitter)
    eigenVectors = []
    for vector in eigenVectorsString.split(vectorSplitter):
        newVector = []
        for component in vector.split(valSplitter):
            newVector.append(float(component))
        eigenVectors.append(newVector)
    eigenVectors = np.array(eigenVectors)
    averageVector = []
    for component in averageVectorString.split(valSplitter):
        averageVector.append(float(component))
    averageVector = np.array(averageVector)
    vectors = []
    # print(len(vectorsString))
    for vector in vectorsString.split(vectorSplitter):
        newVector = []
        # print(len(vector.split(valSplitter)))
        for component in vector.split(valSplitter):
            if len(component)>0:
                newVector.append(float(component))
            pass
        vectors.append(newVector)
    return eigenVectors, averageVector, vectors

def decode(s):
    splitter = '741789654' * 4
    contents, facestring = s.split(splitter)
    eigenVectors, averageVector, vectors = convertToData(facestring)
    result = ''
    for c in contents:
        if c.isalpha():
            fill = 'A' if c.isupper() else 'a'
            newChar = chr(ord(fill) + ((ord(c) - ord(fill)) - 5)%26)
            result += newChar
        else:
            result += c
    return result, eigenVectors, averageVector, vectors

def getMagnitude(vector):
    #Takes in a vector and returns the magnitude recursively
    if len(vector) <= 1:
        return vector[0]
    if len(vector) == 2:
        return (vector[0]**2+vector[1]**2)**.5
    else:
        return (vector[0]**2 + getMagnitude(vector[1:])**2)**.5

def collectFaces(sampleSize):
    minSize = 128
    sampleFaces = []
    cap = cv2.VideoCapture(0)
    cap.open(0)
    ret, frame = cap.read()
    count = 0
    while len(sampleFaces) < sampleSize:
        k = cv2.waitKey(5) &0xff
        ret, frame = cap.read()
        width, height = len(frame), len(frame[0])
        progress = len(sampleFaces)/sampleSize
        cv2.rectangle(frame, (0, 0), (int(width*progress)+1, 20), (0,255),-1)
        faces = detect_face(frame)
        for face in faces:
            x, y, w, h = face
            widthcrop, heightcrop = w//10, h//10
            y += heightcrop * 2
            h -= heightcrop * 2
            x += widthcrop
            w -= widthcrop * 2
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255), 3)
            cv2.rectangle(frame, (x,y), (x+w, y+(h//10)*(count%10)), (0,255), 2)
            if count%10 == 0:
                crop = frame[y:y+h, x:x+w]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                crop = cv2.blur(crop, (2,2))
                sampleFaces.append(crop)
        cv2.imshow('Frame', frame)
        # for i in range(len(sampleFaces)):
        #     cv2.imshow('face %d' %(i), sampleFaces[i])
        if k == 27:
            cv2.destroyAllWindows()
            cap.release()
            break
        count += 1
    cv2.destroyAllWindows()
    cap.release()
    result = []
    i = 0
    for face in sampleFaces:
        face = cv2.resize(face, (minSize, minSize))
        result.append(face)
        i += 1
    return result


def transpose(a):
    #transposes a so its rows become columns and vice versa
    rows, cols = len(a), len(a[0])
    result = []
    for col in range(cols):
        addedRow = []
        for row in range(rows):
            addedRow.append(a[row][col])
        result.append(addedRow)
    return result

#From the web, normalizes vectors so that magnitude is 1.
def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0: 
       return v
    return v/norm

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))
    face_cascade = cv2.CascadeClassifier("C:/Users/Jason/Desktop/!!!112/!!!TERM PROJECT/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def convertToVector(images):
    #So it takes a list of images, crops them to be the same size NxN, and returns a 2d array where every column is a vecotor of N^2 dimensions.
    #That'll make it able to used by the PCA function.
    result = []
    size = len(images[0])
    #They should all be the same size and square
    for image in images:
        # print(image)
        vector = []
        for row in range(size):
            for col in range(size):
                vector.append(image[row,col])
        result.append(vector)
    return result

def PCA(a):
    #Takes a list of vectors and returns the eigenvectors of their covariance. Used for facial recognition.
    Arrayized = np.array(a).T
    rows, cols = len(a), len(a[0])
    averageVector = []
    for col in range(cols):
        total = 0
        for row in range(rows):
            total += a[row][col]
        total /= rows
        averageVector.append(total)
    differences = []
    for row in range(rows):
        delta = []
        for col in range(cols):
            delta.append(a[row][col]-averageVector[col])
        differences.append(delta)
    transposedDifferences = transpose(differences)
    differences, transposedDifferences = np.array(differences), np.array(transposedDifferences)
    C = np.dot(differences, transposedDifferences)
    smallEigenValues, smallEigenVectors = np.linalg.eig(C)
    eigenVectors = []
    for smallEigenVector in smallEigenVectors:
        test = np.dot(smallEigenVector,differences)
        eigenVector = []
        for val in test:
            eigenVector.append(val)
        eigenVectors.append(normalize(np.array(eigenVector)))
    return np.array(eigenVectors), np.array(averageVector)

def showEigenVectors(eigenVectors):
    #For debugging purposes only
    while True:
        for i in range(len(eigenVectors)):
            size = int(len(eigenVectors[i])**.5)
            minVal, maxVal = min(eigenVectors[i]), max(eigenVectors[i])
            result = []
            for row in range(size):
                newRow = []
                for col in range(size):
                    newRow.append(((eigenVectors[i][row*size+col])-minVal)/(maxVal-minVal))
                result.append(newRow)
            print(np.array(result))
            cv2.imshow('vector %d' %(i), np.array(result))
        cv2.waitKey()

def storeFace(sampleSize):
    images = collectFaces(sampleSize)
    vectors = convertToVector(images)
    eigenVectors, averageVector =  PCA(vectors)
    return eigenVectors, averageVector, vectors

def isFace(eigenVectors, averageVector, vectors):
    testImages = collectFaces(7)
    testVectors = convertToVector(testImages)
    for testVector in testVectors:
        testVector = np.array(testVector)
        wList = []
        # WeightList, shows how important it is.
        for eigenFace in eigenVectors:
            dotProduct = np.dot(testVector,eigenFace)
            wList.append(dotProduct)
            # print(dotProduct)
        wList = np.array(wList)
        # print(wList)
        minDiff = None
        for trainingVector in vectors:
            wList2 = []
            for eigenFace in eigenVectors:
                wList2.append(np.dot(trainingVector, eigenFace))
            wList2 = np.array(wList2)
            # print(wList2)
            difference = getMagnitude((wList-wList2)/len(trainingVector))
            difference /= (len(eigenVectors)**.5)
            if minDiff == None or difference < minDiff:
                minDiff = difference
        print(minDiff)
        if minDiff < .1:
            print('---'*5)
            return True
    print('---'*5)
    return False

def lockFile(data):
    path = data.currentPath
    name = path.split(os.sep)[-1]
    contents = readFile(path)
    if '741789654' * 4 in contents:
        data.status = 'File already locked!'
        return
    eigenVectors, averageVector, vectors = storeFace(15)
    faceData = eigenVectors, averageVector, vectors
    code = encode(contents, faceData)
    writeFile(path, code)
    data.status = 'File locked.'

def unlockFile(data):
    # eigenVectors = file['eigenVectors']
    # averageVector = file['averageVector']
    # vectors = file['vectors']
    contents = readFile(data.currentPath)
    path = data.currentPath
    try:
        contents, eigenVectors, averageVector, vectors = decode(contents)
    except:
        data.status = 'Not a locked file!'
        return
    if isFace(eigenVectors, averageVector, vectors):
        writeFile(path, contents)
        # data.lockedFiles.remove((file['Name'], file))
        data.status = 'File unlocked!'
    else:
        data.status = 'Face not recognized!\nTry again!'

def testStringConversion():
    e, a, v, = storeFace(5)
    s = convertToString((e,a,v))
    e1, a1, v1 = convertToData(s)
    print(1, e, e1)
    print(2, a, a1)
    print(3, np.array(v), np.array(v1))

#Tkinter framework from course website
# events-example0.py
# Barebones timer, mouse, and keyboard events

from tkinter import *

####################################
# customize these functions
####################################

def init(data):
    data.mode = 'starterScreen'
    data.status = 'FaceLocker V0.1!\nPress H for help.'
    data.startButtons = set()
    data.startButtons.add(Button('Lock files',
     (data.width//2, data.height//4*3),
     (data.width//2, data.height//4)))
    data.startButtons.add(Button('Unlock files',
     (0, data.height//4*3),
     (data.width//2, data.height//4)))
    data.lockButtons = set()
    data.lockButtons.add(Button('starterScreen',
     (0, data.height//4*3),
     (data.width, data.height//4)))
    data.currentPath = data.initialPath = os.sep + 'Users' + os.sep + 'Jason' + os.sep + 'Desktop'
    data.fileSelected = False
    data.fileButtons = set()
    data.lockedFiles = []
    data.unlockFilesButtons = set()
    data.unlockFilesButtons.add(Button('starterScreen',
        (0, data.height//4*3),
        (data.width, data.height//4)))
    data.lockFilePage = 0
    data.lastMode = None
    data.helpScreenText = '''
To lock files:
Choose the text file you want to hide.
Look into the camera while the progress bar fills up.
Try to vary the lighting a little.
Afterwards, the file should be scrambled.
To unlock files:
Choose the file you want to unlock.
Look into the camera and stay still until the
progress bar fills up.
Afterwards, the file should be unscrambled.
Make sure your face is well-lit!
Click to return to your last page'''
    loadIcon(data)
    fileList(data)
    data.font = 'Terminal'

def loadIcon(data):
    # https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/White_lock.svg/500px-White_lock.svg.png
    # Authored by Kelvinsong (or Kelvin13 on WikiMedia)
    # Gave permission to use in both commercial and noncommercial programs without notifying author
    # (so in public domain)
    # Created by InkScape
    fileName = 'lockIcon.png'
    data.background = PhotoImage(file = fileName)
    data.imageSize = min(data.width, data.height) - 100
    data.scale = data.imageSize/(data.background.width())
    data.scale *= 10
    data.scale = round(data.scale)
    data.background = data.background.zoom(data.scale, data.scale)
    data.background = data.background.subsample(10,10)

class Button(object):
    def __init__(self, label, position, dimensions, isFile = False, contents = None, color = None, textColor = 'green'):
        self.x, self.y = position
        self.w, self.h = dimensions
        self.label = label
        self.color = color
        self.isFile = isFile
        self.isHighlighted = False
        self.textColor = textColor

    def hasClicked(self, x, y):
        if (x >= self.x and x <= self.x + self.w and 
            y >= self.y and y <= self.y + self.h):
            if self.isHighlighted:
                return True
            else:
                self.isHighlighted = True
                return False
        else:
            self.isHighlighted = False
            return False
    def draw(self, canvas, font = 'Arial'):
        if self.isFile:
            text = self.label.split(os.sep)[-1]
        else:
            text = self.label
        color = 'lightGreen' if self.isHighlighted else self.color
        canvas.create_rectangle(self.x, self.y, self.x+self.w, self.y+self.h, fill = color, outline = color)
        canvas.create_text(self.x + self.w//2, self.y + self.h//2,
                        text = text, font = '%s %d' %(font, self.h//3), fill = self.textColor)

    def __hash__(self):
        return hash((self.x, self.y, self.w, self.h))

def mousePressed(event, data):
    if data.mode == 'helpScreen': data.mode = data.lastMode
    elif data.mode == 'starterScreen': starterScreenMousePressed(event, data)
    elif data.mode == 'Lock files': lockFilesMousePressed(event, data)
    elif data.mode == 'Unlock files': unlockFilesMousePressed(event, data)
    # fileList(data)

def lockFilesMousePressed(event, data):
    for button in data.lockButtons:
        if button.hasClicked(event.x, event.y):
            data.mode = button.label
            data.status = 'FaceLocker V0.1!\nPress H for help.'
            break
    for button in data.fileButtons:
        if button.hasClicked(event.x, event.y):
            if button.label == 'Next':
                data.lockFilePage += 1
                fileList(data)
            elif button.label == 'Prev':
                if data.lockFilePage > 0:
                    data.lockFilePage -= 1
                    fileList(data)
            elif button.isFile:
                data.currentPath = button.label
                if not os.path.isdir(data.currentPath):
                    if data.currentPath.split('.')[-1] == 'txt':
                        lockFile(data)
                        data.lockFilePage = 0
                    else:
                        data.status = 'Not a text file!'
                    data.currentPath = data.initialPath
                    fileList(data)
                    data.mode = 'starterScreen'
                fileList(data)
            break

def starterScreenMousePressed(event, data):
    for button in data.startButtons:
        if button.hasClicked(event.x, event.y) and not button.isFile:
            data.mode = button.label

def unlockFilesMousePressed(event, data):
    for button in data.unlockFilesButtons:
        if button.hasClicked(event.x, event.y):
            # if button.isFile:
            #     unlockFile(button.contents, data)
            #     updateLockedFiles(data)
            #     data.mode = 'starterScreen'
            # else:
            data.mode = button.label
            data.status = 'FaceLocker V0.1!\nPress H for help.'
            break
    for button in data.fileButtons:
        if button.hasClicked(event.x, event.y):
            if button.label == 'Next':
                data.lockFilePage += 1
                fileList(data)
            elif button.label == 'Prev':
                if data.lockFilePage > 0:
                    data.lockFilePage -= 1
                fileList(data)
            elif button.isFile:
                data.currentPath = button.label
                if not os.path.isdir(data.currentPath):
                    if not data.currentPath.split('.')[-1] == 'txt':
                        data.status = 'Not a text file!'
                    else:
                        unlockFile(data)
                    data.currentPath = os.sep + 'Users' + os.sep + 'Jason' + os.sep + 'Desktop'
                    data.mode = 'starterScreen'
                fileList(data)
                data.lockFilePage = 0
            break

def keyPressed(event, data):
    # use event.char and event.keysym
    if event.keysym == 'h':
        data.lastMode = data.mode
        data.mode = 'helpScreen'
    if event.keysym == 'p':
        print(data.currentPath)
    if event.keysym == 'w':
        writeFile(os.sep + 'Users' + os.sep + 'Jason' + os.sep + 'Desktop' + os.sep + '!!ALSOTOP SECRET.txt', '''\
Krabby Patty Recipe:
Krabbies
Patties
Plankton
Seaweed
Pickles''')
# ^For testing purposes only

def timerFired(data):
    if data.mode == 'Lock this file':
        pass

def fileList(data):
    newButtons = set()
    data.fileButtons = newButtons
    if not os.path.isdir(data.currentPath):
        data.fileSelected = True
        return
    nameList = os.listdir(data.currentPath)[data.lockFilePage * 4:]
    if nameList == []:
        data.lockFilePage -= 1
        nameList = os.listdir(data.currentPath)[data.lockFilePage * 4:]
    files = []
    for name in nameList:
        files.append(data.currentPath + os.sep + name)
    width = data.width//2
    height = (data.height//4*3)/(5)
    lastPath = data.currentPath.split(os.sep)[:-1]
    result = ''
    for file in lastPath:
        result += file
        result += os.sep
    lastPath = result[:-1]
    firstButton = Button(lastPath, (0, 0), (data.width, height), isFile= True)
    data.fileButtons.add(firstButton)
    for i in range(4):
        if i >= len(files):
            break
        button = Button(files[i], (data.width//4, (i+1)*height), (width, height), isFile = True)
        data.fileButtons.add(button)
    data.fileButtons.add(Button('Next', (data.width//4*3, data.height//4), (data.width//4, data.height//4)))
    data.fileButtons.add(Button('Prev', (0, data.height//4), (data.width//4, data.height//4)))

def redrawAll(canvas, data):
    # canvas.create_image(data.width//2, data.height//2, image = data.background)
    if data.mode == 'helpScreen': helpScreenRedrawAll(canvas, data)
    if data.mode == 'starterScreen': starterScreenRedrawAll(canvas, data)
    if data.mode == 'Lock files': lockFilesRedrawAll(canvas, data)
    if data.mode == 'Unlock files': unlockFilesRedrawAll(canvas, data)

def helpScreenRedrawAll(canvas, data):
    canvas.create_text(10, 10, text = data.helpScreenText, anchor = NW, font = '%s 20' %(data.font), fill = 'green')

def lockFilesRedrawAll(canvas, data):
    for button in data.lockButtons:
        button.draw(canvas, data.font)
    if not data.fileSelected:
        for button in data.fileButtons:
            button.draw(canvas, data.font)

def starterScreenRedrawAll(canvas, data):
    canvas.create_text(data.width//2, data.height//4, text = data.status, font = '%s 50' %(data.font), fill = 'green')
    for button in data.startButtons:
        button.draw(canvas, data.font)

def unlockFilesRedrawAll(canvas, data):
    for button in data.fileButtons:
        button.draw(canvas, data.font)
    for button in data.unlockFilesButtons:
        button.draw(canvas, data.font)


####################################
# use the run function as-is
####################################

def run(width=300, height=300):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='black', width=0)
        redrawAll(canvas, data)
        canvas.update()    

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFired(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)
    root = Tk()
    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timerDelay = 100 # milliseconds
    # create the root and the canvas
    init(data)
    data.root = root
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")

run(900, 500)