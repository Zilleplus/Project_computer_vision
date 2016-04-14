import numpy as np
import cv2
import cv2.cv as cv

pathOfData = "..\\Project_Data\_Data"
pathOfLandmarks = pathOfData + "\\Landmarks"
pathOfXLandmarks = pathOfLandmarks + "\\original"
pathOfYLandmarks = pathOfLandmarks + "\\mirrored"

def ReadLandmarksFromFiles(idLandMarkSetData):
    # construct an empy array
    landmark = np.zeros((80,8))

    for rank in range(1,9):
        # open the text file with one collum of data

        # select the right folder
        if(idLandMarkSetData>15):
            pathOfXYLandmarks = pathOfYLandmarks
        else:
            pathOfXYLandmarks = pathOfXLandmarks

        # construct the file location
        fileLocation = pathOfXYLandmarks+'\\landmarks'+str(idLandMarkSetData)+'-'+str(rank)+'.txt'
        # open the file
        f = open(fileLocation, 'r')

        # read out the numbers in the text file
        listWithNumbers = []
        for line in f:
            listWithNumbers.append(float(line))

        # after reading the whole file close it
        f.close()

        # put the collum in the array
        landmark[:,rank-1]= np.array(listWithNumbers)

    # when all collums are done, return the result
    return landmark

def readRadiograph(landmarkId):
    if landmarkId<10 :
        return cv2.imread(pathOfData+'\\Radiographs\\0'+str(landmarkId)+'.tif')
    else:
        return cv2.imread(pathOfData+'\\Radiographs\\'+str(landmarkId)+'.tif')

if __name__ == '__main__':
    print "running Read Data file : \n"
    print ReadLandmarksFromFiles(1);
