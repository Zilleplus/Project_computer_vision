import readData as rd
import numpy as np


class LandmarkSet:
    coordinateCount =40
    def __init__(self,id):
        if(id > 28):
            print "!! error id is "+str(id)+"thats to large can be 28 at max"
        self.id = id
        self.data = rd.ReadLandmarksFromFiles(id)

    # get x and y coordinates
    def getCoordinates(self,rankCoordinates,rankLandmark):
        return (int(self.data[2*rankCoordinates,rankLandmark]) , int(self.data[2*rankCoordinates+1,rankLandmark]))

    def getCoordinateCount(self):
        return self.coordinateCount

    # return a copy of the dataset
    def getDataSet(self,rankLandmark):
        return self.data[:,rankLandmark].copy()

    # inject new data in an exisisting landmark
    def setData(self,data):
        self.data = data
        self.id = self.id

    def getCenter(self,rank):
        average_x=0
        average_y =0
        for i in range(0,40):
            coor = self.getCoordinates(i,rank)
            average_x += coor[0]/40
            average_y += coor[1]/40

        return (int(average_x),int(average_y))
    def getWidth(self,rank):
        for i in range(0,40):
            # find the max left, min value of x
            x_min=1000;
            # find the max right, max value of x
            x_max=0;
            coor = self.getCoordinates(i,rank)
            if coor[0]>x_max:
                x_max=coor[0]
            if coor[0]<x_min:
                x_min= coor[0]

        # substract both
        width = x_max-x_min

        return width
    def getHeight(self,rank):
        for i in range(0,40):
            # find the max left, min value of x
            y_min=0;
            # find the max right, max value of x
            y_max=0;
            coor = self.getCoordinates(i,rank)
            if coor[0]>y_max:
                y_max=coor[0]
            if coor[0]<y_min:
                y_min= coor[0]

        # substract both
        width = y_max-y_min

        return width



def getAveragePositionsLandmarks( listWithLandmarks ):
    # limit the amount of landmarks used in the average algo
    # listWithLandmarks=listWithLandmarks[1:12]

    # determine the size of the data arrays
    (n, m) = np.shape(listWithLandmarks[0].data)

    data = np.zeros((n, m))
    for i in range(0,len(listWithLandmarks)):
        for i_n in range(0,n):
            for i_m in range(0,m):
                buffer = int(listWithLandmarks[i].data[i_n,i_m]/len(listWithLandmarks))
                data[i_n,i_m] = data[i_n,i_m] + buffer
    return data

def reshapeLandmark(landmark):
    (n, m) = np.shape(landmark.data)

    newData = np.zeros((n, m))
    for i in range(0,40):
        for j in range(0,m):
            coor = landmark.getCoordinates(i,j)
            newData[i,j] = coor[0]
            newData[i+40,j] = coor[1]

    return newData

if __name__ == '__main__':
    print "running LandMarkSet.py "
    test = LandmarkSet(20)
    # print test.getCoordinates(1,1)
    print test.data