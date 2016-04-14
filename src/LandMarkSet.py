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

if __name__ == '__main__':
    print "running LandMarkSet.py "
    test = LandmarkSet(1)
    print test.getCoordinates(1,1)