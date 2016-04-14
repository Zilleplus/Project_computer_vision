import numpy as np
import scipy as sp
import scipy.spatial as sps
import numpy.linalg as npl

class Teeth:
    numberOfPointsLandmark = 40
    def procus(self):
        # define the array that will contain the transformed landmarks
        self.procrustes_data = np.zeros((80,self.numberOfLandmarks))

        # define the two arrays use with the procrustes
        data1 = np.zeros((self.numberOfPointsLandmark,2))
        data2 = np.zeros((self.numberOfPointsLandmark,2))

        for i_rankLandmark in range(0,self.numberOfLandmarks):
            for i in range(0,self.numberOfPointsLandmark):
                if i_rankLandmark == 0:
                    data1[i,0] = self.data[i*2,0]
                    data1[i,1] = self.data[i*2 +1,0]
                else:
                    data2[i,0] = self.data[i*2,i_rankLandmark]
                    data2[i,1] = self.data[i*2 +1,i_rankLandmark]

                    # execute the transformation
                    (mtx1, mtx2, disparity) = sps.procrustes(data1,data2)

                    # safe the result in the format [ x1 x2 ... xn y1 y2 ... yn ]^T
                    ## safe x coordinates
                    self.procrustes_data[0:self.numberOfPointsLandmark,i_rankLandmark]= mtx2[:,0]
                    ## safe y coordinates
                    self.procrustes_data[self.numberOfPointsLandmark:2*self.numberOfPointsLandmark,i_rankLandmark] = mtx2[:,1]


                    # save the normalized version of data1, safe it in the first collum
                    # its obvious that we should do this only once as each loop will have the same one
                    if i_rankLandmark==1:
                        self.procrustes_data[0:self.numberOfPointsLandmark,0] = mtx1[:,0]
                        self.procrustes_data[self.numberOfPointsLandmark:2*self.numberOfPointsLandmark,0] = mtx1[:,1]

        print self.procrustes_data[0:20,0:4]

    # return the normalized coordinates times 1000, usefull for debugging
    def getNormalizedCoordinatesModelIndividualTeeth(self,rankLandmark,rankCoordinates):
        return (
                    int(self.procrustes_data[rankCoordinates,rankLandmark]*1000+500)
                   ,int(self.procrustes_data[rankCoordinates+self.numberOfPointsLandmark,rankLandmark]*1000+500)
                )

    def __init__(self,rankLandmark,listWithLandmarks):
            self.rankLandmark = rankLandmark
            self.numberOfLandmarks = len(listWithLandmarks)

            self.data = np.zeros((80,self.numberOfLandmarks))

            for i in range(0,self.numberOfLandmarks):
                self.data[:,i] = listWithLandmarks[i].getDataSet(self.rankLandmark)

            self.procus()