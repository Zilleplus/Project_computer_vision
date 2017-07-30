# the format of the eigenvectors is [x1 x2 ... xn y1 y2 ... yn]^T

import numpy as np

class ActiveShapeModel:
    def __init__(self,mean,eigenvectors,eigenvalues):
        self.mean = mean
        self.eigenvectors = eigenvectors
        self.eigenvalues=eigenvalues
        self.dimension = np.size(eigenvalues)
        # set the default shape on eigenvalues
        self.setShape(eigenvalues)
        self.x_offset = 0
        self.y_offset = 0
        self.scale = 1
    def scaleUp(self,scale):
        self.setShape(self.weights_eigenvectors*scale)

    # set the shape according to the weights you want
    def setShape(self,weights_eigenvectors):
        self.weights_eigenvectors=weights_eigenvectors
        numberOfWeights = np.size(weights_eigenvectors)

        if(numberOfWeights != self.dimension):
            print "error incorrect weights the dim="+str(self.dimension)+ "you gave me "+numberOfWeights+ "weights"
            return 0;
        else:
            # set the starting point on the mean
            self.data = self.mean.reshape( np.size(self.eigenvectors[:,0]) )
            # self.data = np.zeros(np.size(self.eigenvectors[:,1]))

            # add the eigenvectors occording too there weight
            for i in range(0,numberOfWeights):
                self.data = self.data + weights_eigenvectors[i]*self.eigenvectors[:,i]

    # after setShape has been called, its possible to get the individual coordinates
    def getCoordinatesModel(self,rankCoordinates):
        return (
                            int(self.data[rankCoordinates]*self.scale + self.x_offset)
                           # ,int(self.data[rankCoordinates+self.dimension/2]*self.scale + self.y_offset)
                            ,int(self.data[rankCoordinates+40]*self.scale + self.y_offset)
                        )
    def getCentrum(self):
        mx = 0
        my = 0
        for i in range(0,len(self.mean)/2 -1):
            m_x =+ self.mean[i]/len(self.mean)/2
            m_y =+ self.mean[i+len(self.mean)/2]/len(self.mean)/2
        return (mx+self.x_offset,my+self.y_offset)

    def getGrowVector(self,rank_vector,scale):
        # growVector = np.array([self.data[rank_vector], self.data[rank_vector+self.dimension/2] ])

        # calculate the vector
        b = np.array(self.getCoordinatesModel((rank_vector)%40)) \
            - np.array(self.getCoordinatesModel((rank_vector-1)%40))
        growVector = np.array([-b[1] , b[0] ])

        # normalize the vector
        growVector = growVector/np.linalg.norm(growVector)

        distance_center_point = np.linalg.norm(np.array(self.getCoordinatesModel(rank_vector))-np.array(self.getCentrum()))
        distance_center_grownpoint = np.linalg.norm(np.array(self.getCoordinatesModel(rank_vector)) + growVector*10 - np.array(self.getCentrum()))

        # determine the direction
        if(distance_center_point>distance_center_grownpoint):
            growVector = -growVector

        # scale it up
        growVector = growVector*scale

        position = np.array(self.getCoordinatesModel(rank_vector))
        growVector = growVector+position

        return (int(growVector[0]),int(growVector[1]))

    def getGrowVectorStartScale(self,rank_vector):
        growVector = np.array([self.data[rank_vector], self.data[rank_vector+self.dimension/2] ])
        startingScale = np.linalg.norm(growVector)
        return 0

    def calcNewModel(self,ref_points):
        # calculate new shape with new ref points, LEAST SQUARES
        # Ax = b find x when rank(A)<n
        print "--calculating new model"
        b = np.zeros(80)

        for i in range(0,40):
            point = ref_points[i]

            point_x = (point[0] - self.x_offset)/self.scale
            point_y = (point[1] - self.y_offset)/self.scale

            b[i] = int(point_x)
            b[i+40] = int(point_y)

        # A = eigenvectors
        # b =  b-self.mean
        # solve the system
        (x_solution, res, R, s) = np.linalg.lstsq(self.eigenvectors,b)

        # create a new model (mean,eigenvectors,eigenvalues)
        newModel = ActiveShapeModel(self.mean,self.eigenvectors,x_solution)
        # scale everything back
        newModel.x_offset=self.x_offset
        newModel.y_offset=self.y_offset
        newModel.scale = self.scale
        # return the new model
        return newModel
    def rotate(self,angle):
        # rotate all the eigenvectors
        (lenght_eigenvectors, amount_of_eigenvectors) = np.shape(self.eigenvectors)
        for i in range(0,amount_of_eigenvectors):
            for j in range(0,lenght_eigenvectors/2):
                x = self.eigenvectors[j,i]
                y = self.eigenvectors[j+lenght_eigenvectors/2,i]
                (newx,newy) = self.rotate_point(x,y,angle)
                self.eigenvectors[j,i] = newx
                self.eigenvectors[j+lenght_eigenvectors/2,i] = newy
        # rotate the mean
        for i in range(0,40):
            x = self.mean[i]
            y = self.mean[i+40]
            (newx,newy) = self.rotate_point(x,y,angle)
            self.mean[i] = newx
            self.mean[i+40] = newy
        self.setShape(self.weights_eigenvectors)
    def getNewRotatedModel(self,angle):
        mean = self.mean.copy()
        eigenvectors = self.eigenvectors.copy()
        eigenvalues = self.eigenvalues.copy()

        newModel = ActiveShapeModel(mean,eigenvectors,eigenvalues)

        newModel.x_offset=self.x_offset
        newModel.y_offset=self.y_offset
        newModel.scale= self.scale

        newModel.setShape(self.weights_eigenvectors.copy())

        newModel.rotate(angle)

        return newModel
    def getNewShiftedModel(self,x_shift,y_shift):
        mean = self.mean.copy()
        eigenvectors = self.eigenvectors.copy()
        eigenvalues = self.eigenvalues.copy()

        newModel = ActiveShapeModel(mean,eigenvectors,eigenvalues)

        newModel.x_offset=self.x_offset+x_shift
        newModel.y_offset=self.y_offset+y_shift
        newModel.scale= self.scale

        newModel.setShape(self.weights_eigenvectors.copy())

        return newModel

    def rotate_point(self, px, py, angle):
        # center = self.getCentrum()
        # cx= center[0]
        # cy= center[1]

        s = np.sin(angle)
        c = np.cos(angle)

        # translate point back to origin:
        # px -= cx
        # py -= cy

        # rotate point
        xnew = px * c - py * s
        ynew = px * s + py * c

        # translate point back:
        # px = xnew + cx
        # py = ynew + cy

        return (xnew , ynew)
