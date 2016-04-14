# the format of the eigenvectors is [x1 x2 ... xn y1 y2 ... yn]^T

class ActiveShapeModel:
    def __init__(self,mean,eigenvectors,eigenvalues):
        self.mean = mean
        self.eigenvectors = eigenvectors
    def getCoordinatesModel(self,weights_eigenvectors):
        return 0 #TODO