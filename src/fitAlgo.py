# used libs
import cv2
import cv2.cv as cv
import numpy as np

# self made libs
import readData as rd
import Teeth as th
import ActiveShapeModel as asm
import math

number_of_points = 40

def findNewRefPoints(edges,scales,model,shrink,max_opti,stepsize,range_weights,tresh_hold):
    notFound = np.ones(number_of_points)
    # grow the vectors till we find "enough edge"
    for delta in range(0,max_opti):
        for j in range(0, number_of_points):
            if(shrink==True):
                currentWeight=0
            else:
                currentVector = model.getGrowVector(j,scales[j])
                currentWeight=calc_weight(edges,currentVector,range_weights=range_weights)
            # shrinking algo
            if(notFound[j]==1):
                newVector = model.getGrowVector(j,scales[j]+delta*stepsize)
                if(newVector[0]>100):
                    break
                if(shrink==True):
                    # increase the scale
                    newScale= scales[j]-delta*stepsize
                    # get the new vector
                    newVector = model.getGrowVector(j,newScale)
                    # shape of edges
                    (size_n , size_m) = edges.shape

                    if(newVector[0]>100):
                        # stop its too big, your at the border
                        notFound[j] = 0
                    else:
                        if( (newVector[0]+range_weights<=size_m-1) and  (newVector[1]+range_weights<=size_n-1) ):
                            weight = calc_weight(edges,newVector,range_weights)

                            if weight>tresh_hold and weight>currentWeight:
                                # print "stopped"
                                notFound[j] =0
                                scales[j] = newScale

            # growing algo
            if(notFound[j]==1):
                # increase the scale
                newScale= scales[j]+delta*stepsize
                # get the new vector
                newVector = model.getGrowVector(j,newScale)

                (size_n , size_m) = edges.shape
                if( (newVector[0]+range_weights<=size_m-1) and  (newVector[1]+range_weights<=size_n-1) and newVector[0]>=0 and newVector[1]>=0):
                    weight = calc_weight(edges,newVector,range_weights)
                    # if you are against an edge then stop
                    if weight>tresh_hold and weight>currentWeight:
                        notFound[j] = 0
                        scales[j] = newScale

    return scales

def calc_weight(edges,newVector,range_weights):
    weight=0
    (size_n , size_m) = edges.shape
    if( (newVector[0]+range_weights<=size_m-1) and  (newVector[1]+range_weights<=size_n-1) and newVector[0]>=0 and newVector[1]>=0):
        for k in range(0,range_weights):
            weight += edges[newVector[1],newVector[0]]
            weight += edges[newVector[1]+k,newVector[0]]
            weight += edges[newVector[1],newVector[0]+k]
            weight += edges[newVector[1]+k,newVector[0]+k]

            weight += edges[newVector[1],newVector[0]]
            weight += edges[newVector[1]-k,newVector[0]]
            weight += edges[newVector[1],newVector[0]-k]
            weight += edges[newVector[1]-k,newVector[0]-k]
    return weight
def calc_weight_model(edges,model,range_weights):
    weight =0;
    for i in range(0,40):
        coor = model.getCoordinatesModel(i)
        weight += calc_weight(edges,coor,range_weights)
    return weight