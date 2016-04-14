# view the landmarks onto the pictures

# used libs
import cv2
import cv2.cv as cv
import numpy as n

# self made libs
import readData as rd
import Teeth as th
import ActiveShapeModel as asm

import LandMarkSet as lms
demo1=False
demo2=False
demo3=True

if demo1==True:
    test = lms.LandmarkSet(1)
    img = rd.readRadiograph(1)

    for teeth in range(0,8):
        for i in range(0,20):
            cv2.line(img,test.getCoordinates(i*2,teeth),test.getCoordinates(i*2+1,teeth),100,5)

        small = img.copy()
        small = cv2.resize(small,(700,500))

    cv2.imshow('small',small)
    cv2.waitKey(0)

if demo2==True:
    # do procus... anal... -------------
    listWithLandmarks = [];
    #read out the first 10 landmarks
    for i in range(1,11):
        listWithLandmarks.append(lms.LandmarkSet(i))

    # construct a teeth model
    t = th.Teeth(1,listWithLandmarks)

    img = rd.readRadiograph(1)
    #draw line between 2 points to get a striped version of the model
    for teeth in range(0,8):
        for i in range(0,20):
            #print "="+str(t.getNormalizedCoordinatesModelIndividualTeeth(teeth,i*2))
            cv2.line(img,t.getNormalizedCoordinatesModelIndividualTeeth(teeth,i*2),t.getNormalizedCoordinatesModelIndividualTeeth(teeth,i*2+1),100,5)


        small = img.copy()
        small = cv2.resize(small,(700,500))

        cv2.imshow('small',small)
        cv2.waitKey(0)

if demo3==True:
    listWithLandmarks = [];
    #read out the first 10 landmarks
    for i in range(1,10):
        listWithLandmarks.append(lms.LandmarkSet(i))

    # construct a teeth model
    t = th.Teeth(1,listWithLandmarks)

    model = t.PCA()
cv2.destroyAllWindows()