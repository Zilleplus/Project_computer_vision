# view the landmarks onto the pictures

# used libs
import cv2
import cv2.cv as cv
import numpy as np

# self made libs
import readData as rd
import Teeth as th
import ActiveShapeModel as asm
import math
import sys

import LandMarkSet as lms
import fitAlgo as falgo


# to help functions
def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)
def normLandmarkModel(model,landmark,rank_landmark):
    norm =0
    for i in range(0,40):
        coor_landmark = landmark.getCoordinates(i,rank_landmark)
        coor_model = model.getCoordinatesModel(i)
        norm +=  ( coor_landmark[0]-coor_model[0] )**2 + ( coor_landmark[1]-coor_model[1] )**2
    return norm

# define what demo's should be run
demo1 = False
demo2 = False
demo3 = False
demo3b = False
demo4 = False
demo5 = False
demo6 = False
demo7 = True
demo8 = False
demo9 = False

if demo1 == True:
    test = lms.LandmarkSet(1)
    img = rd.readRadiograph(1)

    for teeth in range(0, 8):
        for i in range(0, 20):
            cv2.line(img, test.getCoordinates(i * 2, teeth), test.getCoordinates(i * 2 + 1, teeth), 100, 5)

    img = img[720:1280, 1280:1750]
    small = img.copy()
    small = cv2.resize(small, (700, 500))



    cv2.imshow('small', small)
    cv2.waitKey(0)

# display the normalized landmarks of each picture
if demo2 == True:
    # do procus... anal... -------------
    listWithLandmarks = [];
    # read out the first 10 landmarks
    for i in range(1, 11):
        listWithLandmarks.append(lms.LandmarkSet(i))

    # construct a teeth model
    t = th.Teeth(1, listWithLandmarks)

    img = rd.readRadiograph(1)
    # draw line between 2 points to get a striped version of the model
    for teeth in range(0, 8):
        for i in range(0, 20):
            # print "="+str(t.getNormalizedCoordinatesModelIndividualTeeth(teeth,i*2))
            cv2.line(img, t.getNormalizedCoordinatesModelIndividualTeeth(teeth, i * 2),
                     t.getNormalizedCoordinatesModelIndividualTeeth(teeth, i * 2 + 1), 100, 5)

        small = img.copy()
        small = cv2.resize(small, (700, 500))

        cv2.imshow('small', small)
        cv2.waitKey(0)

# display an active shape model
if demo3 == True:
    listWithLandmarks = [];
    # read out the first 10 landmarks
    for i in range(2, 29):
        listWithLandmarks.append(lms.LandmarkSet(i))

    # construct a teeth model
    t = th.Teeth(1, listWithLandmarks)

    # do the PCA analysis and get the model
    model = t.PCA(28)  # activeshapemodel object

    print  model.eigenvalues

    # read out the image
    img = np.ones((500,1200,3), np.uint8)*255

    model.scale = .2
    model.x_offset = 80
    model.y_offset = 230

    for i in range(0, 40):
        firstPoint = model.getCoordinatesModel(i)
        secondPoint = model.getCoordinatesModel((i + 1) % 40)
        cv2.line(img, firstPoint, secondPoint, 100, 2)

    model.scale = 40
    model.x_offset = 200
    model.y_offset = 230

    weights=np.zeros(27)
    weights[0]=1
    model.setShape(weights)

    for i in range(0, 40):
        firstPoint = model.getCoordinatesModel(i)
        secondPoint = model.getCoordinatesModel((i + 1) % 40)
        cv2.line(img, firstPoint, secondPoint, 100, 2)

    for i in range(1,10):
        model.x_offset=model.x_offset+100
        weights[i-1]=0
        weights[i]=1
        model.setShape(weights)

        for i in range(0, 40):
            firstPoint = model.getCoordinatesModel(i)
            secondPoint = model.getCoordinatesModel((i + 1) % 40)
            cv2.line(img, firstPoint, secondPoint, 100, 2)


    cv2.imshow('img', img)
    cv2.waitKey(0)

if demo3b == True:
    listWithLandmarks = [];
    # read out the first 10 landmarks
    for i in range(2, 29):
        listWithLandmarks.append(lms.LandmarkSet(i))

    # construct a teeth model
    t = th.Teeth(1, listWithLandmarks)

    # do the PCA analysis and get the model
    model = t.PCA(28)  # activeshapemodel object

    print  model.eigenvalues

    # read out the image
    img = np.ones((500,1200,3), np.uint8)*255

    model.scale = 40
    model.x_offset = 80
    model.y_offset = 230

    weights=np.zeros(27)
    weights[0]=1
    model.setShape(weights)

    for i in range(0, 40):
        firstPoint = model.getCoordinatesModel(i)
        secondPoint = model.getCoordinatesModel((i + 1) % 40)
        cv2.line(img, firstPoint, secondPoint, 100, 2)

    for i in range(1,10):
        model.x_offset=model.x_offset+100
        weights[i]=1
        model.setShape(weights)

        for i in range(0, 40):
            firstPoint = model.getCoordinatesModel(i)
            secondPoint = model.getCoordinatesModel((i + 1) % 40)
            cv2.line(img, firstPoint, secondPoint, 100, 2)


    cv2.imshow('img', img)
    cv2.waitKey(0)

if demo4 == True:
    # read out the image
    img = rd.readRadiograph(9)

    blur = img.copy()
    blur = cv2.GaussianBlur(img, (3, 3), 1)
    blur = cv2.medianBlur(blur, 5)
    small_blur = cv2.resize(blur, (900, 700))

    edges = cv2.Canny(blur, 15, 40)

    small_edge = edges.copy()
    small_edge = cv2.resize(small_edge, (900, 700))

    small = img.copy()
    small = cv2.resize(small, (900, 700))

    cv2.imshow('small', small)
    cv2.waitKey(0)
    cv2.imshow('small_blur', small_blur)
    cv2.waitKey(0)
    cv2.imshow('edge', small_edge)
    cv2.waitKey(0)
# zoom in on one tooth
if demo5 == True:
    # read out the image
    img = rd.readRadiograph(1)

    blur = img.copy()
    blur = cv2.GaussianBlur(img, (3, 3), 1)
    blur = cv2.medianBlur(blur, 5)

    edges = cv2.Canny(blur, 15, 40)

    # zoom in
    edges = edges[750:1200, 1300:1750]

    small_edge = edges.copy()
    small_edge = cv2.resize(small_edge, (700, 500))

    cv2.imshow('edge', small_edge)
    cv2.waitKey(0)

if demo6 == True:
    # read out the image
    img = rd.readRadiograph(1)

    blur = img.copy()
    blur = cv2.GaussianBlur(img, (3, 3), 1)
    blur = cv2.medianBlur(blur, 5)

    edges = cv2.Canny(blur, 15, 40)

    listWithLandmarks = [];
    # read out the first 10 landmarks
    for i in range(1, 11):
        listWithLandmarks.append(lms.LandmarkSet(i))

    # construct a teeth model
    t = th.Teeth(0, listWithLandmarks)

    # do the PCA analysis and get the model
    model = t.PCA(28)  # activeshapemodel object

    # zoom in
    edges = edges[750:1010, 1300:1400]

    # draw the model on the picture
    model.scale = 1
    model.x_offset = 60
    model.y_offset = 160

    model.rotate(np.pi)
    model.scaleUp(2)

    # map the model from the middle out
    centrum = model.getCentrum()

    number_of_points = 40
    for i in range(0,40):
        growVec = model.getGrowVector(i,10)
        cv2.line(edges, model.getCoordinatesModel(i), growVec, 1500, 1)

    # print the untouched model
    for i in range(0, 40):
        firstPoint = model.getCoordinatesModel(i)
        secondPoint = model.getCoordinatesModel((i + 1) % 40)
        cv2.line(edges, firstPoint, secondPoint, 1000, 2)

    # display everything
    small_edge = edges.copy()
    small_edge = cv2.resize(small_edge, (700, 500))

    cv2.imshow('edge', small_edge)
    cv2.waitKey(0)

# destroy windows of all demo's
cv2.destroyAllWindows()
#
if demo7 == True:
    # read out the image
    img = rd.readRadiograph(1)

    blur = img.copy()
    blur = cv2.GaussianBlur(img, (3, 3), 1)
    blur = cv2.medianBlur(blur, 5)

    edges = cv2.Canny(blur, 15, 40)

    listWithLandmarks = [];
    # read out the first 10 landmarks
    for i in range(2, 11):
        listWithLandmarks.append(lms.LandmarkSet(i))

    # construct a teeth model
    t = th.Teeth(0, listWithLandmarks)

    # do the PCA analysis and get the model
    model = t.PCA(28)  # activeshapemodel object

    # zoom in
    edges = edges[750:1010, 1300:1400]

    # draw the model on the picture
    model.scale = 1
    model.x_offset = 60
    model.y_offset = 155

    model.scaleUp(3.4)
    model.rotate(np.pi+0.1)

    number_of_points = 40
    newModel = model

    scales = np.zeros((number_of_points))

    # find the first points to work from
    max_opti = 100
    stepsize = 0.5
    print "level 0"
    scales = falgo.findNewRefPoints(edges, scales, model, False, max_opti, stepsize, range_weights=5, tresh_hold=250)

    # prepare the data save it in an array
    refpoints = []
    for i in range(0, 40):
        # draw the line
        refpoints.append(newModel.getGrowVector(i, scales[i]))

    # send it to the model
    newModel = newModel.calcNewModel(refpoints)

    print "level 1"
    max_opti = 1000
    stepsize = 0.1
    # 2 cyclis
    for loop in range(0,2):
        for i in range(0, number_of_points):
            scales[i] = newModel.getGrowVectorStartScale(i)

        scales = falgo.findNewRefPoints(edges, scales, newModel, True, max_opti, stepsize, 3,300)

        # prepare the data save it in an list
        refpoints = []
        for i in range(0, 40):
            refpoints.append(newModel.getGrowVector(i, scales[i]))

        # send it to the model
        newModel = newModel.calcNewModel(refpoints)

    print "level 2"
    max_opti = 50
    stepsize = 0.1
    # 4 cyclis
    for loop in range(0,0):
        for i in range(0, number_of_points):
            scales[i] = newModel.getGrowVectorStartScale(i)

        scales = falgo.findNewRefPoints(edges, scales, newModel, True, max_opti, stepsize, 1,300)

        # prepare the data save it in an list
        refpoints = []
        for i in range(0, 40):
            refpoints.append(newModel.getGrowVector(i, scales[i]))

        # send it to the model
        newModel = newModel.calcNewModel(refpoints)

    # print the untouched model
    for i in range(0, 40):
        firstPoint = model.getCoordinatesModel(i)
        secondPoint = model.getCoordinatesModel((i + 1) % 40)
        cv2.line(edges, firstPoint, secondPoint, 1000, 2)

    # print out the new model
    for i in range(0, 40):
        firstPoint = newModel.getCoordinatesModel(i)
        secondPoint = newModel.getCoordinatesModel((i + 1) % 40)
        cv2.line(edges, firstPoint, secondPoint, 100, 2)

    # display everything
    small_edge = edges.copy()
    small_edge = cv2.resize(small_edge, (700, 500))

    cv2.imshow('edge', small_edge)
    cv2.waitKey(0)

if demo8 == True:
    # read out the image
    img = rd.readRadiograph(1)

    blur = img.copy()
    blur = cv2.GaussianBlur(img, (3, 3), 1)
    blur = cv2.medianBlur(blur, 5)

    edges = cv2.Canny(blur, 15, 40)

    listWithLandmarks = []
    # read out the landmarks
    for i in range(2, 28):
        listWithLandmarks.append(lms.LandmarkSet(i))

    currentLandmark = listWithLandmarks[2]
    currentLandmark.setData( lms.getAveragePositionsLandmarks(listWithLandmarks[1:10]) )

    # construct a teeth model
    t = th.Teeth(0, listWithLandmarks)

    # do the PCA analysis and get the model
    model = t.PCA(28)  # activeshapemodel object

    # fit the model on this
    # do this for 30 different rotations of the model

    refpoints_all = lms.reshapeLandmark(currentLandmark)

    for rank in range(0,8):
        refpoints = refpoints_all[:,rank]
        # convert to a list of tuples
        list_of_ref_points = []
        for i in range(0,40):
            x_coor = refpoints[i]
            y_coor = refpoints[i+40]
            list_of_ref_points.append((x_coor,y_coor))

        (x_center_landmark, y_center_landmark) = currentLandmark.getCenter(rank)
        w = currentLandmark.getWidth(rank)

        model.x_offset = x_center_landmark
        model.y_offset = y_center_landmark

        newModel = model.calcNewModel(list_of_ref_points)
        (x_center_newModel, y_center_newModel) = newModel.getCentrum()

        # optimize the location
        start_x = newModel.x_offset
        start_y = newModel.y_offset

        newModel.x_offset = newModel.x_offset
        print normLandmarkModel(newModel,currentLandmark,rank)

        best_score= sys.float_info.max
        x_offset_answer=0
        y_offset_answer=0
        range_seek =100

        for x_d in range(0,range_seek):
            x_d = int(x_d - x_d/2)
            for y_d in range(0,range_seek):
                y_d = int(y_d - y_d/2)
                # minimize the norm of the difference between average landmark and our model
                newModel.x_offset = start_x + x_d
                newModel.y_offset = start_y + x_d

                score_local = normLandmarkModel(newModel,currentLandmark,rank)
                if score_local<best_score :
                    best_score=score_local

                    x_offset_answer=x_d
                    y_offset_answer=y_d


        newModel.x_offset = start_x + x_offset_answer
        newModel.y_offset = start_y  + y_offset_answer

        for i in range(0, 40):
            firstPoint = newModel.getCoordinatesModel(i)
            secondPoint = newModel.getCoordinatesModel((i+1) % 40)

            cv2.line(edges, firstPoint, secondPoint, 100, 3)

        # for i in range(0, 40):
        #     firstPoint = currentLandmark.getCoordinates(i,rank)
        #     secondPoint = currentLandmark.getCoordinates((i+1) % 40,rank)
        #
        #     cv2.line(edges, firstPoint, secondPoint, 100, 3)

        # print normLandmarkModel(newModel,currentLandmark,rank)
    # zoom in
    edges = edges[760:1250, 1300:1750]

    # display everything
    small_edge = edges.copy()
    small_edge = cv2.resize(small_edge, (700, 500))

    cv2.imshow('edge', small_edge)
    cv2.waitKey(0)

if demo9 == True:
    # read out the image 1 or 9
    img = rd.readRadiograph(1)

    blur = img.copy()
    blur = cv2.GaussianBlur(img, (3, 3), 1)
    blur = cv2.medianBlur(blur, 5)

    edges = cv2.Canny(blur, 15, 40)

    listWithLandmarks = [];
    # read out the first 10 landmarks
    for i in range(2, 11):
        listWithLandmarks.append(lms.LandmarkSet(i))

    # construct a teeth model
    t = th.Teeth(0, listWithLandmarks)
    rank_landmark =1

    # do the PCA analysis and get the model
    model = t.PCA(28)  # activeshapemodel object

    # zoom in
    edges = edges[750:1010, 1300:1400]
    landmarkoffset_x = 1300
    landmarkoffset_y = 750
    # edges = edges[650:1200, 1350:1500]

    # draw the model on the picture
    model.scale = 1
    model.x_offset = 60
    model.y_offset = 155
    # model.x_offset = 75
    # model.y_offset = 285
    # scale 3.4
    model.scaleUp(3.4)
    model.rotate(np.pi+0.1)

    number_of_points = 40
    newModel = model

    scales = np.zeros((number_of_points))

    # find the first points to work from
    max_opti = 100
    stepsize = 0.5
    print "level 0"
    scales = falgo.findNewRefPoints(edges, scales, model, False, max_opti, stepsize, range_weights=5, tresh_hold=250)

    # prepare the data save it in an array
    refpoints = []
    for i in range(0, 40):
        # draw the line
        refpoints.append(newModel.getGrowVector(i, scales[i]))

    # send it to the model
    newModel = newModel.calcNewModel(refpoints)
    print "the new norm is:"
    print normLandmarkModel(newModel,lms.LandmarkSet(rank_landmark),rank_landmark)

    print "level 1"
    max_opti = 1000
    stepsize = 0.1
    # 2 cyclis
    for loop in range(0,2):
        for i in range(0, number_of_points):
            scales[i] = newModel.getGrowVectorStartScale(i)

        scales = falgo.findNewRefPoints(edges, scales, newModel, True, max_opti, stepsize, 3,300)

        # prepare the data save it in an list
        refpoints = []
        for i in range(0, 40):
            refpoints.append(newModel.getGrowVector(i, scales[i]))

        # send it to the model
        newModel = newModel.calcNewModel(refpoints)
    print "the new norm is:"
    print normLandmarkModel(newModel,lms.LandmarkSet(rank_landmark),rank_landmark)

    print "level 2"
    max_opti = 50
    stepsize = 0.1
    # 4 cyclis
    for loop in range(0,4):
        for i in range(0, number_of_points):
            scales[i] = newModel.getGrowVectorStartScale(i)

        scales = falgo.findNewRefPoints(edges, scales, newModel, True, max_opti, stepsize, 1,300)

        # prepare the data save it in an list
        refpoints = []
        for i in range(0, 40):
            refpoints.append(newModel.getGrowVector(i, scales[i]))

        # send it to the model
        newModel = newModel.calcNewModel(refpoints)
    print "the new norm is:"
    print normLandmarkModel(newModel,lms.LandmarkSet(rank_landmark),rank_landmark)

    # rotation optimilisation:
    print "optimizing angle"
    max_opti = 2
    stepsize = (2*np.pi)/360

    score = falgo.calc_weight_model(edges,model,1)
    best_model =newModel;
    for i in range(0,max_opti*2):
        angle = 0.01*(-i + i/2)

        # rotate the mode
        rModel = newModel.getNewRotatedModel(angle)

        print "optimizing rotation"
        #optimize the model a bit
        for loop in range(0,2):
            max_opti = 50
            stepsize = 0.1
            for i in range(0, number_of_points):
                scales[i] = rModel.getGrowVectorStartScale(i)

            scales = falgo.findNewRefPoints(edges, scales, rModel, True, max_opti, stepsize, 1,300)

            # prepare the data save it in an list
            refpoints = []
            for i in range(0, 40):
                refpoints.append(newModel.getGrowVector(i, scales[i]))

            # send it to the model
            rModel = rModel.calcNewModel(refpoints)


        # get the score for this rotation
        new_score = falgo.calc_weight_model(edges,rModel,1)

        # if we get a higher score then save this angel, this is a better fit
        if(new_score>score):
            print "found an better angle"
            score = new_score
            best_model=rModel
    newModel=best_model

    print "the new norm is:"
    print normLandmarkModel(newModel,lms.LandmarkSet(rank_landmark),rank_landmark)


    print "searching for new position"
    max_opti = 20
    stepsize = 0.1

    score = falgo.calc_weight_model(edges,model,1)
    best_model =newModel;
    for i_shift in range(0,2):
        for j_shift in range(0,2):
            print i_shift
            print j_shift
            print "---"
            x_shift = -i_shift + max_opti/2
            y_shift = -j_shift + max_opti/2
            if (x_shift==0 and y_shift==0):
                break
            # rotate the mode
            rModel = newModel.getNewShiftedModel(x_shift,y_shift)

            #optimize the model a bit
            for loop in range(0,4):
                max_opti = 50
                stepsize = 0.1
                for i in range(0, number_of_points):
                    scales[i] = rModel.getGrowVectorStartScale(i)

                scales = falgo.findNewRefPoints(edges, scales, rModel, True, max_opti, stepsize, 1,300)

                # prepare the data save it in an list
                refpoints = []
                for i in range(0, 40):
                    refpoints.append(newModel.getGrowVector(i, scales[i]))

                # send it to the model
                rModel = rModel.calcNewModel(refpoints)


            # get the score for this rotation
            new_score = falgo.calc_weight_model(edges,rModel,1)

            # if we get a higher score then save this angel, this is a better fit
            if(new_score>score):
                print "found an better position"
                score = new_score
                best_model=rModel
    newModel=best_model


    # print the untouched model
    for i in range(0, 40):
        firstPoint = model.getCoordinatesModel(i)
        secondPoint = model.getCoordinatesModel((i + 1) % 40)
        cv2.line(edges, firstPoint, secondPoint, 1000, 2)

    # print out the new model
    for i in range(0, 40):
        firstPoint = newModel.getCoordinatesModel(i)
        secondPoint = newModel.getCoordinatesModel((i + 1) % 40)
        cv2.line(edges, firstPoint, secondPoint, 100, 2)


    # display everything
    small_edge = edges.copy()
    small_edge = cv2.resize(small_edge, (700, 500))

    cv2.imshow('edge', small_edge)
    cv2.waitKey(0)

    img = np.zeros((2000,2000,3), np.uint8)
    # draw the landmarks
    for i in range(0,40):
        mark = lms.LandmarkSet(1).getCoordinates(i,0)
        mark_x = mark[0]
        mark_y = mark[1]
        img[mark_y,mark_x] = 255
    # draw the model
    for i in range(0,40):
        point = newModel.getCoordinatesModel(i)
        img[point[1]+landmarkoffset_y,point[0]+ landmarkoffset_x] = 255

    img = img[700:1070, 1300:1450]
    small_img = img.copy()
    small_img = cv2.resize(small_img, (700, 500))
    cv2.imshow('img', small_img)
    cv2.waitKey(0)
