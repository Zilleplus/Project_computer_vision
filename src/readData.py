import numpy as np

# read out the data from the datafiles

pathOfData = "..\\Project_Data\_Data"
pathOfLandmarks = pathOfData + "\\Landmarks"
pathOfXLandmarks = pathOfLandmarks + "\\original"
pathOfYLandmarks = pathOfLandmarks + "\\mirrored"

def ReadLandmark(idLandMark):
    # construct an empy array
    landmark = np.zeros((80,8))

    for rank in range(1,9):
        # open the text file with one collum of data
        fileLocation = pathOfXLandmarks+'\\landmarks'+str(idLandMark)+'-'+str(rank)+'.txt'
        print fileLocation
        f = open(fileLocation, 'r')

        # read out the numbers in the text file
        listWithNumbers = []
        for line in f:
            listWithNumbers.append(float(line))
        f.close()
        # put the collum in the array
        landmark[:,rank-1]= np.array(listWithNumbers)

    # when all collums are done, return the result
    return landmark

if __name__ == '__main__':
    print "running Read Data file : \n"
    print ReadLandmark(1);
