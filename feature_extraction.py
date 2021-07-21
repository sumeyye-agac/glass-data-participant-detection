# Sumeyye Agac - 2018800039
# CmpE 58Z - Introduction to Biometrics (Spring 2021)
# Project
# ------------------------------------------------------------------------
# Extraction of feature files for each gesture-participant combination
# This code will use the files under ./dataset/ folder to extract 36 features
# and save under ./features/ folder.
# Example: by using ./dataset/p15_Square.csv raw data it will generate ./features/Features_p15_Square.csv
# ------------------------------------------------------------------------
# NOTE: Features are already extracted and saved under ./features folder in the project.
# However, after deleting this folder, it is possible to regenerate the same features by running this code.
# ------------------------------------------------------------------------

import numpy as np
import csv

def addRow(newRow, fd):
    wr = csv.writer(fd, lineterminator='\n')
    wr.writerow(newRow)

def calculateFeaturesofWindow(numberOfFirstRow, dataset, sizeOfWindow):
    iD = ':'.join(map(str, [numberOfFirstRow, (numberOfFirstRow+sizeOfWindow)]))
    AccaxisX = dataset[(numberOfFirstRow):(numberOfFirstRow+sizeOfWindow), 0]
    AccaxisY = dataset[(numberOfFirstRow):(numberOfFirstRow+sizeOfWindow), 1]
    AccaxisZ = dataset[(numberOfFirstRow):(numberOfFirstRow+sizeOfWindow), 2]
    GyraxisX = dataset[(numberOfFirstRow):(numberOfFirstRow+sizeOfWindow), 3]
    GyraxisY = dataset[(numberOfFirstRow):(numberOfFirstRow+sizeOfWindow), 4]
    GyraxisZ = dataset[(numberOfFirstRow):(numberOfFirstRow+sizeOfWindow), 5]
    MagFieldaxisX = dataset[(numberOfFirstRow):(numberOfFirstRow+sizeOfWindow), 6]
    MagFieldaxisY = dataset[(numberOfFirstRow):(numberOfFirstRow+sizeOfWindow), 7]
    MagFieldaxisZ = dataset[(numberOfFirstRow):(numberOfFirstRow+sizeOfWindow), 8]
    RotVecaxisX = dataset[(numberOfFirstRow):(numberOfFirstRow+sizeOfWindow), 9]
    RotVecaxisY = dataset[(numberOfFirstRow):(numberOfFirstRow+sizeOfWindow), 10]
    RotVecaxisZ = dataset[(numberOfFirstRow):(numberOfFirstRow+sizeOfWindow), 11]

    Accmin_X = min(AccaxisX)
    Accmin_Y = min(AccaxisY)
    Accmin_Z = min(AccaxisZ)
    Accmax_X = max(AccaxisX)
    Accmax_Y = max(AccaxisY)
    Accmax_Z = max(AccaxisZ)
    Accmean_X = np.mean(AccaxisX)
    Accmean_Y = np.mean(AccaxisY)
    Accmean_Z = np.mean(AccaxisZ)

    Gyrmin_X = min(GyraxisX)
    Gyrmin_Y = min(GyraxisY)
    Gyrmin_Z = min(GyraxisZ)
    Gyrmax_X = max(GyraxisX)
    Gyrmax_Y = max(GyraxisY)
    Gyrmax_Z = max(GyraxisZ)
    Gyrmean_X = np.mean(GyraxisX)
    Gyrmean_Y = np.mean(GyraxisY)
    Gyrmean_Z = np.mean(GyraxisZ)


    MagFieldmin_X = min(MagFieldaxisX)
    MagFieldmin_Y = min(MagFieldaxisY)
    MagFieldmin_Z = min(MagFieldaxisZ)
    MagFieldmax_X = max(MagFieldaxisX)
    MagFieldmax_Y = max(MagFieldaxisY)
    MagFieldmax_Z = max(MagFieldaxisZ)
    MagFieldmean_X = np.mean(MagFieldaxisX)
    MagFieldmean_Y = np.mean(MagFieldaxisY)
    MagFieldmean_Z = np.mean(MagFieldaxisZ)


    RotVecmin_X = min(RotVecaxisX)
    RotVecmin_Y = min(RotVecaxisY)
    RotVecmin_Z = min(RotVecaxisZ)
    RotVecmax_X = max(RotVecaxisX)
    RotVecmax_Y = max(RotVecaxisY)
    RotVecmax_Z = max(RotVecaxisZ)
    RotVecmean_X = np.mean(RotVecaxisX)
    RotVecmean_Y = np.mean(RotVecaxisY)
    RotVecmean_Z = np.mean(RotVecaxisZ)

    window = [iD,
              Accmin_X, Accmin_Y, Accmin_Z,
              Accmax_X, Accmax_Y, Accmax_Z,
              Accmean_X, Accmean_Y, Accmean_Z, \
              Gyrmin_X, Gyrmin_Y, Gyrmin_Z,
              Gyrmax_X, Gyrmax_Y, Gyrmax_Z,
              Gyrmean_X, Gyrmean_Y, Gyrmean_Z, \
              MagFieldmin_X, MagFieldmin_Y, MagFieldmin_Z,
              MagFieldmax_X, MagFieldmax_Y, MagFieldmax_Z,
              MagFieldmean_X, MagFieldmean_Y, MagFieldmean_Z,
              RotVecmin_X, RotVecmin_Y, RotVecmin_Z,
              RotVecmax_X, RotVecmax_Y, RotVecmax_Z,
              RotVecmean_X, RotVecmean_Y, RotVecmean_Z
              ]
    return window

def createFeatureFile(data, features_path):
    global samplingRate, windowSize

    fd = open(features_path, "a")

    numberOfLine = data.shape[0]
    print("numberOfLine: ", numberOfLine)
    window = samplingRate*windowSize
    sizeOfOverlap = np.floor(window/2)
    sizeOfWindow = sizeOfOverlap*2
    rest1 = numberOfLine % sizeOfWindow
    numberOfWindow1 = np.floor(numberOfLine / sizeOfWindow)

    print("rest1: ", rest1)
    print("numberOfWindow1: ", numberOfWindow1)

    rest2 = (numberOfLine-np.floor(window/2)) % sizeOfWindow
    numberOfWindow2 = np.floor((numberOfLine-np.floor(window/2)) / sizeOfWindow)

    print("rest2: ", rest2)
    print("numberOfWindow2: ", numberOfWindow2)

    numberOfWindow = numberOfWindow1 + numberOfWindow2
    print("numberOfWindow: ", numberOfWindow)


    for i in range(int(numberOfWindow)):
        newWindow = calculateFeaturesofWindow(i*int(sizeOfOverlap), data, int(sizeOfWindow)) # %50 overlapping
        addRow(newWindow, fd)

    fd.close()


samplingRate, windowSize = 55, 1
gesture_list = ["Circle", "Updown", "Tilt", "Triangle", "Turn", "Square"]
participant_no_list = range(1,16)

for gesture in gesture_list:
    input_path = "./dataset/"
    output_path = "./features/"
    for participant_no in participant_no_list:
        data_path = input_path + "p" + str(participant_no) + "_" + gesture + ".csv"
        print("input: ", data_path)
        features_path = output_path + "Features_p" + str(participant_no) + "_" + gesture + ".csv"
        data = np.loadtxt(data_path, delimiter=",", usecols=range(1,13))
        createFeatureFile(data, features_path)
        print("output: ", features_path)

