# ------------------------------------------------------------------------
# It can be used to visualize raw data.
# It is useful to regenerate plots used in the project presentation.
# ------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def getData(person, gesture, sensor):
    racine = "./dataset/p"
    dataFile = racine + person + "_" + gesture + ".csv"

    if sensor == "Acc": columns = (1, 2, 3)
    if sensor == "Gyr": columns = (4, 5, 6)
    if sensor == "MagField": columns = (7, 8, 9)
    if sensor == "RotVec": columns = (10, 11, 12)

    dataframe = np.loadtxt(dataFile, delimiter=',', skiprows=1, usecols=columns)  # load data

    x = dataframe[:, 0]
    y = dataframe[:, 1]
    z = dataframe[:, 2]
    print(x.shape)

    return x, y, z


gesture = "Square" # "Circle", "Updown", "Tilt", "Triangle", "Turn", "Square"
sensor = "RotVec"  # "Gyr" # "Acc", "RotVec", "MagField"

x1, y1, z1 = getData("4", gesture, sensor)
x2, y2, z2 = getData("10", gesture, sensor)
x3, y3, z3 = getData("13", gesture, sensor)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x1, y1, z1, label='participant 4', color="c")
ax.plot(x2, y2, z2, label='participant 10', color="m")
ax.plot(x3, y3, z3, label='participant 12', color="y")

ax.set_title(gesture + " ("+sensor+")", loc='center')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

