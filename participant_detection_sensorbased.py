# ------------------------------------------------------------------------
# If features are extracted (use feature_extraction.py to extract if needed.)
# it will generate the performance results for each gesture-participant combination
# by using best 5, 10, 15 and 20 features and 36 features.
#
# By setting log to True (log=True) results can be saved in a csv file
# and be investigated in a more detailed way further.
#
# By changing gestures, participants, selected_features_list variables we realize a specific experiment
# instead of all which is the default setting.
# ------------------------------------------------------------------------

import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from scipy.optimize import brentq
from sklearn.metrics import accuracy_score
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sys


def printAll(expected, predicted, gesture, participant, model, selected_features):
    auc = roc_auc_score(expected, predicted)
    fpr, tpr, thresholds = roc_curve(expected, predicted, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    acc = accuracy_score(expected, predicted)
    tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()

    far = fp / (fp + tn)
    frr = fn / (fn + tp)
    tpr = tp / (tp + fn)

    print("RF-501, " + gesture + ", " + str(selected_features) + ", "
          + str(participant) + ", {:.2f}" .format(acc) + ", {:.2f}" .format(auc)
          + ", " + str(tp + fn) + ", " + str(fp + tn) + ", "
          + str(tn) + ", " + str(fp) + ", " + str(fn) + ", " + str(tp)
          + ", {:.2f}".format(far) + ", {:.2f}" .format(frr) + ", " + "{:.2f}" .format(eer))
		  
def loadData(gesture, participant, selected_features, overlapping):
    for p in range(1, 16):

        if overlapping == True:
            path = "./features_w_overlapping/"+ "Features_p"\
                   + str(p) + "_" + gesture + ".csv"
        else:
            path = "./features_wo_overlapping/" + "Features_p" \
                    + str(p) + "_" + gesture + ".csv"

        # iD, Accmin_X, Accmin_Y, Accmin_Z,  0              1 2 3
        # Accmax_X, Accmax_Y, Accmax_Z,                     4 5 6
        # Accmean_X, Accmean_Y, Accmean_Z,                  7 8 9
        # Gyrmin_X, Gyrmin_Y, Gyrmin_Z,                     10 11 12
        # Gyrmax_X, Gyrmax_Y, Gyrmax_Z,,                    13 14 15
        # Gyrmean_X, Gyrmean_Y, Gyrmean_Z,                  16 17 18
        # MagFieldmin_X, MagFieldmin_Y, MagFieldmin_Z,      19 20 21
        # MagFieldmax_X, MagFieldmax_Y, MagFieldmax_Z,      22 23 24
        # MagFieldmean_X, MagFieldmean_Y, MagFieldmean_Z,   25 26 27
        # RotVecmin_X, RotVecmin_Y, RotVecmin_Z,            28 29 30
        # RotVecmax_X, RotVecmax_Y, RotVecmax_Z,            31 32 33
        # RotVecmean_X, RotVecmean_Y, RotVecmean_Z,         34 35 36

        acc_features = list(range(1, 10))
        gyr_features = list(range(10, 19))
        magfield_features = list(range(19, 28))
        rotvec_features = list(range(28, 37))

        if selected_features == "Acc":
            columns = acc_features
        if selected_features == "Gyr":
            columns = gyr_features
        if selected_features == "RotVec":
            columns = rotvec_features
        if selected_features == "MagField":
            columns = magfield_features
        if selected_features == "AccGyr":
            columns = acc_features + gyr_features
        if selected_features == "AccRotVec":
            columns = acc_features + rotvec_features
        if selected_features == "AccMagField":
            columns = acc_features + magfield_features
        if selected_features == "GyrRotVec":
            columns = gyr_features + rotvec_features
        if selected_features == "GyrMagField":
            columns = gyr_features + magfield_features
        if selected_features == "RotVecMagField":
            columns = rotvec_features + magfield_features
        if selected_features == "AccGyrRotVec":
            columns = acc_features + gyr_features + rotvec_features
        if selected_features == "AccGyrMagField":
            columns = acc_features + gyr_features + magfield_features
        if selected_features == "GyrRotVecMagField":
            columns = gyr_features + rotvec_features + magfield_features
        if selected_features == "AccRotVecMagField":
            columns = acc_features + rotvec_features + magfield_features
        if selected_features == "AccGyrRotVecMagField":
            columns = acc_features + gyr_features + rotvec_features + magfield_features

        new_x = np.loadtxt(path, delimiter=",", usecols=tuple(columns))

        n, _ = new_x.shape

        if p == participant:
            y_ = np.ones((n, 1))
        else:
            y_ = np.zeros((n, 1))
        new_x = np.hstack((new_x, y_))

        if p == 1:
            x = new_x
        else:
            x = np.concatenate((x, new_x), axis=0)

    y = x[:, -1]
    x = x[:, :-1]

    return x, y

gestures = ["Circle", "Updown", "Tilt", "Triangle", "Turn", "Square"]

participants = range(1,16)

selected_sensors = ["Acc", "Gyr", "RotVec", "MagField",
                    "AccGyr", "AccRotVec", "AccMagField", "GyrRotVec", "GyrMagField",
                    "RotVecMagField", "AccGyrRotVec", "AccGyrMagField", "GyrRotVecMagField",
                    "AccRotVecMagField", "AccGyrRotVecMagField"]

log = True
overlapping = False

if log == True:
    sys.stdout = open("results/results_sensor_based_wo_overlapping.txt", "w")

print("ALGORITHM, GESTURE, # OF SELECTED SENSORS, PARTICIPANT_NO, ACCURACY, AUC, #ofPositiveInstance, #ofNegativeInstance,"
      " TN, FP, FN, TP, FAR, FRR, EER")

for selected_features in selected_sensors:
    for gesture in gestures:
        for participant in participants:

            x_data, y_data = loadData(gesture, participant, selected_features, overlapping)

            x_train, x_test, y_train, final_y_test = \
                train_test_split(x_data, y_data, test_size=0.30, random_state=42, shuffle=True)

            clfRF = RandomForestClassifier(n_estimators=501)

            clfRF.fit(x_train, y_train)
            final_y_pred_RF = clfRF.predict(x_test)

            printAll(final_y_test, final_y_pred_RF, gesture, participant, clfRF, selected_features)

if log == True:
    sys.stdout.close()