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

    feature_importance = list(model.feature_importances_)

    print("RF-501, " + gesture + ", " + str(selected_features) + ", "
          + str(participant) + ", {:.2f}" .format(acc) + ", {:.2f}" .format(auc)
          + ", " + str(tp + fn) + ", " + str(fp + tn) + ", "
          + str(tn) + ", " + str(fp) + ", " + str(fn) + ", " + str(tp)
          + ", {:.2f}".format(far) + ", {:.2f}" .format(frr) + ", " + "{:.2f}" .format(eer)
          )#+ ", " + str(feature_importance)[1:-1])
		  
def loadData(gesture, participant, selected_features):
    for p in range(1, 16):
        path = "./features/"+ "Features_p"\
               + str(p) + "_" + gesture + ".csv"

        # iD, Accmin_X, Accmin_Y, Accmin_Z,  0              1 2 3
        # Accmax_X, Accmax_Y, Accmax_Z,                     4 5 6
        # Accmean_X, Accmean_Y, Accmean_Z,                  7 8 9
        # Gyrmin_X, Gyrmin_Y, Gyrmin_Z,                     10 11 12
        # Gyrmax_X, Gyrmax_Y, Gyrmax_Z,,                    13 14 15
        # Gyrmean_X, Gyrmean_Y, Gyrmean_Z,                  16 17 18
        # MagFieldmin_X, MagFieldmin_Y, MagFieldmin_Z,,     19 20 21
        # MagFieldmax_X, MagFieldmax_Y, MagFieldmax_Z,,     22 23 24
        # MagFieldmean_X, MagFieldmean_Y, MagFieldmean_Z,   25 26 27
        # RotVecmin_X, RotVecmin_Y, RotVecmin_Z,            28 29 30
        # RotVecmax_X, RotVecmax_Y, RotVecmax_Z,            31 32 33
        # RotVecmean_X, RotVecmean_Y, RotVecmean_Z,         34 35 36


        columns_circle_all = [29, 30, 24, 23, 22, 32, 12, 31, 35, 20, 14, 11, 21, 4, 8,
                               26, 15, 33, 2, 28, 19, 5, 27, 3, 25, 1, 18, 17, 35, 34, 13, 10, 6, 7, 9, 16]
        columns_updown_all = [22, 25, 19, 32, 20, 33, 30, 35, 23, 24, 35, 29, 4, 21, 28,
                               26, 31, 8, 1, 7, 6, 13, 5, 2, 27, 3, 10, 34, 12, 9, 15, 14, 11, 17, 18, 16]
        columns_tilt_all = [27, 20, 24, 21, 29, 23, 26, 30, 28, 31, 32, 6, 9, 34, 33, 19,
                             2, 3, 35, 8, 15, 22, 35, 14, 4, 12, 25, 1, 11, 10, 5, 13, 17, 7, 16, 18]
        columns_triangle_all = [22, 20, 31, 29, 23, 25, 32, 24, 19, 30, 21, 35, 26, 28, 34,
                                 27, 14, 4, 33, 12, 1, 6, 15, 35, 11, 2, 7, 5, 3, 18, 13, 8, 17, 10, 9, 16]
        columns_turn_all = [20, 26, 23, 21, 24, 29, 22, 31, 3, 8, 9, 28, 32, 19, 4, 33, 27,
                             34, 30, 6, 1, 25, 14, 2, 11, 15, 5, 12, 7, 35, 35, 13, 10, 18, 16, 17]
        columns_square_all = [22, 19, 30, 23, 24, 29, 32, 33, 20, 31, 21, 28, 25, 1, 18, 26,
                               4, 27, 12, 15, 35, 35, 5, 34, 7, 8, 2, 3, 14, 6, 11, 17, 13, 16, 10, 9]

        if selected_features == 36:
            new_x = np.loadtxt(path, delimiter=",", usecols=range(1, 37))

        else:
            columns_circle = tuple(columns_circle_all[:selected_features])
            columns_updown = tuple(columns_updown_all[:selected_features])
            columns_tilt = tuple(columns_tilt_all[:selected_features])
            columns_triangle = tuple(columns_triangle_all[:selected_features])
            columns_turn = tuple(columns_turn_all[:selected_features])
            columns_square = tuple(columns_square_all[:selected_features])

            if gesture == "Circle":
                new_x = np.loadtxt(path, delimiter=",", usecols=columns_circle)
            if gesture == "Updown":
                new_x = np.loadtxt(path, delimiter=",", usecols=columns_updown)
            if gesture == "Tilt":
                new_x = np.loadtxt(path, delimiter=",", usecols=columns_tilt)
            if gesture == "Triangle":
                new_x = np.loadtxt(path, delimiter=",", usecols=columns_triangle)
            if gesture == "Turn":
                new_x = np.loadtxt(path, delimiter=",", usecols=columns_turn)
            if gesture == "Square":
                new_x = np.loadtxt(path, delimiter=",", usecols=columns_square)

        if selected_features == 1:
            new_x = np.array([new_x]).T # convert rom row array to column array

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
selected_features_list = range(1,37)
log = True
if log == True:
    sys.stdout = open("results/results_1to36_feature_importances.txt", "w")

print("ALGORITHM, GESTURE, # OF SELECTED FEATURES, PARTICIPANT_NO, ACCURACY, AUC, #ofPositiveInstance, #ofNegativeInstance,"
      " TN, FP, FN, TP, FAR, FRR, EER") #, FEATURE_IMPORTANCE")

for selected_features in selected_features_list:
    for gesture in gestures:
        for participant in participants:

            x_data, y_data = loadData(gesture, participant, selected_features)

            x_train, x_test, y_train, final_y_test = \
                train_test_split(x_data, y_data, test_size=0.30, random_state=42, shuffle=True)

            clfRF = RandomForestClassifier(n_estimators=501)

            clfRF.fit(x_train, y_train)
            final_y_pred_RF = clfRF.predict(x_test)

            printAll(final_y_test, final_y_pred_RF, gesture, participant, clfRF, selected_features)

if log == True:
    sys.stdout.close()