import numpy as np
import sys
from imblearn.over_sampling._smote import SMOTE
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier


def printAll(expected, predicted, gesture, selected_features):
    print("-" * 50)
    print(gesture + " " + selected_features)
    results = metrics.classification_report(expected, predicted)
    print(results)
    print(metrics.confusion_matrix(expected, predicted))



def loadData(gesture, selected_features, overlapping):

    for participant in range(1, 16):
        if overlapping == True:
            path = "./features_w_overlapping/" + "Features_p" \
                   + str(participant) + "_" + gesture + ".csv"
        else:
            path = "./features_wo_overlapping/" + "Features_p" \
                   + str(participant) + "_" + gesture + ".csv"

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

        y_ = np.ones((n, 1))
        y_[:,:] = participant

        new_x = np.hstack((new_x, y_))

        if participant == 1:
            x = new_x
        else:
            x = np.concatenate((x, new_x), axis=0)

    y = x[:, -1]
    x = x[:, :-1]

    return x, y


gestures = ["Circle", "Updown", "Tilt", "Triangle", "Turn", "Square"]

selected_sensors = ["Acc", "Gyr", "RotVec", "MagField",
                    "AccGyr", "AccRotVec", "AccMagField", "GyrRotVec", "GyrMagField",
                    "RotVecMagField", "AccGyrRotVec", "AccGyrMagField", "GyrRotVecMagField",
                    "AccRotVecMagField", "AccGyrRotVecMagField"]


log = True
overlapping = True

smote = True
validation = "CV10"
seed = 42

if log == True:
    sys.stdout = open("results/results_gesture_recognition_w_overlapping_w_smote_CV10_Adaboost_501.txt", "w")

for selected_features in selected_sensors:
    for gesture in gestures:

        x_data, y_data = loadData(gesture, selected_features, overlapping)

        kf = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=seed) 
        final_y_test, final_y_pred = [], []

        for train_index, test_index, in kf.split(x_data, y_data):
            x_train = x_data[train_index]
            y_train = y_data[train_index]
            x_test = x_data[test_index]
            y_test = y_data[test_index]
            final_y_test = np.concatenate([final_y_test, y_test])

            if smote == True:
                sm = SMOTE(sampling_strategy='auto', k_neighbors=2, random_state=42)
                x_train, y_train = sm.fit_resample(x_train, y_train)

            clf = AdaBoostClassifier(n_estimators=501, random_state=seed)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            final_y_pred = np.concatenate([final_y_pred, y_pred])

        printAll(final_y_test, final_y_pred, gesture, selected_features)

if log == True:
    sys.stdout.close()