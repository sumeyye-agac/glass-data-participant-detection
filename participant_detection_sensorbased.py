

import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from scipy.optimize import brentq
from sklearn.metrics import accuracy_score
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestClassifier
import sys
from imblearn.over_sampling._smote import SMOTE
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier


def printAll(classifier, expected, predicted, gesture, participant, model, selected_features):
    auc = roc_auc_score(expected, predicted)
    fpr, tpr, thresholds = roc_curve(expected, predicted, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    acc = accuracy_score(expected, predicted)
    tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()

    far = fp / (fp + tn)
    frr = fn / (fn + tp)
    tpr = tp / (tp + fn)

    print(classifier + ", " + gesture + ", " + str(selected_features) + ", "
          + str(participant) + ", {:.2f}".format(acc) + ", {:.2f}".format(auc)
          + ", " + str(tp + fn) + ", " + str(fp + tn) + ", "
          + str(tn) + ", " + str(fp) + ", " + str(fn) + ", " + str(tp)
          + ", {:.2f}".format(far) + ", {:.2f}".format(frr) + ", " + "{:.2f}".format(eer))


def loadData(gesture, participant, selected_features, overlapping):
    for p in range(1, 16):

        if overlapping == True:
            path = "./features_w_overlapping/" + "Features_p" \
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

participants = range(1, 16)

selected_sensors = ["Acc", "Gyr", "RotVec", "MagField",
                    "AccGyr", "AccRotVec", "AccMagField", "GyrRotVec", "GyrMagField",
                    "RotVecMagField", "AccGyrRotVec", "AccGyrMagField", "GyrRotVecMagField",
                    "AccRotVecMagField", "AccGyrRotVecMagField"]

log = True

overlapping = False
smote = True
seed = 42

validation = "CV"
k = 10

classifiers = ["RandomForest",
               "AdaBoost",
               "MLP",
               "SVM_rbf",
               "SVM_poly",
               "SVM_rbf+SVM_poly",
               "MLP+SVM_rbf",
               "MLP+SVM_poly"]

classifiers = ["RandomForest_101", "AdaBoost_101"]

classifiers = ["AdaBoost_501"]

for classifier in classifiers:

    if log == True:
        sys.stdout = open("results/sensor_based_wo_overlapping_w_smote_CV" + str(k) + "_" + classifier + ".txt", "w")

    print(
        "ALGORITHM, GESTURE, # OF SELECTED SENSORS, PARTICIPANT_NO, ACCURACY, AUC, #ofPositiveInstance, #ofNegativeInstance,"
        " TN, FP, FN, TP, FAR, FRR, EER")

    for selected_features in selected_sensors:
        for gesture in gestures:
            for participant in participants:

                x_data, y_data = loadData(gesture, participant, selected_features, overlapping)

                if validation == "CV":
                    kf = model_selection.StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
                    final_y_test, final_y_pred = [], []

                    for train_index, test_index, in kf.split(x_data, y_data):
                        x_train = x_data[train_index]
                        y_train = y_data[train_index]
                        x_test = x_data[test_index]
                        y_test = y_data[test_index]
                        final_y_test = np.concatenate([final_y_test, y_test])

                        if smote == True:
                            sm = SMOTE(sampling_strategy='auto', k_neighbors=2, random_state=seed)
                            x_train, y_train = sm.fit_resample(x_train, y_train)

                        if classifier == "RandomForest_501":
                            clf = RandomForestClassifier(n_estimators=501)
                        if classifier == "AdaBoost_501":
                            clf = AdaBoostClassifier(n_estimators=501, random_state=seed)
                        if classifier == "MLP":
                            clf = MLPClassifier(hidden_layer_sizes=(100, 10))
                        if classifier == "SVM_rbf":
                            clf = SVC(kernel='rbf')
                        if classifier == "SVM_poly":
                            clf = SVC(kernel='poly')

                        if classifier == "SVM_rbf+SVM_poly":
                            estimators = []
                            model1 = SVC(kernel='rbf')
                            estimators.append(('svm_rbf', model1))
                            model2 = SVC(kernel='poly')
                            estimators.append(('svm_poly', model2))
                            clf = VotingClassifier(estimators)

                        if classifier == "MLP+SVM_rbf":
                            estimators = []
                            model1 = MLPClassifier(hidden_layer_sizes=(100, 10))
                            estimators.append(('mlp', model1))
                            model2 = SVC(kernel='rbf')
                            estimators.append(('svm_rbf', model2))
                            clf = VotingClassifier(estimators)

                        if classifier == "MLP+SVM_poly":
                            estimators = []
                            model1 = MLPClassifier(hidden_layer_sizes=(100, 10))
                            estimators.append(('mlp', model1))
                            model2 = SVC(kernel='poly')
                            estimators.append(('svm_poly', model2))
                            clf = VotingClassifier(estimators)


                        clf.fit(x_train, y_train)
                        y_pred = clf.predict(x_test)
                        final_y_pred = np.concatenate([final_y_pred, y_pred])

                #if validation == "TT":
                #    x_train, x_test, y_train, final_y_test = \
                #        model_selection.train_test_split(x_data, y_data, test_size=0.30, random_state=42, shuffle=True)
                #
                #    if smote == True:
                #        sm = SMOTE(sampling_strategy='auto', k_neighbors=2, random_state=42)
                #        x_train, y_train = sm.fit_resample(x_train, y_train)
                #
                #    clfRF = RandomForestClassifier(n_estimators=501)

                #    clfRF.fit(x_train, y_train)
                #    final_y_pred_RF = clfRF.predict(x_test)

                printAll(classifier, final_y_test, final_y_pred, gesture, participant, clf, selected_features)

    if log == True:
        sys.stdout.close()