

import numpy as np

Acc = range(1,10)
Gyr = range(10,19)
MagField = range(19,28)
RotVec = range(28,37)

columns_circle_all = [29, 30, 35, 24, 32, 19, 2, 31, 14, 22, 8, 20, 12, 23, 25, 21, 11, 27, 33, 36, 26, 4,
                      28, 5, 3, 15, 34, 17, 13, 18, 10, 1, 6, 7, 9, 16]
columns_updown_all = [19, 22, 25, 20, 32, 23, 33, 29, 36, 35, 30, 21, 24, 28, 26, 27, 31, 4, 1, 7, 34, 10,
                      6, 2, 13, 8, 5, 3, 14, 9, 11, 15, 17, 12, 18, 16]
columns_tilt_all = [27, 24, 21, 29, 30, 20, 31, 9, 28, 32, 26, 34, 6, 23, 35, 3, 33, 36, 2, 19, 22, 8, 12,
                    25, 1, 14, 15, 4, 11, 5, 13, 10, 7, 17, 16, 18]
columns_triangle_all = [32, 22, 19, 24, 29, 30, 25, 21, 20, 31, 28, 34, 23, 27, 35, 33, 26, 36, 4, 1, 14,
                        2, 15, 12, 6, 5, 11, 8, 7, 3, 18, 13, 10, 17, 9, 16]
columns_turn_all = [20, 24, 26, 23, 22, 31, 29, 21, 28, 27, 33, 3, 19, 34, 4, 8, 9, 30, 32, 6, 25, 2, 7,
                    5, 35, 14, 1, 11, 12, 15, 36, 10, 13, 18, 16, 17]
columns_square_all = [19, 24, 29, 25, 22, 30, 27, 20, 32, 28, 21, 33, 23, 36, 35, 31, 26, 34, 18, 1, 15,
                      12, 7, 4, 5, 8, 3, 2, 6, 11, 14, 17, 13, 9, 10, 16]

selected_features_list = range(1,37)

sensors = ["Acc", "Gyr", "MagField", "RotVec"]
gestures = ["Circle", "Updown", "Tilt", "Triangle", "Turn", "Square"]


for selected_features in selected_features_list:
    for gesture in gestures:

        sensors = []

        if gesture == "Circle":
            features = columns_circle_all[:selected_features]
        if gesture == "Updown":
            features = columns_updown_all[:selected_features]
        if gesture == "Tilt":
            features = columns_tilt_all[:selected_features]
        if gesture == "Triangle":
            features = columns_triangle_all[:selected_features]
        if gesture == "Turn":
            features = columns_turn_all[:selected_features]
        if gesture == "Square":
            features = columns_square_all[:selected_features]

        for feature in features:
            if feature in Acc:
                sensors.append("Acc")
            if feature in Gyr:
                sensors.append("Gyr")
            if feature in MagField:
                sensors.append("MagField")
            if feature in RotVec:
                sensors.append("RotVec")

        sensors_needed = sorted(np.unique(np.array(sensors)))

        sensor_list = ""

        if "Acc" in sensors_needed: sensor_list += "Acc, "
        else: sensor_list += ", "

        if "Gyr" in sensors_needed: sensor_list += "Gyr, "
        else: sensor_list += ", "

        if "RotVec" in sensors_needed: sensor_list += "RotVec, "
        else: sensor_list += ", "

        if "MagField" in sensors_needed: sensor_list += "MagField, "
        else: sensor_list += ", "

        print(gesture, ", ", selected_features, ", ", sensor_list, ", ", len(sensors_needed))



