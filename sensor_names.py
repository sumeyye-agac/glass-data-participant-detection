

import numpy as np

Acc = range(1,10)
Gyr = range(10,19)
MagField = range(19,28)
RotVec = range(28,37)

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



