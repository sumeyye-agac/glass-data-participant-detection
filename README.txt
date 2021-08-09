----------------------------------------------------------------------------------------------------
                                      Sumeyye Agac
----------------------------------------------------------------------------------------------------
The code requires the following dependencies:
Python 3.7.10
numpy:  1.19.5
matplotlib:  3.4.2
sklearn: 0.24.2
csv: 1.0
----------------------------------------------------------------------------------------------------
./
├── code
│   ├── dataset
│   │    ├── p1_Circle.csv
│   │    ├── p1_Square.csv
│   │    ├── ...
│   │    └── p15_Updown.csv
│   ├── features
│   │    ├── Features_p1_Circle.csv
│   │    ├── Features_p1_Square.csv
│   │    ├── ...
│   │    └── Features_p15_Updown.csv
│   ├── 3dPlot.py
│   ├── feature_extraction.py
│   ├── participant_detection.py
│   └── README.txt
 

----------------------------------------------------------------------------------------------------
            		participant_detection.py
           ------------------------------------------------------------
Run participant_detection.py in a Python IDE or the terminal by typing:
>> python main.py

If features are extracted (use feature_extraction.py to extract if needed.)
it will generate the performance results for each gesture-participant combination
by using best 5, 10, 15 and 20 features and 36 features.

By setting log to True (log=True) results can be saved in a csv file
and be investigated in a more detailed way further.

By changing gestures, participants, selected_features_list variables we realize a specific experiment
instead of all which is the default setting.

----------------------------------------------------------------------------------------------------
                          AN EXAMPLE OF OUTPUT
----------------------------------------------------------------------------------------------------

participant_detection.py

