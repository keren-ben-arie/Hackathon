import datetime
import re
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

pkl_files = ["knn_pickle.pkl", "forest_pkl", "tree_pickle.pkl", "logistic_pickle.pkl"]

crimes_dict = {'BATTERY': 0, 'THEFT': 1, 'CRIMINAL DAMAGE': 2, 'DECEPTIVE PRACTICE': 3, 'ASSAULT': 4}
T = 20


def get_from_pickle():
    classifiers = []
    for i in range(len(pkl_files)):
        with open(pkl_files[i], 'rb') as file:
            pickle_model = pickle.load(file)
            classifiers[i] = pickle_model
    return classifiers


CLASSIFIERS = get_from_pickle()


def most_frequent(List):
    return max(set(List), key=List.count)


def predict(X):
    y_hats = []  # list of y vectors
    winning_y = []
    for classifier in CLASSIFIERS:
        y_hats.append(classifier.predict(X))
    j = 0
    while j < len(y_hats[0]):
        col_matrix = []
        for i in range(len(y_hats)):
            col_matrix.append(y_hats[i][j])
        winning_y[j] = most_frequent(col_matrix)
        j += 1
    return winning_y

def send_police_cars(X):
    pass

