import numpy as np
from sklearn.cluster import KMeans
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

import train_models

pkl_files = ["knn_pickle.pkl", "forest_pkl.pkl", "tree_pickle.pkl", "logistic_pickle.pkl"]

crimes_dict = {'BATTERY': 0, 'THEFT': 1, 'CRIMINAL DAMAGE': 2, 'DECEPTIVE PRACTICE': 3, 'ASSAULT': 4}
T = 10


def get_from_pickle():
    classifiers = []
    for i in range(len(pkl_files)):
        with open(pkl_files[i], 'rb') as file:
            pickle_model = pickle.load(file)
            classifiers.append(pickle_model)
    return classifiers


CLASSIFIERS = get_from_pickle()


def most_frequent(List):
    return max(set(List), key=List.count)


def predict(X):
    X = pd.read_csv(X)
    y_hats = []  # list of y vectors
    winning_y = []
    X = train_models.process_data(X).dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    for classifier in CLASSIFIERS:
        y_hats.append(classifier.predict(X))
    j = 0
    while j < len(y_hats[0]):
        col_matrix = []
        for i in range(len(y_hats)):
            col_matrix.append(y_hats[i][j])
        winning_y.append(most_frequent(col_matrix))
        j += 1
    return winning_y

def calc_dist(p0, p1):
    return np.linalg.norm(p0 - p1)

def send_police_cars(X):
    """
    method that receives a date (in the same date/time format as the dataset - the
    time is ignored) and outputs a list with 30 entries. Each entry is a tuple of the form (x,y,time) where
    "x" and "y" are of the same format as the dataset, and time is a date/time stamp of the same format
    as the dataset, where "time" falls in the specified day
    :param X:
    :return:
    """
    train_path = "Dataset_crimes.csv"
    train_path2 = "crimes_dataset_part2.csv"
    train = pd.read_csv(train_path).dropna()
    train2 = pd.read_csv(train_path2).dropna()
    train_co = pd.concat([train, train2], axis=0, ignore_index=True)
    train_co['hour'] = pd.to_datetime(train_co['Date']).apply(lambda y: int(y.hour))
    train_co['minutes'] = pd.to_datetime(train_co['Date']).apply(lambda y: int(y.minute))
    train = train_co[["X Coordinate", "Y Coordinate", "hour", "minutes"]].dropna()
    winners_to_return = []
    for date in X:
        kmeans = KMeans(n_clusters=700)
        coords_KMeans = kmeans.fit(train)
        lables = coords_KMeans.labels_
        cluster_centers = coords_KMeans.cluster_centers_
        score = dict(zip(range(700), np.zeros(700)))
        for i in range(len(lables)):
            p0 = cluster_centers[lables[i]]
            p1 = train.iloc[i]
            if p0[2] > p1[2]:
                hour_diff = p0[2] - p1[2]
                minutes_diff = (p0[3] - p1[3])/60
            else:
                hour_diff = p1[2] - p0[2]
                minutes_diff = (p1[3] - p0[3])/60
            time_diff = hour_diff + minutes_diff
            if calc_dist(p0[:2], p1[:2]) <= 500 and time_diff <= 0.5:
                score[lables[i]] += 1
        counter = 0
        winners = []
        sorted_vals = reversed(sorted(score.values()))
        for value in sorted_vals:
            for key in score.keys():
                if counter < 30 and score[key] == value:
                    winners.append(cluster_centers[key])
                    counter += 1
                if counter >= 30:
                    break

        lst_final_winners = []
        for winner in winners:
            win1 = winner[0]
            win2 = winner[1]
            win3 = int(winner[2])
            win4 = int(winner[3])
            win_unite = date.strftime("%m/%d/%Y ") + str(win3) + ":" + str(win4) + ":" + "00"
            lst_final_winners.append((win1, win2, win_unite))
        winners_to_return.append(lst_final_winners)

    return winners_to_return
