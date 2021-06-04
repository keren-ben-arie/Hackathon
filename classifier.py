

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}

def predict(X):
    pass


def calc_dist(p0, p1):
    return np.linalg.norm(p0 - p1)

def send_police_cars(date):
    """
    method that receives a date (in the same date/time format as the dataset - the
    time is ignored) and outputs a list with 30 entries. Each entry is a tuple of the form (x,y,time) where
    "x" and "y" are of the same format as the dataset, and time is a date/time stamp of the same format
    as the dataset, where "time" falls in the specified day
    :param X:
    :return:
    """
    train_path = "trainCrimes.csv"
    X = pd.read_csv(train_path)
    X['hour'] = pd.to_datetime(X['Date']).apply(lambda y: int(y.hour))
    X['minutes'] = pd.to_datetime(X['Date']).apply(lambda y: int(y.minute))
    train = X[["X Coordinate", "Y Coordinate", "hour", "minutes"]].dropna()
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
        win_unite = str(win3) + ":" + str(win4) + ":" + "00"
        lst_final_winners.append((win1, win2, win_unite))
    return lst_final_winners


def most_frequent(List):
    return max(set(List), key=List.count)
