import datetime
import re
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import KNN

crimes_dict = {'BATTERY': 0, 'THEFT': 1, 'CRIMINAL DAMAGE': 2, 'DECEPTIVE PRACTICE': 3, 'ASSAULT': 4}


def predict(X):
    pass


def send_police_cars(X):
    pass


COLUMNS_TYPES = {'ID': 'int',
                 'Date': 'datetime',
                 'Year': 'int',
                 'Updated On': 'datetime',
                 'Block': 'str',
                 'Location Description': 'str',
                 'Arrest': 'bool',
                 'Domestic': 'bool',
                 'Beat': 'str',
                 'District': 'int',
                 'Ward': 'int',
                 'Community Area': 'int',
                 'X Coordinate': 'float',
                 'Y Coordinate': 'float',
                 'Latitude': 'float',
                 'Longitude': 'float'}


def test_apply_int(x):
    try:
        return int(x)
    except ValueError:
        return None


def test_apply_float(x):
    try:
        return float(x)
    except ValueError:
        return None


def test_apply_bool(x):
    try:
        return bool(x)
    except ValueError:
        return None


def check_types_validity(data):
    for col in data:
        if col not in COLUMNS_TYPES:
            continue
        if COLUMNS_TYPES[col] == 'int':
            data[col].apply(test_apply_int)
        if COLUMNS_TYPES[col] == 'float':
            data[col].apply(test_apply_float)
        if COLUMNS_TYPES[col] == 'bool':
            data[col].apply(test_apply_bool)
        if COLUMNS_TYPES[col] == 'datetime':
            data[col] = pd.to_datetime(data[col])
        else:
            continue
    return data


def clean_year(data):
    data['Year'].apply(lambda year: None if year > 2021 else year)


def clean_arrest(data):
    data['District'].apply(lambda d: None if d < 1 or d == 13 or d == 21 or d == 23 or d > 25 else d)


def clean_ward(data):
    data['Ward'].apply(lambda w: None if w < 1 or w > 50 else w)


def clean_community_area(data):
    data['Community Area'].apply(lambda c: None if c < 1 or c > 77 else c)


def clean_x_y_coordinates(data):
    data["X Coordinate"].apply(lambda w: None if w < 1111111 or 1999999 <= w else w)
    data["Y Coordinate"].apply(lambda w: None if w < 1111111 or 1999999 <= w else w)


def clean_block(data):
    reg = "[0-9]X{1,2} [NESW] [A-Z]+ [A-Z]"
    data['Block'].apply(lambda block: None if not re.match(reg, block) else block)


def clean_longitude_latitude(data):
    data['Longitude'].apply(lambda lng: None if lng not in range(-88, -86) else lng)
    data['Latitude'].apply(lambda lat: None if lat not in range(40, 43) else lat)


def clean_dates(data):
    data['Date'].apply(lambda y: None if y.year > 2021 else y)
    data['Updated On'].apply(lambda y: None if y.year > 2021 else y)


# Preprocessing

def filtering(data):
    data['Case Number'] = data['Case Number'].apply(lambda str: int(str[2:]))
    remove_for_task1 = ['Description', 'FBI Code', 'IUCR']
    remove = ['ID', 'Block', 'Location Description', 'Year', "Location"]
    for r in remove:
        data = data.drop(r, axis=1)
    for r in remove_for_task1:
        data = data.drop(r, axis=1)
    binary_category = ['Arrest', 'Domestic']
    for r in binary_category:
        data[r] = data[r].apply(lambda bin: 1 if bin == True else 0)
    return data


def preprocess_dates(data):
    # Setting Day Feature
    days_dict = {'Sunday': 1, 'Monday': 2, 'Tuesday': 3, 'Wednesday': 4, 'Thursday': 5, 'Friday': 6, 'Saturday': 7}
    data['Day'] = data['Date'].apply(lambda y: days_dict[y.strftime("%A")])
    # Setting Time Feature
    data['Time'] = data['Date'].apply(lambda y: int(y.strftime("%H%M")))
    # Setting Month Feature - Optional

    # Setting Updated On - differences in minutes
    data['Updated On'] = data['Updated On'] - data['Date']
    data['Updated On'] = data['Updated On'].apply(lambda y: y.total_seconds() / 60)

    # Drop Date Feature
    return data.drop('Date', axis=1)




def process_data():
    train_set = pd.read_csv("trainCrimes.csv")
    check_types_validity(train_set)
    clean_dates(train_set)
    clean_block(train_set)
    clean_ward(train_set)
    clean_year(train_set)
    clean_arrest(train_set)
    clean_community_area(train_set)
    clean_longitude_latitude(train_set)
    clean_x_y_coordinates(train_set)
    return filtering(preprocess_dates(train_set))


if __name__ == '__main__':
    train = process_data()
    y_train = train["Primary Type"].apply(lambda bin: crimes_dict.get(bin))
    X_train = train.drop("Primary Type", axis=1)
    print(y_train)
    print(X_train)
    X_train = X_train.fillna(0)
    knn = KNN.KNearest()
    knn.fit(X_train, y_train)
    #y_hat = knn.predict(X_train)
    score = knn.model.score(X_train, y_train)
    #print(y_hat)
    print(score)
