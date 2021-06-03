import datetime
import re
import pandas as pd

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


def test_apply_datetime(x):
    try:
        return datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S')
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
            data[col].apply(test_apply_datetime)
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


def check_valid_date(s):
    str_arr = s.split(" ")
    date = str_arr[0]
    str_date_arr = date.split("/")
    if int(str_date_arr[0]) < 1 or int(str_date_arr[0]) > 12:
        return None
    if int(str_date_arr[1]) < 1 or int(str_date_arr[1]) > 31:
        return None
    if int(str_date_arr[2]) > 2021:
        return None


def clean_dates(data):
    data['Date'].apply(check_valid_date)
    data['Updated On'].apply(check_valid_date)


if __name__ == '__main__':
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
    y_train = train_set['Primary Types']
    x_train = train_set.drop(['Primary Types'])

