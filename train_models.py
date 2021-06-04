import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

crimes_dict = {'BATTERY': 0, 'THEFT': 1, 'CRIMINAL DAMAGE': 2,
               'DECEPTIVE PRACTICE': 3, 'ASSAULT': 4}
CLASSIFIERS = []
T = 10


def most_frequent(List):
    """ function for finding finding the best model prediction """
    return max(set(List), key=List.count)


# dictionary of all types of the columns
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
    """ helper function for the validity checks - checks ins """
    try:
        return int(x)
    except ValueError:
        return None


def test_apply_float(x):
    """ helper function for the validity checks - checks floats """
    try:
        return float(x)
    except ValueError:
        return None


def test_apply_bool(x):
    """ helper function for the validity checks - checks booleans """
    try:
        return bool(x)
    except ValueError:
        return None


def check_types_validity(data):
    """ function for all validity checks """
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


def clean_ward(data):
    """ helper function for cleaning the ward """
    data['Ward'].apply(lambda w: None if w < 1 or w > 50 else w)


def clean_community_area(data):
    """ helper function for cleaning the community area """
    data['Community Area'].apply(lambda c: None if c < 1 or c > 77 else c)


def clean_x_y_coordinates(data):
    """ helper function for cleaning the X coordinates and Y coordinates """
    data["X Coordinate"].apply(
        lambda w: None if w < 1111111 or 1999999 <= w else w)
    data["Y Coordinate"].apply(
        lambda w: None if w < 1111111 or 1999999 <= w else w)


def clean_dates(data):
    """" helper function for cleaning the X coordinates and Y coordinates """
    data['Date'].apply(lambda y: None if y.year > 2021 else y)
    data['Updated On'].apply(lambda y: None if y.year > 2021 else y)


def filtering(data):
    """ Preprocessing the data """
    data['Case Number'] = data['Case Number'].apply(lambda str: int(str[2:]))
    remove = ['ID', 'Year', "Location", "Block", "Location Description",
              'Description', 'FBI Code', 'IUCR', 'Longitude', 'Latitude',
              'District']
    # removing categories
    for r in remove:
        data = data.drop(r, axis=1)
    binary_category = ['Arrest', 'Domestic']
    for r in binary_category:
        data[r] = data[r].apply(lambda binary: 1 if binary is True else 0)
    return data


def preprocess_dates(data):
    """ Preprocessing the dates """
    # Setting Day Feature
    days_dict = {'Sunday': 1, 'Monday': 2, 'Tuesday': 3, 'Wednesday': 4,
                 'Thursday': 5, 'Friday': 6, 'Saturday': 7}
    data['Day'] = data['Date'].apply(lambda y: days_dict[y.strftime("%A")])
    # Setting Time Feature
    data['Time'] = data['Date'].apply(lambda y: int(y.strftime("%H%M")))
    # Setting Updated On - differences in minutes
    data['Updated On'] = data['Updated On'] - data['Date']
    data['Updated On'] = data['Updated On'].apply(lambda y: y.total_seconds() / 60)
    # Drop Date Feature
    return data.drop('Date', axis=1)


def preprocess():
    """ main function that does the preprocessing """
    train_path = "Dataset_crimes.csv"
    train_path2 = "crimes_dataset_part2.csv"
    train = process_data(pd.read_csv(train_path)).dropna()
    train2 = process_data(pd.read_csv(train_path2)).dropna()
    train_co = pd.concat([train, train2], axis=0, ignore_index=True)
    y_train = train_co["Primary Type"].apply(lambda bin: crimes_dict.get(bin))
    X_train = train_co.drop("Primary Type", axis=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    return X_train, y_train


def Bagging(x_train, y_train, T, classifiers):
    """ the function that helps the bagging classifier"""
    for i in range(len(classifiers)):
        classifiers[i] = BaggingClassifier(base_estimator=classifiers[i],
                                           n_estimators=T, random_state=0).fit \
            (x_train, y_train)
    return classifiers


def Adaboost(x_train, y_train, T, classifiers):
    """ the function that helps the adaboost classifier"""
    for i in range(len(classifiers)):
        classifiers[i] = AdaBoostClassifier(base_estimator=classifiers[i],
                                            n_estimators=T,
                                            random_state=0).fit(x_train,
                                                                y_train)
    return classifiers


def process_data(data):
    """ processing and cleaning the data"""
    data = check_types_validity(data)
    clean_dates(data)
    clean_ward(data)
    clean_community_area(data)
    clean_x_y_coordinates(data)
    return filtering(preprocess_dates(data))


def create_classifiers():
    """ creating all the classifiers to work on """
    knn = KNeighborsClassifier(n_neighbors=5)
    forest = RandomForestClassifier(n_estimators=20, criterion='entropy',
                                    random_state=42, max_depth=5)
    tree = DecisionTreeClassifier(criterion="gini", random_state=100,
                                  max_depth=3, min_samples_leaf=5)
    logistic = LogisticRegression(solver='liblinear')

    return [knn, forest, tree, logistic]


def train():
    """ main function to train the model """
    X_train, y_train = preprocess()
    CLASSIFIERS = create_classifiers()
    CLASSIFIERS_1 = Bagging(X_train, y_train, T, CLASSIFIERS[1:2])
    CLASSIFIERS_2 = Adaboost(X_train, y_train, T, CLASSIFIERS[2:])
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    models = [knn]
    models += CLASSIFIERS_1 + CLASSIFIERS_2
    pkl_files = ["knn_pickle.pkl", "forest_pkl.pkl", "tree_pickle.pkl", "logistic_pickle.pkl"]
    for i in range(len(pkl_files)):
        with open(pkl_files[i], 'wb') as file:
            pickle.dump(models[i], file, protocol=pickle.HIGHEST_PROTOCOL)
