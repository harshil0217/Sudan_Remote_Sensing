import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def load_data():
    # Load data from the CSV file
    data = pd.read_csv('data/village_LANDSAT.csv')
    return data

def equalize_classes(data):
    # Equalize the number of samples in each class
    damaged = data[data['STATUS'] == 'DAMAGED']
    no_damage = data[data['STATUS'] == 'NO DAMAGE']
    no_damage = no_damage.sample(n=len(damaged), random_state=69)
    data = pd.concat([damaged, no_damage])
    return data

def drop_latlong(data):
    # drop lat and long info
    data = data.drop(columns=['LAT_DD', 'LONG_DD'])
    return data

def standardize_data(data):
    #replace strings in "STATUS" with integers
    data['STATUS'] = data['STATUS'].replace(['DAMAGED', 'NO DAMAGE'], [0, 1])

    #normalize the other columns
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return data

def split_data(data):
    # Split the data into training and testing sets
    X = data.drop(columns=['STATUS'])
    y = data['STATUS']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69, stratify=y)
    return X_train, X_test, y_train, y_test

def preprocess_data():
    data = load_data()
    data = equalize_classes(data)
    data = drop_latlong(data)
    data = standardize_data(data)
    X_train, X_test, y_train, y_test = split_data(data)
    return X_train, X_test, y_train, y_test



    