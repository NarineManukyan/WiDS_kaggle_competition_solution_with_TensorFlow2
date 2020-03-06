import pandas as pd

import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler


def read_data(training_file, predictions_file, predictions_features_file):
    training = pd.read_csv(training_file)
    predictions = pd.read_csv(predictions_features_file)

    return training, predictions


def data_cleaning(training, predictions):

    # Identify training features
    categorical_cols = list(training.select_dtypes('object').columns)
    id_cols = ['encounter_id', 'patient_id', 'hospital_id','readmission_status', 'icu_id', 'ethnicity']
    target_col = ['hospital_death']
    exclude_cols = categorical_cols + id_cols + target_col
    numerical_col = [col for col in training.columns if col not in exclude_cols]

    training_target = training[target_col]
    training_features = training[numerical_col]
    prediction_features = predictions[numerical_col]

    # handle missing data
    training_features.fillna(0, inplace=True)
    prediction_features.fillna(0, inplace=True)

    # scale data, note this turns DataFrames to numpy arrays
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    scaler.fit(training_features)
    training_features = scaler.transform(training_features)
    prediction_features = scaler.transform(prediction_features)


    return training_target, training_features, prediction_features


def build_model(training_features, training_target):
    model = tf.keras.Sequential([keras.layers.Dense(units=174, activation='tanh', input_shape=[174]),
                                 keras.layers.Dense(64, activation='relu'),
                                 keras.layers.Dense(64, activation='relu'),
                                 keras.layers.Dense(1, activation='tanh')])

    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(training_features, training_target.values, epochs=5)

    print('here')

    return model


def main():
    training_file = 'data/training_v2.csv'
    predictions_file = 'data/solution_template.csv'
    predictions_features_file = 'data/unlabeled.csv'

    training, predictions = read_data(training_file, predictions_file, predictions_features_file)
    training_target, training_features, prediction_features = data_cleaning(training, predictions)

    model = build_model(training_features, training_target)
    output = model.predict(prediction_features)
    result = pd.DataFrame()
    result['encounter_id'] = predictions['encounter_id']
    result['hospital_death'] = output > 0.5
    result.hospital_death = result.hospital_death.astype(int)
    result.to_csv('solution.csv')


main()


#print(model.predict([7.0]))


