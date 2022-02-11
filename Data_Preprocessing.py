import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek


def data_preprocessing():
    # reading the data from the "bank-additional-full.csv"
    df = pd.read_csv("./Dataset/bank-additional-full.csv", delimiter=';')

    # checking if the dataset has a missing value
    checking_missing_values = df.isnull().sum()
    print("----------------------------------------------------------------------------")
    print("The sum of missing value in each features:")
    print("----------------------------------------------------------------------------")
    print(checking_missing_values)

    # slicing the features and labels from the dataset
    x_data = df.loc[:, df.columns != "y"]
    y_data = df["y"].copy(deep=True)

    # printing the dataset information
    print("----------------------------------------------------------------------------")
    print("Dataset Description:")
    print("----------------------------------------------------------------------------")
    print("\nNumber of features/instances in a dataset:", x_data.shape[1])
    print("\nTotal number of samples in a dataset:", x_data.shape[0])
    print("\nList of unique class-labels: ", np.unique(y_data))
    print("----------------------------------------------------------------------------")

    # one hot encoding on the categorical features
    x_data = pd.get_dummies(x_data)
    x_data = x_data.values
    x_data = x_data.astype(int)

    # check for class imbalance
    print("Number of instances in each classes:")
    print(y_data.value_counts())
    print("----------------------------------------------------------------------------")

    # label encoding for y label
    label_encoder = LabelEncoder()
    y_data = label_encoder.fit_transform(y_data)

    return x_data, y_data


def over_sampling(x_train, y_train):
    # applying SMOTE (oversampling) method
    smote = SMOTE(sampling_strategy='auto')
    x_oversampled, y_oversampled = smote.fit_resample(x_train, y_train)
    x_train = x_oversampled
    y_train = y_oversampled

    return x_train, y_train


def under_sampling(x_train, y_train):
    # applying RandomUnderSampler (under sampling) method
    under_sample = RandomUnderSampler(replacement=False)
    x_undersampled, y_undersampled = under_sample.fit_resample(x_train, y_train)
    x_train = x_undersampled
    y_train = y_undersampled

    return x_train, y_train


def re_sampling(x_train, y_train):
    # applying SMOTE with under sampling method
    resample_sample = SMOTEENN()
    x_resample, y_resample = resample_sample.fit_resample(x_train, y_train)
    x_train = x_resample
    y_train = y_resample

    return x_train, y_train