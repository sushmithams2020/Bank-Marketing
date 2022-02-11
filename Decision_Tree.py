from Data_Preprocessing import data_preprocessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit , GridSearchCV
from sklearn.metrics import accuracy_score,recall_score,precision_score,roc_curve,roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


def classification(x_data, y_data):

    scalar = MinMaxScaler()

    classifier = DecisionTreeClassifier()
    acc_scores = []
    prec_scores= []
    rec_scores = []
    auc_scores = []

    params_dict = {"max_depth":range(1,10),
        "min_samples_split":range(2 , 10),
        "min_samples_leaf": range(1 , 5)}

    gridsearch  = GridSearchCV(classifier , param_grid=params_dict , cv = 5 , verbose=1,n_jobs=-1)
    gridsearch.fit(x_data , y_data)

    best_estimator= gridsearch.best_estimator_
    best_score = gridsearch.best_score_

    print("Best estimated parameters:", best_estimator)
    print("Best score:" , best_score)


def main():
    # loading the x_data and y_data from the dataset
    x_data, y_data = data_preprocessing()
    # performing classification
    classification(x_data, y_data)


if __name__ == '__main__':
    main()





