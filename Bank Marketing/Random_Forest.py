
from Data_Preprocessing import data_preprocessing
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, confusion_matrix


def classification(x_data, y_data):

    # defining the classifier
    classifier = RandomForestClassifier()
    # hyper parameter tuning
    n_estimator = [int(i) for i in np.linspace(start=10, stop=100, num=10)]
    max_depth = [int(i) for i in np.linspace(1, 50, num=5)]
    max_features = ['auto', 'sqrt']
    parameters = {'n_estimators': n_estimator, 'max_depth': max_depth, 'max_features': max_features}


    # scoring = {'Confusion matrix': confusion_matrix, 'Accuracy': accuracy_score}
    # grid search
    RF_grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, cv=5)
    RF_grid_search.fit(x_data, y_data)

    best_parameter = RF_grid_search.best_params_
    print(best_parameter)
    print("Overall Accuracy of Random Forest:", RF_grid_search.best_score_)


def main():
    # loading the x_data and y_data from the dataset
    x_data, y_data = data_preprocessing()
    # performing classification
    classification(x_data, y_data)


if __name__ == '__main__':
    main()