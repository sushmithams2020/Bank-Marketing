from Data_Preprocessing import data_preprocessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS



def classification(x_data, y_data):
    # defining classifier
    classifier = LogisticRegression()
    # 5 fold cross validation
    random_split = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
    acc_scores = []
    prec_scores = []
    rec_scores = []
    auc_scores = []


    for train_idx, test_idx in random_split.split(x_data, y_data):
        # fetching train and test index
        x_train = x_data[train_idx]
        y_train = y_data[train_idx]

        x_test = x_data[test_idx]
        y_test = y_data[test_idx]

        # Normalization on the x_train and x_test separately
        scalar = MinMaxScaler()
        x_train = scalar.fit_transform(x_train)
        x_test = scalar.fit_transform(x_test)

        # Sequential Floating Forward Selection
        sffs = SFS(classifier, k_features=(3, 15), forward=True, floating=True, cv=0)
        sffs = sffs.fit(x_train, y_train)
        # finding the best feature subset
        print('best combination (ACC: %.3f): %s\n' % (sffs.k_score_, sffs.k_feature_idx_))

        # fetching the important feature index
        x_train_sfs = sffs.transform(x_train)
        x_test_sfs = sffs.transform(x_test)

        # training and testing using classifier
        classifier.fit(x_train_sfs, y_train)
        y_predicted = classifier.predict(x_test_sfs)

        acc = accuracy_score(y_test, y_predicted)
        acc_scores.append(acc)

        prec = precision_score(y_test, y_predicted)
        prec_scores.append(prec)

        recall = recall_score(y_test, y_predicted)
        rec_scores.append(recall)

        # Calculate area under curve (AUC)
        y_pred_prob = classifier.predict_proba(x_test_sfs)
        y_pred_prob = y_pred_prob[:, 1]
        auc = roc_auc_score(y_test, y_pred_prob)
        auc_scores.append(auc)

    average_accuracy = np.mean(acc_scores)
    average_precision = np.mean(prec_scores)
    average_recall = np.mean(rec_scores)
    average_auc = np.mean(auc_scores)

    print("Accuracy of each fold - {}".format(acc_scores))
    print("Average accuracy : {}".format(average_accuracy))
    print("Average precision : {}".format(average_precision))
    print("Average recall : {}".format(average_recall))
    print("Average AUC : {}".format(average_auc))




def main():
    # loading the x_data and y_data from the dataset
    x_data, y_data = data_preprocessing()
    # performing classification
    classification(x_data, y_data)


if __name__ == '__main__':
    main()
