import random
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import sklearn
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn import model_selection
from sklearn.model_selection import cross_validate
from Source.LVWMetadataPrep import preprocess_metadata

# import of classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# imports for cross validation
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def LVW(meta_x, meta_Y, K, classifier):
    err = 0
    k = 0
    C = 100
    # Best features
    S = np.array([])
    # Create list of features
    while k < K:
        # Create a list of available features
        features_ind = np.array(list(range(0, len(meta_x.columns))))
        num_features = len(features_ind)
        # Generate how many features should be selected
        C1 = np.random.randint(1,num_features+1)
        # Randomly pick num_selected_features as subset
        S1_ind = np.random.choice(features_ind, size=C1, replace = False)
        # Sort it, because it looks nicer...
        S1_ind = np.sort(S1_ind)
        # Train the classifier
        classifier.fit(meta_x.iloc[:,S1_ind], meta_Y)
        # Predict the results
        pred = classifier.predict(meta_x.iloc[:,S1_ind])
        # Comptue f1 measure (since in the paper it is stated that they tune it based on the f1 score)
        f1 = f1_score(meta_Y, pred)

        if (f1 > err or (f1 == err and C1 < C)):
            k = 0
            S = S1_ind
            err = f1
            C = C1
        else:
            k = k+1
    # Test on testset
    classifier.fit(meta_x.iloc[:,S], meta_Y)
    #test_pred = classifier.predict()

    # Get column names
    columns = meta_x.iloc[:,S].columns
    return columns


def perform_cross_validation(X, y, classi):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    predictions = []
    ground_truth = []
    for train_idx, test_idx in kf.split(X, y):
        X_train = X.loc[train_idx]
        X_test = X.loc[test_idx]
        y_train = y.loc[train_idx]
        y_test = y.loc[test_idx]
        # Perform Las Vegas Wrapper
        ind = LVW(X_train, y_train, 10, classi)

        classi.fit(X_train[ind], y_train)
        # Predict
        fold_predictions = classi.predict(X_test[ind])
        ground_truth.extend(y_test)
        predictions.extend(fold_predictions)

    # Compute the metrics
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    return precision, recall, f1


def classify(type, classifiers, data_x, data_y, scoring):
    for classi in classifiers:
        # Perform LVW (not for random forest)
        if isinstance(classi, RandomForestClassifier):
            ind = LVW(data_x, data_y, 10, classi)
            scores = cross_validate(classi, X=data_x, y=data_y, cv=10,scoring=scoring)
            f_1 = np.mean(scores["test_f1_score"])
            precision = np.mean(scores["test_precision"])
            recall = np.mean(scores["test_recall"])
        else:
            ind = LVW(data_x, data_y, 10, classi)
            scores = perform_cross_validation(data_x[ind], data_y, classi)
            precision = scores[0]
            recall = scores[1]
            f_1 = scores[2]
        print("Classifier: %s, Modality: %s, Precision: %.3f, Recall: %.3f, F1: %.3f" % (classi.__class__.__name__, type, precision, recall, f_1))



warnings.filterwarnings('ignore')
np.random.seed(50)

# Define Scoring methods
scoring = {'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score),
           'f1_score' : make_scorer(f1_score)}

# Read Audio Data
audio_df = pd.read_csv("../Results/audio.csv", sep=",")
audio_x = audio_df.drop("goodforairplane", axis=1)
audio_Y = audio_df["goodforairplane"]

# Read Textual Data
textual_df = pd.read_csv("../Results/textual.csv", sep=",")
textual_x = textual_df.drop("goodforairplane", axis=1)
textual_Y = textual_df["goodforairplane"]

# Read Visual Data
visual_df = pd.read_csv("../Results/visual.csv", sep=",")
visual_x = visual_df.drop("goodforairplane", axis=1)
visual_Y = visual_df["goodforairplane"]

# Read in Meta Data
metadata_df = pd.read_csv("../Results/meta_plus_ratings.csv", sep=",")
# Preprocess
metadata_x, metadata_Y = preprocess_metadata(metadata_df)

metadata_classifiers = [KNeighborsClassifier(), NearestCentroid(),
               DecisionTreeClassifier(), LogisticRegression(),
               SVC(), BaggingClassifier(), RandomForestClassifier(),
               AdaBoostClassifier(), GradientBoostingClassifier(), GaussianNB()]

audio_classifiers = [LogisticRegression(),
               GradientBoostingClassifier()]

textual_classifiers = [KNeighborsClassifier(),
               SVC(), GaussianNB()]

visual_classifiers = [KNeighborsClassifier(),
               DecisionTreeClassifier(), LogisticRegression(),
               SVC(), RandomForestClassifier(),
               AdaBoostClassifier(), GradientBoostingClassifier()]

#classify('metadata', metadata_classifiers, metadata_x, metadata_Y, scoring)
classify('audio', audio_classifiers, audio_x, audio_Y, scoring)
classify('textual', textual_classifiers, textual_x, textual_Y, scoring)
classify('visual', visual_classifiers, visual_x, visual_Y, scoring)