"""
Airbnb New User Bookings Comptetition
https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings

Author: Sandro Vega Pons (sv.pons@gmail.com)

Classifiers based on scikit-learn code.
"""

import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from letor_metrics import ndcg_score


def clf_sklearn(clf, data, random_state=0, calibrated=False, clf_name='lr',
                ext_name=''):
    """
    General function for the application of scikit-learn classifiers.
    The functions apply the classifier twice:
    - First: Fit the classifier to (X_train, y_train) and predict on (X_valid).
             The prediction is stored in 'save/valid' folder.
    - Second: Fit the classifier to (X, y) = (X_train + X_valid, y_train + y_valid)
             and predict on (X_test). The prediction is stored in 'save/test'
             folder.

    Parameters:
    ----------
    clf: sklearn Classifier
        The classifier
    data: list
         [X_train, y_train, X_valid, y_valid, X_test]
    random_state: numpy RandomState
         RandomState used for reproducibility
    calibrated: bool
         Whether to calibrate the output probabilities with CalibratedClassifierCV
    clf_name: string
         String that represents the name of the classifier. Used to identify
         the solutions in the save/valid and save/test folders
    ext_name: string
         Extra string to be used in the name of the stored prediction, e.g. it
         can be used to identify specific parameter values that were used.

    Result:
    ------
    y_valid_pred: numpy ndarray shape=(n_samples_validation, n_classes)
              Labels of the predictions for the validation set.
    y_test_pred: numpy ndarray shape=(n_samples_test, n_classes)
              Labels of the predictions for the test set.

    Save:
    ----
    y_valid_pred: it is stored in save/valid folder
    y_test_pred: it is stored in save/test folder
    """
    X_train, y_train, X_valid, y_valid, X_test = data

    ###Working on (X_Train => X_Valid)###
    #Normalizing the data
    ss = StandardScaler()
    XX_train = ss.fit_transform(X_train)
    XX_valid = ss.transform(X_valid)

    #Computing binary labels (required by the evaluation measure)
    lb = LabelBinarizer()
    lb.fit(y_train)
    yb_valid = lb.transform(y_valid)

    #Training the classifier
    clf.fit(XX_train, y_train)
    y_valid_pred = clf.predict_proba(XX_valid)

    #Calibration
    if calibrated:
        cc_clf = CalibratedClassifierCV(base_estimator=clf, method='isotonic',
                                        cv=3)
        cc_clf.fit(XX_train, y_train)
        y_cc_pred = cc_clf.predict_proba(XX_valid)
        #The calibrated solution is merged with the original one.Experimentally
        #produces better results than using the calibrated solution directly.
        y_valid_pred = (y_valid_pred + y_cc_pred) / 2.

    ndcg = np.mean([ndcg_score(tr, pr, k=5) for tr, pr in  \
                   zip(yb_valid.tolist(), y_valid_pred.tolist())])
    logloss = log_loss(y_valid, y_valid_pred)

    print 'Validation results with calibration, ndcg5: %s, logloss: %s' \
          %(ndcg, logloss)

    #Saving the result
    rnd = random_state.randint(1000, 9999)
    pickle.dump(y_valid_pred, open('save/valid/v_%s_%s_%s_%s_%s'  \
    %(clf_name, ext_name, rnd, round(ndcg, 4), round(logloss, 4)), 'w'))

    ###Working on X => X_test###
    X = np.vstack((X_train, X_valid))
    y = np.hstack((y_train, y_valid))

    XX = ss.fit_transform(X)
    XX_test = ss.transform(X_test)

    clf.fit(XX, y)
    y_test_pred = clf.predict_proba(XX_test)

    if calibrated:
        cc_clf.fit(XX, y)
        y_cc_pred = cc_clf.predict_proba(XX_test)
        y_test_pred = (y_test_pred + y_cc_pred) / 2.

    pickle.dump(y_test_pred, open('save/test/t_%s_%s_%s'%(clf_name, ext_name, rnd), 'w'))

    return y_valid_pred, y_test_pred



def clf_log_regression(data, random_state, calibrated=False, ext_name=""):
    """
    Application of logistic regression classifier. For details look at
    'clf_sklearn' function.
    """
    ###Defining the classifier###
    lr = LogisticRegression(penalty='l2', C=0.01,
                            multi_class='ovr',
                            max_iter=300,
                            solver='lbfgs',
                            n_jobs=-1, random_state=random_state)

    return clf_sklearn(lr, data, random_state, calibrated, clf_name='LR',
                       ext_name=ext_name)



def clf_random_forest(data, random_state, calibrated=False, ext_name=""):
    """
    Application of random forest classifier. For details look at
    'clf_sklearn' function.
    """

    rf = RandomForestClassifier(n_estimators=400, n_jobs=-1,

                                max_depth=19,
                                max_features=0.13,
                                min_samples_split=90,
                                random_state=random_state, verbose=10)

    return clf_sklearn(rf, data, random_state, calibrated, clf_name='RF',
                       ext_name=ext_name)



def clf_extra_trees(data, random_state, calibrated=False, ext_name=""):
    """
    Application of extra trees classifier. For details look at
    'clf_sklearn' function.
    """
    et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1,

                                max_depth=17,
                                max_features=0.2,
                                min_samples_split=80,
                                random_state=random_state, verbose=10)

    return clf_sklearn(et, data, random_state, calibrated, clf_name='ET',
                       ext_name=ext_name)


