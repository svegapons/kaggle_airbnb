""" 
Airbnb New User Bookings Comptetition
https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings

Author: Sandro Vega Pons (sv.pons@gmail.com)

Routines to do feature selection.
"""

import xgboost
import numpy as np
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.ensemble import RandomForestClassifier


def xgb_feat_selection(X_train, y_train, X_valid, y_valid, random_state):
    """
    Feature selection based on the scores given to the features by an 
    XGB Classifier.
    """
    #Parameters of the xgb classifier to be used for feature selection
    params = {'eta': 0.09,
              'max_depth': 6,
              'subsample': 0.5,
              'colsample_bytree': 0.5,
              'objective': 'multi:softprob',
              'eval_metric': 'mlogloss',
              'num_class': 12}
    num_rounds = 1000
    xg_train = xgboost.DMatrix(X_train, label=y_train)  
    xg_valid = xgboost.DMatrix(X_valid, label=y_valid)  
    watchlist = [(xg_train,'train'), (xg_valid, 'validation')]
    #Training the model and stopping at the best iteration
    xgb = xgboost.train(params, xg_train, num_rounds, watchlist,
                        early_stopping_rounds=10)
    #Getting the scores for each feature
    f_score = xgb.get_fscore()
    feats = np.zeros(X_train.shape[1])
    #Scores are given in the format => fn:x meaning n-th feature has a value x.
    for k,v in f_score.items():
        feats[int(k[1:])] = v
    #Normalizing the scores to [0,1.]
    feats = feats/float(np.max(feats))
    
    np.save('save/feat_sel_xgb.npy', feats)
    
    return feats
    
    

def log_reg_feat_selection(X_train, y_train, X_valid, y_valid, random_state):
    """
    Feature selection based on the scores given to the features by the 
    RandomizedLogisticRegression algorithm.
    """
    
    rlr = RandomizedLogisticRegression(C=[0.001, 0.01, 0.1, 1.], 
                                       sample_fraction=0.7,
                                       n_resampling=200, selection_threshold=0.25,
                                       verbose=5, n_jobs=-1, random_state=0)                                   
    rlr.fit(X_train, y_train)
    np.save('save/feat_sel_log_reg.npy', rlr.scores_)
    
    return rlr.scores_


def random_forest_selection(X_train, y_train, X_valid, y_valid, random_state):
    """
    Feature selection based on the scores given to the features by the 
    RandomForest algorithm.
    """
    rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, 
                                criterion='gini',
                                max_depth=15,
                                max_features=0.2,
                                max_leaf_nodes = 20,
                                min_samples_split=50,
                                random_state=random_state, verbose=10)
    rf.fit(X_train, y_train) 
    feat_imp = rf.feature_importances_
    np.save('save/feat_sel_random_forest.npy', feat_imp)
    
#    th = np.sort(feat_imp)[::-1][int(len(feat_imp)*7./10.)]
#    feats = feat_imp > th    
    
    return feat_imp
    
    
def apply_feat_sel(X_train, X_valid, X_test, f_type='xgboost', th=0.001):
    """
    Selects features with highest scores according to a precomputed feature
    scoring.
    
    Parameters:
    ----------
    X_train: training set
    X_valid: validation set
    X_test: test set
    f_type: string
          Type of feature scoring system to be used. Possible values are:
          xgboost, for feature selection based on xgboost scoring.
          log_reg, for feature selection based on logistic regression scoring.
    th: float
        Threshold. Features with a score higher than th are kept the others are
        discarded.
    
    Return:
    ------
    X_train, X_valid, X_test: Train, validation and test sets after feature extraction.
    """
    if f_type == 'xgboost':
        scores = np.load('save/feat_sel_xgb.npy')
    elif f_type == 'log_reg':
        scores = np.load('save/feat_sel_log_reg.npy')
    elif f_type == 'random_forest':
        scores = np.load('save/feat_sel_random_forest.npy')
        
    feats = scores > th
    
    X_train = X_train[:, feats]
    X_valid = X_valid[:, feats]
    X_test = X_test[:, feats]
    
    print 'Keeping %s features from the original %s' %(X_train.shape[1], feats.shape[0])
    
    return X_train, X_valid, X_test
    
    
def apply_feat_sel_by_percent(X_train, X_valid, X_test, f_type='xgboost', percent=0.7):
    """
    Selects features with highest scores according to a precomputed feature
    scoring. The top percent features are kept.
    
    Parameters:
    ----------
    X_train: training set
    X_valid: validation set
    X_test: test set
    f_type: string
          Type of feature scoring system to be used. Possible values are:
          xgboost, for feature selection based on xgboost scoring.
          log_reg, for feature selection based on logistic regression scoring.
    percent: float [0, 1]
        percent*100 gives the percent of features to keep.
    
    Return:
    ------
    X_train, X_valid, X_test: Train, validation and test sets after feature extraction.
    """       
    if f_type == 'xgboost':
        scores = np.load('save/feat_sel_xgb.npy')
    elif f_type == 'log_reg':
        scores = np.load('save/feat_sel_log_reg.npy')
    elif f_type == 'random_forest':
        scores = np.load('save/feat_sel_random_forest.npy')
        
    th = np.sort(scores)[::-1][int(len(scores)*7./10.)]   
    
    feats = scores > th
    
    X_train = X_train[:, feats]
    X_valid = X_valid[:, feats]
    X_test = X_test[:, feats]
    
    print 'Keeping %s features from the original %s' %(X_train.shape[1], feats.shape[0])
    
    return X_train, X_valid, X_test
    
    