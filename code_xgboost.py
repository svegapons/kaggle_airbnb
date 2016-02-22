""" 
Airbnb New User Bookings Comptetition
https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings

Author: Sandro Vega Pons (sv.pons@gmail.com)

Classifiers based on xgboost code.
"""


import numpy as  np
import pickle
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.metrics import log_loss
from letor_metrics import ndcg_score
from sklearn.utils import compute_sample_weight


def clf_xgboost(data, cl_weight=None, random_state=0, ext_name="", verbose=True):
    """
    XGBoost classifier
    The function applies the classifier twice:
    - First: Fit the classifier to (X_train, y_train) and predict on (X_valid).
             The prediction is stored in 'save/valid' folder.
    - Second: Fit the classifier to (X, y) = (X_train + X_valid, y_train + y_valid)
             and predict on (X_test). The prediction is stored in 'save/test' 
             folder.
             
    Parameters:
    ----------
    data: list
         [X_train, y_train, X_valid, y_valid, X_test]
    cl_weight: None or Dictionary
         Class weights, e.g. {0:1, 1:1.5, 2:1.6...} => weight for class 0 is 1, 
         for class 1 is 1.5, for class 2 is 1.6, and so on.
    random_state: numpy RandomState
         RandomState used for reproducibility
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
     
    xgb = XGBClassifier(max_depth=6, learning_rate=0.01, n_estimators=10000,
                    objective='multi:softprob', gamma=1., min_child_weight=1.,
                    max_delta_step=5., subsample=0.7, colsample_bytree=0.7,
                    reg_alpha=0., reg_lambda=1., seed=random_state)  
                    
    X_train, y_train, X_valid, y_valid, X_test = data                
    
    ###Working on (X_Train => X_Valid)###
    ss = StandardScaler()
    XX_train = ss.fit_transform(X_train)
    XX_valid = ss.transform(X_valid)
    
    lb = LabelBinarizer()
    lb.fit(y_train)
    yb_valid = lb.transform(y_valid)
        
    if cl_weight == None:
        xgb.fit(XX_train, y_train, 
                eval_set=[(XX_valid, y_valid)],
                eval_metric = 'mlogloss',
                early_stopping_rounds=25, verbose=verbose)
    else:   
        #Computing sample weights from class weights
        sw_train = compute_sample_weight(class_weight=cl_weight, y=y_train)    
        xgb.fit(XX_train, y_train, 
                sample_weight=sw_train,
                eval_set=[(XX_valid, y_valid)],
                eval_metric = 'mlogloss',
                early_stopping_rounds=25, verbose=verbose) 
        
    best_iter = xgb.best_iteration
    y_valid_pred = xgb.predict_proba(XX_valid, ntree_limit = best_iter)

    ndcg_xg = np.mean([ndcg_score(tr, pr, k=5) for tr, pr in zip(yb_valid.tolist(), y_valid_pred.tolist())])
    print 'NDCG: %s' %(ndcg_xg)
    logloss_xg = log_loss(y_valid, y_valid_pred)
    print 'Log-loss: %s' %(logloss_xg)
    
    rnd = random_state.randint(1000, 9999)
    pickle.dump(y_valid_pred, open('save/valid/v_XGB_%s_%s_%s_%s'%(ext_name, rnd, round(ndcg_xg, 4), round(logloss_xg, 4)), 'w'))
    
    ###Working on X => X_test###
    X = np.vstack((X_train, X_valid))
    y = np.hstack((y_train, y_valid))
    
    XX = ss.fit_transform(X)
    XX_test = ss.transform(X_test)
        
    xgb.n_estimators = best_iter + 20  
    
    if cl_weight == None:
        xgb.fit(XX, y)
    else:
        sw = compute_sample_weight(class_weight=cl_weight, y=y)
        xgb.fit(XX, y, sample_weight=sw)
        
    y_test_pred = xgb.predict_proba(XX_test)
    
    pickle.dump(y_test_pred, open('save/test/t_XGB_%s_%s'%(ext_name, rnd), 'w'))
    
    return y_valid_pred, y_test_pred

