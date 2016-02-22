""" 
Airbnb New User Bookings Comptetition
https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings

Author: Sandro Vega Pons (sv.pons@gmail.com)

Ensemble techniques.
"""

import os
import numpy as  np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.base import BaseEstimator
from scipy.optimize import minimize
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelBinarizer
from letor_metrics import ndcg_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV


def get_X_X_Test(valid_folder='save/valid', test_folder='save/test'):
    """
    Auxiliary function to compute the new X and X_test from the data
    stored in the 'save/valid' and 'save/test' folders respectively. 
    """
    #Computing X from data in valid folder
    lv_path = os.listdir(valid_folder)
    lv_path.sort()
    print len(lv_path)
    list_valid = []        
    for p in lv_path:
        arr = pickle.load(open(os.path.join(valid_folder, p), 'r'))
        list_valid.append(arr)    
    X = np.hstack(list_valid)

    #Computing X_test from data in test folder
    lt_path = os.listdir(test_folder)
    lt_path.sort()
    list_test = []        
    for p in lt_path:
        arr = pickle.load(open(os.path.join(test_folder, p), 'r'))
        list_test.append(arr)    
    X_test = np.hstack(list_test)
    
    #
    n_preds = len(list_valid)
    n_class = list_valid[0].shape[1]
    
    return X, X_test, n_preds, n_class
    
    
##############################################
##Optimization based ensemblers
#############################################

def opt_1_obj_func(w, X, y, n_class):
    """
    Function to be minimized in the EN_OPT_1 ensembler.
    Parameters:
    ----------
    w: ndarray size=(n_preds * n_class)
       Candidate solution to the optimization problem (vector of weights).
    X: ndarray size=(n_samples, n_preds * n_class)
       Solutions to be combined horizontally concatenated.
    y: ndarray size=(n_samples,)
       Class labels
    n_class: int
       Number of classes in the problem, i.e. = 12
    """
    #Constraining the weights for each class to sum 1.
    #This constrain can be defined in the scipy.minimize function, but doing it here
    #gives more flexibility to the scipy.minimize function (e.g. more solvers 
    #are allowed).
    w_range = np.arange(len(w))%n_class 
    for i in range(n_class): 
        w[w_range==i] = w[w_range==i] / np.sum(w[w_range==i])
        
    sol = np.zeros((X.shape[0], n_class))
    for i in range(len(w)):
        sol[:, i % n_class] += X[:, i] * w[i]
    #The quantity to minimize is the log_loss.     
    sc_ll = log_loss(y, sol)
    return sc_ll
    

class EN_OPT_1(BaseEstimator):
    """
    Ensembler. There is a weight for each class for each prediction.
    Given a set of candidate solutions x1, x2, ..., xn, where each xi has
    m=12 predictions (one for each class), i.e. xi = xi1, xi2,...,xim. The 
    algorithms finds the optimal set of weights w1, w2, ..., w(n*m); such that 
    minimizes log_loss(y_true, x_ensemble), where x_ensemble = x11*w1 + x12*w2 +
    ...+ x21*w(m+1) +... xnm*w(n*m).
    """
    def __init__(self, n_preds, n_class):
        super(EN_OPT_1, self).__init__()
        self.n_preds = n_preds
        self.n_class = n_class
        
    def fit(self, X, y):
        """
        Learn the optimal weights by solving an optimization problem.
        """
        x0 = np.ones(self.n_class * self.n_preds) / float(self.n_preds) 
        bounds = [(0,1)]*len(x0)   
        res = minimize(opt_1_obj_func, x0, args=(X, y, self.n_class), 
                       method='L-BFGS-B', 
                       bounds=bounds, 
                       )
        self.w = res.x
        return self
    
    def predict_proba(self, X):
        """
        Use the weights learned in training to predict class probabilities.
        """
        y_pred = np.zeros((X.shape[0], self.n_class))
        for i in range(len(self.w)):
            y_pred[:, i % self.n_class] += X[:, i] * self.w[i]   
        return y_pred      
        

def apply_en_opt_1(y_valid, valid_folder='save/valid', test_folder='save/test'):
    """
    Applies the EN_OPT_1 ensembler to the solutions stored in 'save/valid' and 
    'save/test'.
    """
    #Loading data
    X, X_test, n_preds, n_class = get_X_X_Test(valid_folder, test_folder)
    y = y_valid
   
    validation = False
    if validation:
        #This is just to test the results of the ensemble on validation data.
        sss = StratifiedKFold(y, n_folds=5, random_state=0)
        for id_train, id_valid in sss:
            X_train, X_valid = X[id_train], X[id_valid]
            y_train, y_valid = y[id_train], y[id_valid]
            
            lb = LabelBinarizer()
            lb.fit(y_train)
            yb_valid = lb.transform(y_valid)
            
            print 'Individual results'
            for i in range(n_preds):
                sing_pred = X_valid[:,i*n_class:(i+1)*n_class]
                score_sing_pred= np.mean([ndcg_score(tr, pr, k=5) for tr, pr in zip(yb_valid.tolist(), sing_pred.tolist())])
                print score_sing_pred, log_loss(y_valid, sing_pred)
            print ' '
            
            ec1 = EN_OPT_1(n_preds, n_class)
            ec1.fit(X_train, y_train)
            y_ec_pred = ec1.predict_proba(X_valid)
            print np.round(ec1.w.reshape(n_preds, -1), 2)
            sc_ec = np.mean([ndcg_score(tr, pr, k=5) for tr, pr in zip(yb_valid.tolist(), y_ec_pred.tolist())])              
         
            print 'Result of EC1'    
            print sc_ec, log_loss(y_valid, y_ec_pred)
            print '------ \n'
    else:
        #Applying the EN_OPT_1 ensembler
        ens = EN_OPT_1(n_preds, n_class)    
        ens.fit(X, y)
        y_pred = ens.predict_proba(X_test)

    return y_pred



def opt_2_obj_func(w, X, y, n_class):
    """
    Function to be minimized in the EN_OPT_2 ensembler.
    In this case there is only one weight for each classification restlt to be 
    combined.
    Parameters:
    ----------
    w: ndarray size=(n_preds)
       Candidate solution to the optimization problem (vector of weights).
    X: ndarray size=(n_samples, n_preds * n_class)
       Solutions to be combined horizontally concatenated.
    y: ndarray size=(n_samples,)
       Class labels
    n_class: int
       Number of classes in the problem, i.e. = 12
    """
    w = np.abs(w)
    sol = np.zeros((X.shape[0], n_class))
    for i in range(len(w)):
        sol += X[:, i*n_class:(i+1)*n_class] * w[i]
    #Minimizing the logloss   
    sc_ll = log_loss(y, sol)
    return sc_ll     
        

class EN_OPT_2(BaseEstimator):
    """
    Ensembler. There is only one weight for each independent solution.
    Given a set of candidate solutions x1, x2, ..., xn; it computes
    the optimal set of weights w1, w2, ..., wn; such that minimizes 
    log_loss(y_true, x_ensemble), where x_ensemble = x1*w1 + x2*w2 +..., xn*wn.
    """
    def __init__(self, n_preds, n_class):
        super(EN_OPT_2, self).__init__()
        self.n_preds = n_preds
        self.n_class = n_class
        
    def fit(self, X, y):
        """
        Learn the optimal weights by solving an optimization problem.
        """
        x0 = np.ones(self.n_preds) / float(self.n_preds) 
        bounds = [(0,1)]*len(x0)   
        cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
        res = minimize(opt_2_obj_func, x0, args=(X, y, self.n_class), 
                       method='SLSQP', 
                       bounds=bounds,
                       constraints=cons
                       )
        self.w = res.x
        return self
    
    def predict_proba(self, X):
        """
        Use the weights learned in training to predict class probabilities.
        """
        y_pred = np.zeros((X.shape[0], self.n_class))
        for i in range(len(self.w)):
            y_pred += X[:, i*self.n_class:(i+1)*self.n_class] * self.w[i] 
        return y_pred  
        
        
def apply_en_opt_2(y_valid, valid_folder='save/valid', test_folder='save/test'):
    """
    Applies the EN_OPT_2 ensembler to the solutions stored in 'save/valid' and 
    'save/test'.
    """
    #Loading data
    X, X_test, n_preds, n_class = get_X_X_Test(valid_folder, test_folder)
    y = y_valid
    
    validation = False
    if validation:
        #This is just to test the results of the ensemble on validation data.
        sss = StratifiedKFold(y, n_folds=5, random_state=0)
        for id_train, id_valid in sss:
            X_train, X_valid = X[id_train], X[id_valid]
            y_train, y_valid = y[id_train], y[id_valid]
            
            lb = LabelBinarizer()
            lb.fit(y_train)
            yb_valid = lb.transform(y_valid)
            
            print 'Individual results'
            for i in range(n_preds):
                sing_pred = X_valid[:,i*n_class:(i+1)*n_class]
                score_sing_pred= np.mean([ndcg_score(tr, pr, k=5) for tr, pr in zip(yb_valid.tolist(), sing_pred.tolist())])
                print score_sing_pred, log_loss(y_valid, sing_pred)
            print ' '
            
            ec2 = EN_OPT_2(n_preds, n_class)          
            ec2.fit(X_train, y_train)
            y_ec_pred = ec2.predict_proba(X_valid)
            print np.round(ec2.w.reshape(n_preds, -1), 2)
            sc_ec = np.mean([ndcg_score(tr, pr, k=5) for tr, pr in zip(yb_valid.tolist(), y_ec_pred.tolist())])              
            
            print 'Result of EC1'    
            print sc_ec, log_loss(y_valid, y_ec_pred)            
            print '------ \n'
        
    else:
        #Applying the EN_OPT_2 ensembler
        ens = EN_OPT_2(n_preds, n_class)    
        ens.fit(X, y)
        y_pred = ens.predict_proba(X_test)

    return y_pred  
    
    
    
def apply_mix_opt(y_valid, valid_folder='save/valid', test_folder='save/test'):
    """
    Ensembler that is a combination of EN_OPT_1 and EN_OPT_2 + its calibrated
    versions. If 
    en1 => prediction of EN_OPT_1
    cal_en1 => prediction of calibrated EN_OPT_1
    en2 => prediction of EN_OPT_2
    cal_en2 => prediction of EN_OPT_2
    then, the prediction is ((en1*2 + cal_en1)/3 + ((en2*2 + cal_en2)/3)*2)/3
    """
    #Loading data
    X, X_test, n_preds, n_class = get_X_X_Test(valid_folder, test_folder)
    y = y_valid
    
    validation = False
    if validation:
        #This is just to test the results of the ensemble on validation data.
        sss = StratifiedKFold(y, n_folds=5, random_state=0)
        for id_train, id_valid in sss:
            X_train, X_valid = X[id_train], X[id_valid]
            y_train, y_valid = y[id_train], y[id_valid]
            
            lb = LabelBinarizer()
            lb.fit(y_train)
            yb_valid = lb.transform(y_valid)
            
            print 'Individual results'
            for i in range(n_preds):
                sing_pred = X_valid[:,i*n_class:(i+1)*n_class]
                score_sing_pred= np.mean([ndcg_score(tr, pr, k=5) for tr, pr in zip(yb_valid.tolist(), sing_pred.tolist())])
                print score_sing_pred, log_loss(y_valid, sing_pred)
            print ' '
            
            ec1 = EN_OPT_1(n_preds, n_class)        
            ec1.fit(X_train, y_train)
            y_ec1_pred = ec1.predict_proba(X_valid)
            
            cc_ec1 = CalibratedClassifierCV(base_estimator=ec1, method='isotonic', cv=3)
            cc_ec1.fit(X_train, y_train)
            y_cc1_pred = cc_ec1.predict_proba(X_valid)
            y_E1_pred = (y_ec1_pred*2 + y_cc1_pred) / 3.
            
            ec2 = EN_OPT_2(n_preds, n_class)        
            ec2.fit(X_train, y_train)
            y_ec2_pred = ec2.predict_proba(X_valid)
            
            cc_ec2 = CalibratedClassifierCV(base_estimator=ec2, method='isotonic', cv=3)
            cc_ec2.fit(X_train, y_train)
            y_cc2_pred = cc_ec2.predict_proba(X_valid)
            y_E2_pred = (y_ec2_pred*2 + y_cc2_pred) / 3.
            
            y_ec_v1_pred = (y_E1_pred + y_E2_pred) / 2.
            y_ec_v2_pred = (y_E1_pred*2 + y_E2_pred) / 3.
            y_ec_v3_pred = (y_E1_pred + y_E2_pred*2) / 3.
            
            sc_ec_v1 = np.mean([ndcg_score(tr, pr, k=5) for tr, pr in zip(yb_valid.tolist(), y_ec_v1_pred.tolist())])              
            sc_ec_v2 = np.mean([ndcg_score(tr, pr, k=5) for tr, pr in zip(yb_valid.tolist(), y_ec_v2_pred.tolist())])              
            sc_ec_v3 = np.mean([ndcg_score(tr, pr, k=5) for tr, pr in zip(yb_valid.tolist(), y_ec_v3_pred.tolist())])              
         
            print 'Results'    
            print sc_ec_v1, log_loss(y_valid, y_ec_v1_pred)
            print sc_ec_v2, log_loss(y_valid, y_ec_v2_pred)
            print sc_ec_v3, log_loss(y_valid, y_ec_v3_pred)            
            print '------ \n'
        
    
    else:
        #Applying the ensembler...     
    
        ec1 = EN_OPT_1(n_preds, n_class)        
        ec1.fit(X, y)
        y_ec1_pred = ec1.predict_proba(X_test)
        print '.'
        
        cc_ec1 = CalibratedClassifierCV(base_estimator=ec1, method='isotonic', cv=3)
        cc_ec1.fit(X, y)
        y_cc1_pred = cc_ec1.predict_proba(X_test)
        y_E1_pred = (y_ec1_pred*2 + y_cc1_pred) / 3.   
        print '..'
        
        ec2 = EN_OPT_2(n_preds, n_class)        
        ec2.fit(X, y)
        y_ec2_pred = ec2.predict_proba(X_test)
        print '...'
        
        cc_ec2 = CalibratedClassifierCV(base_estimator=ec2, method='isotonic', cv=3)
        cc_ec2.fit(X, y)
        y_cc2_pred = cc_ec2.predict_proba(X_test)
        y_E2_pred = (y_ec2_pred*2 + y_cc2_pred) / 3.
        print '....'
        
        y_pred = (y_E1_pred + y_E2_pred*2) / 3.
        
        return y_pred  


##################################
##Classification based ensemblers
#################################
#They produced much poorer results than the optimization based ensemblers.

def apply_log_reg_ens(y_valid, valid_folder='save/valid', test_folder='save/test'):
    """
    Ensembler based on logistic regression.
    """
    #Loading data
    X, X_test, n_preds, n_class = get_X_X_Test(valid_folder, test_folder)
    y = y_valid
    
    #Defining classifier
    lr = LogisticRegression(penalty='l2', C=0.01, multi_class='ovr', 
                            max_iter=300, solver='lbfgs',n_jobs=-1)  

    lr.fit(X, y)   
    y_pred = lr.predict_proba(X_test)
    return y_pred
    

def apply_xgb_ens(y_valid, valid_folder='Valid', test_folder='Test'):
    """
    Ensembler based on xgboost Gradient boosting.
    """
    #Loading data
    X, X_test, n_preds, n_class = get_X_X_Test(valid_folder, test_folder)
    y = y_valid
    
    #Defining classifier
    xgb = XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=200,
                        objective='multi:softprob', gamma=0., 
                        max_delta_step=0., subsample=0.9, colsample_bytree=0.9,
                        seed=0)  
    xgb.fit(X, y)   
    y_pred = xgb.predict_proba(X_test)
    return y_pred      
    
    
    