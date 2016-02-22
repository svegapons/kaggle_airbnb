""" 
Airbnb New User Bookings Comptetition
https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings

Author: Sandro Vega Pons (sv.pons@gmail.com)

Main scripts to generate the submission.
"""

import numpy as np
from utils import process_raw_data, load_dataset, split_train_valid_test, load_train_valid_test, make_submission
from feat_selection import xgb_feat_selection, log_reg_feat_selection, random_forest_selection, apply_feat_sel
from code_xgboost import clf_xgboost
from code_keras import clf_keras
from code_2_step import clf_2_step
from code_sklearn import clf_random_forest, clf_extra_trees, clf_log_regression
from ensemble import apply_en_opt_1, apply_en_opt_2, apply_mix_opt
import pdb


def initial_processing():
    """
    """
    rs = np.random.RandomState(0)
    process_raw_data()
    df_all, id_test, target = load_dataset() 
    X_train, y_train, X_valid, y_valid, X_test, le = split_train_valid_test(df_all, target, ) 
    
    #Feature selection 
    xgb_feat_selection(X_train, y_train, X_valid, y_valid, rs)
    random_forest_selection(X_train, y_train, X_valid, y_valid, rs)
    log_reg_feat_selection(X_train, y_train, X_valid, y_valid, rs)
    

def new_first_level_clf():
    """
    """
    X_train, y_train, X_valid, y_valid, X_test, le = load_train_valid_test()
    


def main():
    """
    ET
    KE
    KE_w_1.5
    KE_w_1.
    ke
    ke
    RF
    XGB
    XGB_w_1.5
    XGB_w_1.
    xgb
    xgb
    """
    
    
    
    
    
    X_train, X_valid, X_test = apply_feat_sel(X_train, X_valid, X_test, 
                                              f_type='xgboost', th=0.001)



if __name__ == '__main__':
    """
    """
    rs = np.random.RandomState(0)
    
    ####Processing raw data and computing X_train,y_train,X_valid,y_valid,X_test####
    
        #First run ([X_train, y_train, X_valid, y_valid, X_test, le] is stored)
    process_raw_data()    
    df_all, id_test, target = load_dataset()   
    X_train, y_train, X_valid, y_valid, X_test, le = split_train_valid_test(df_all, target)   
    
        #From the second run on. The data is stored and there is no need to compute it again.
#    X_train, y_train, X_valid, y_valid, X_test, le = load_train_valid_test()
    
    #Feature selection methods
    
    feats = xgb_feat_selection(X_train, y_train, X_valid, y_valid, rs)
#    X_train, X_valid, X_test = apply_feat_sel(X_train, X_valid, X_test, 
#                                              f_type='xgboost', th=0.001)
#    print X_train.shape
#    
    #Preprocessing
#    X_train = np.log(1+np.where(X_train<0,0,X_train))
#    X_valid = np.log(1+np.where(X_valid<0,0,X_valid))
#    X_test = np.log(1+np.where(X_test<0,0,X_test))
#                                              
#    data = [X_train, y_train, X_valid, y_valid, X_test]

##    #Applying Keras classifier
#    clf_keras(data, rs, ext_name='ke1')
##    
##    #Applying XGBoost clasifier    
#    clf_xgboost(data, rs, ext_name='xgb1')
#
##    Applying ExtraTrees classifier
#    clf_extra_trees(data, rs, calibrated=False, ext_name='et1')
#
##    Applying ExtraTrees classifier
#    clf_random_forest(data, rs, calibrated=False, ext_name='rf1')
    
    #######################################################
#    Second round
    #######################################################
    
    rs = np.random.RandomState(12)
#    
    X_train, X_valid, X_test = apply_feat_sel(X_train, X_valid, X_test, 
                                              f_type='xgboost', th=0.001)
#    print X_train.shape
#    
#    #Preprocessing
    mn = abs(np.min(X_train))
    X_train = np.log(1+X_train+mn)
    X_valid = np.log(1+X_valid+mn)
    X_test = np.log(1+X_test+mn)
#    X_train = np.log(1+np.where(X_train<0,0,X_train))
#    X_valid = np.log(1+np.where(X_valid<0,0,X_valid))
#    X_test = np.log(1+np.where(X_test<0,0,X_test))
#                                              
    data = [X_train, y_train, X_valid, y_valid, X_test]
#    
#    np.save('y_valid.npy', y_valid)
#    y_valid = np.load('y_valid.npy')
##    
#    
#    #Classifiers with weights
##    cl_weight = {0:1.5, 1:1.5, 2:1.5, 3:1.5, 4:1.5, 5:1.5, 6:1.5, 7:1, 8:1.5, 9:1.5, 10:1.5, 11:1.5}
####    #Applying Keras classifier
##    clf_keras_weight(data, cl_weight, rs, ext_name='w_1.5')      
###    #Applying xgboost with weights
##    clf_xgboost_weight(data, cl_weight, rs, ext_name='w_1.5')
#    
#    cl_weight = {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1.5, 8:1, 9:1, 10:1, 11:1}     
#    clf_xgboost_weight(data, cl_weight, rs, ext_name='w_1.')  
#    clf_keras_weight(data, cl_weight, rs, ext_name='w_1.') 
    
    #Applying keras + logistic regression
#    clf_keras_lr(data, rs, ext_name='kelr')  
    
    #Applying clf_2_step
    clf_2_step(data, rs, ext_name='2st') 
    
#    apply_en_opt_1(y_valid)
    
#    apply_en_opt_2(y_valid, valid_folder='save/vv', test_folder='save/tt')
#    apply_en_opt_2(y_valid)
    
#    y_pred = apply_mix_opt(y_valid, valid_folder='save/vv', test_folder='save/tt')
#    make_submission(y_pred, le, sub_name='nc_sub.csv')
    
#    apply_log_reg_ens(y_valid)
    
    pdb.set_trace()
    
    
    
    