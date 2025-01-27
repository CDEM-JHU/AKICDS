#%% AKI DECISION SUPPORT
###############################################################################
# LOAD PACKAGES 
import subprocess
import sys
def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import sklearn

    sk_version = sklearn.__version__
    if sk_version == "1.4.0":
        print("sklearn version matched")
    else:
        subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
except ImportError:
    print("sklearnnot installed")
           
try:
    import xgboost 
    xg_version = xgboost.__version__
    if xg_version == "2.0.3":
        print("xgboost version matched")
    else:
        subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
except ImportError:
    print("xgboost not installed")
    
import time
from turtle import color #, os
tic = time.process_time()
import pandas as pd
import numpy as np
import sklearn
import pickle


import getpass
import matplotlib as mlp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import date, timedelta
from openpyxl import load_workbook

import statsmodels

from xgboost import XGBClassifier
from sklearn.metrics import classification_report as cr, brier_score_loss, precision_recall_curve,roc_auc_score,roc_curve
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression as LR

from datetime import datetime
import seaborn as sns
import scipy

import os, sys, getopt
import glob
import shutil
import logging
import datetime
import configparser



toc = time.process_time()
print ((toc-tic)/60,' minutes to import tools')
del tic, toc


# controller ##################################################################
tspecs = {}
tspecs['version'] = 'v_2_9'

##############################################################################

# establish environment
main_path = '\\'
ed_path = '\\'
model_path = '\\'




#%% 
# READ NORMALIZED DATA 


##############################################################################

read_pickle_data = 'no'

version = 'v_2_9'
pickle_path = '\\'
##############################################################################
if read_pickle_data == 'yes':
    enc_set = pickle.load(open(model_path + 'enc_set_'+version  + '.pickle', 'rb'))
    

#%% Models and AUC FUNCTIONS
def compute_ground_truth_statistics(ground_truth, sample_weight=None):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight

# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.

    Args:
       x - a 1D numpy array
    Returns:
       array of midranks

    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2



def fastDeLong(predictions_sorted_transposed, label_1_count):
    """Fast DeLong test computation.

    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }

    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


# this creates auc confidence intervals
def delong_roc_variance(ground_truth, predictions):
    """Computes ROC AUC variance for a single set of predictions.

    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1

    """
    sample_weight = None
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    #aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov  

def auc_ci(enc_use,ypred):
    auc_delong, auc_cov_delong = delong_roc_variance(enc_use,ypred) #SL
    auc_std = np.sqrt(auc_cov_delong) #SL
    alpha = 0.95 #SL
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2) #SL
    auc_ci_total = scipy.stats.norm.ppf(lower_upper_q, loc=auc_delong, scale=auc_std) #SL
    auc_ci_output = str(round(auc_delong,2))+'(' + str(round(auc_ci_total[0],5))+ '-'+str(round(auc_ci_total[1],5)) + ')'
    return auc_ci_output

def remove_zero(data,pred_list):
    pred_sum = data[pred_list].sum() # this line is taking forever - check datatyping
    pred_sum = pred_sum.loc[pred_sum == 0]
    pred_remove = list(pred_sum.index)
    if len(pred_remove) > 0:
        for pred in pred_remove:
            pred_list.remove(pred)
    return pred_list

def dummy(data,col,ref_var='no'):
    d = pd.get_dummies(data[col])
    for i in d.columns:
        i = int(i)
        d = d.rename(columns={i:col+'_'+str(i)})
    if ref_var !='no':
        del d[col+'_'+str(ref_var)]
    return d

def remove_ref_category(plist,pred_dummy,enc_set):
    for pred in plist:
        if pred in ['age']: 
            pred_dummy = pd.concat([pred_dummy,dummy(enc_set,pred,ref_var=1)],axis=1) 
        elif pred in ['sex']:
            pass
        elif pred in ['ethnicity']:
            pass
        elif pred in ['race']:
            pass
        elif pred in rf_pred['pred_labs'] + rf_pred['pred_labs_more']:
            pred_dummy = pd.concat([pred_dummy,ordinal_encoder(enc_set,pred,var_type='lab')],axis=1) # ordinal encode lab
        elif pred in rf_pred['pred_vitals_triage']:
            pred_dummy = pd.concat([pred_dummy,ordinal_encoder(enc_set,pred,var_type='vital')],axis=1) # ordinal encode vital
        # elif pred in rf_pred['pred_labs'] + rf_pred['pred_labs_more'] +  rf_pred['pred_vitals_triage']:
        #     pred_dummy = pd.concat([pred_dummy,dummy(enc_set,pred,ref_var=0)],axis=1)
        elif pred in pred_library['pred_comps'] + pred_library['pred_probs']: # binary variables generated from norm code in SQL
            pred_dummy[pred] = enc_set[pred]
        else: 
            pred_dummy[pred] = enc_set[pred]
    return pred_dummy


def ordinal_encoder(data, col, var_type):
    d = data[[col]]
    if var_type == 'vital':
        missing_label = d[col].max()
        d = d.replace(missing_label, 0)
    elif var_type == 'lab':
        d[col] = d[col].fillna(0) 
    d.loc[:,col+'_missing'] = d[col].apply(lambda x:1 if x==0 else 0)
    d =  d.rename(columns={col:col+'_ord'})
    return d



def cross_split(enc_set,split_date,split_type='random'):
        if split_type == 'random':
            idx_train, idx_test = sklearn.model_selection.train_test_split(enc_set.index, test_size = 0.33, random_state = 33)
        elif split_type == 'time':
            idx_train = pd.Series(enc_set['arrdt'] < split_date)
            idx_train = idx_train[idx_train].index
            idx_test = pd.Series(enc_set['arrdt'] >= split_date)
            idx_test = idx_test[idx_test].index
        elif split_type == 'spatial':
            idx_train = enc_set.loc[enc_set['hosp'].isin(['JHH','HCGH','SMH'])].index
            idx_test = enc_set.loc[enc_set['hosp'].isin(['BMC','SH'])].index    
        print('train % ' + str(float(len(idx_train))/(len(idx_train)+len(idx_test))))

        return idx_train, idx_test


def specify_predictors(df, data_type='continuous'):
    if data_type == 'categorical': # missing data -> specific category
        pred_library = {}
        pred_library['pred_demographics'] = ['age_raw','sex','race','ethnicity']
        pred_library['pred_comps'] = ['comp_'+str(i) for i in range(1,64)]
        pred_library['pred_probs'] = ['cevd','hp','mld','msld','diab','diabwc','mrend','srend','canc','metacanc','aids_hiv','ami','cpd','chf','pvd','pud','dementia','rheumd', 'AKI_PHX_1YEAR']
        pred_library['pred_vitals_triage'] = ['dbp_first_label','sbp_first_label','spo2_first_label','pulse_first_label','resp_first_label','temp_first_label']
        pred_library['pred_labs'] = [col for col in df.columns if col.endswith('_last_label') and not col.startswith(('sbp','dbp','temp','spo2','pulse','resp'))]
        pred_library['pred_aki'] = ['baseline_cr_exist', 'baseline_cr','aki_init_ed_stage']
        pred_library['outcomes'] = ['out_aki', 'out_aki_greater_than_2']
        pred_library['missing'] = ['out_aki_missing', 'out_ckd_missing']
        pred_library['pred_labs_more'] =[col for col in df.columns if  col.endswith(('_min_label','_max_label')) and not col.startswith(('sbp','dbp','temp','spo2','pulse','resp'))]
        
    elif data_type == 'continuous': # has missing data in predictors
        pred_library = {}
        pred_library['pred_demographics'] = ['age_raw','sex','race','ethnicity']
        pred_library['pred_comps'] = ['comp_'+str(i) for i in range(1,64)]
        pred_library['pred_probs'] = ['cevd','hp','mld','msld','diab','diabwc','mrend','srend','canc','metacanc','aids_hiv','ami','cpd','chf','pvd','pud','dementia','rheumd', 'AKI_PHX_1YEAR']
        pred_library['pred_vitals_triage'] = ['dbp_first','sbp_first','spo2_first','pulse_first','resp_first','temp_first']
        pred_library['pred_labs'] = [col for col in df.columns if col.endswith('_last') and not col.endswith('_last_label') and not col.startswith(('sbp','dbp','temp','spo2','pulse','resp'))]
        pred_library['pred_labs_more'] =[col for col in df.columns if col.endswith(('_min','_max')) and not col.endswith(('_min_label','_max_label')) and not col.startswith(('sbp','dbp','temp','spo2','pulse','resp'))]
        pred_library['outcomes'] = ['out_aki', 'out_aki_greater_than_2']
        pred_library['pred_aki'] = ['baseline_cr_exist', 'baseline_cr','aki_init_ed_stage']
        pred_library['missing'] = ['out_aki_missing', 'out_ckd_missing']
    return pred_library


#%%
# step 1: Complete Case

import warnings
warnings.filterwarnings("ignore")

write_analytics_pickles = 'yes'
write_pickles_path = '\\'

build_cc = 'yes'
outcomes = ['out_aki', 'out_aki_greater_than_2']


if build_cc == 'yes':
    # retrieve predictor list
    pred_library = specify_predictors(enc_set,'continuous')
    rf_pred= specify_predictors(enc_set,'categorical')
    cols_to_fill = rf_pred['pred_vitals_triage'] + rf_pred['pred_labs'] + rf_pred['pred_labs_more']
    enc_set[cols_to_fill] = enc_set[cols_to_fill].fillna(0)

    pred_library['pred_comps'] = remove_zero(enc_set,pred_library['pred_comps'])
    pred_library['pred_probs'] = remove_zero(enc_set,pred_library['pred_probs'])
    plist = pred_library['pred_demographics'] + pred_library['pred_vitals_triage'] + pred_library['pred_labs'] + pred_library['pred_aki'] + pred_library['pred_probs'] + pred_library['pred_comps'] + pred_library['pred_labs_more']

    # get prediction dummy variables
    pred_dummy = dummy(enc_set,'sex',ref_var=1)
    pred_dummy = remove_ref_category(plist,pred_dummy,enc_set)


    # get complete case dataframe: enc_cc
    enc_cc = enc_set.loc[enc_set['out_aki_missing'] == 0].copy()#.reset_index(drop=True)
    print("Complete Case: "+str(len(enc_cc)) + "/"+str(len(enc_set)) + "("+str(len(enc_cc)*100/len(enc_set))+"%)")
    
    pred_dummy_cc = pred_dummy.loc[enc_set['out_aki_missing'] == 0].copy()#.reset_index(drop=True)
    
    # Hyper parameter tuning result
    best_params = {'subsample': 0.2,
                               'reg_lambda': 1, 
                               'reg_alpha': 0.01, 
                               'objective': 'binary:logistic', 
                               'n_estimators': 300, 
                               'max_depth': 15, 
                               'learning_rate': 0.03, 
                               'gamma': 5, 
                               'eval_metric': 'logloss', 
                               'colsample_bytree': 0.7}

    XGBCC = {}
    
    for outcome in outcomes:
        # TRAIN xgboost model on Complete Case ################################################################
        XGBCC[outcome] = XGBClassifier(**best_params)
        XGBCC[outcome].fit(pred_dummy_cc, enc_cc[outcome])
        
        
    # MAKE PREDICTIONS ON IC
    pred_dummy_ic = pred_dummy.loc[enc_set['out_aki_missing'] == 1].copy()
    print("Incomplete Case: "+str(len(pred_dummy_ic)) + "/"+str(len(enc_set)) + "("+str(len(pred_dummy_ic)*100/len(enc_set))+"%)")
    
    y_pred_ic = pd.DataFrame(np.zeros(shape=(len(pred_dummy_ic),2)), columns = [outcomes])
    y_pred_ic.index = pred_dummy_ic.index
    
    for outcome in outcomes:
        y_pred_ic.loc[pred_dummy_ic.index,outcome] = XGBCC[outcome].predict_proba(pred_dummy_ic.loc[pred_dummy_ic.index])[:,1]
        
    
if write_analytics_pickles == 'yes':
    file = open(write_pickles_path + 'y_pred_ic_' + version + '.pickle','wb')
    pickle.dump(y_pred_ic,file)
    file.close()
    
    file = open(write_pickles_path + 'XGBCC_' + version + '.pickle','wb')
    pickle.dump(XGBCC,file)
    file.close()
    
    

#%%
# STEP 2: Imputation
# Create 20 datasets combined with Complete Case with EXACT outcomes and Imcomplete Case with imputed outcomes
# First: Create a imputated outcome dataframe, with dimension as N*(M+1), N is the size of incomplete case,
#       M is the # of imputations. First Column should be the predicted probabilities given by XGBOOST for each index
# Second: In each iteration, outcomes should be added to a new ic pred_dummy, and then concat to a complete case pred_dummy
# Should always has indicator of IC and CC
build_imp = 'yes'


n_imputations = 20

if build_imp == 'yes':
    
    for outcome in outcomes:
        if outcome == 'out_aki':
            path_add = 'imputed_123\\'
        elif outcome == 'out_aki_greater_than_2':
            path_add = 'imputed_23\\'
        else:
            print("Error: not valid outcome : " + str(outcome))

        # create imputed out
        imp_cols = ['pred']
        for i in range(n_imputations):
            imp_cols.append('outcome_'+str(i))
            
        imputed_ic = pd.DataFrame(np.zeros(shape=(len(pred_dummy_ic),n_imputations+1)), columns = imp_cols)
        imputed_ic.index = pred_dummy_ic.index
        imputed_ic['pred'] = y_pred_ic[outcome].copy()
        
        # impute
        for i in range(n_imputations):
            
            imputed_ic['outcome_'+str(i)] = (np.random.rand(len(imputed_ic['pred'])) < imputed_ic['pred']).astype(int)

        # concat CC and IC
        ori_cc = pred_dummy_cc.copy()
        ori_cc[outcome] = enc_cc[outcome].copy()
        ori_cc['IC'] = 0
        
        for i in range(n_imputations):
            imp_ic = pred_dummy_ic.copy()#.reset_index(drop=True)
            imp_ic[outcome] = imputed_ic['outcome_'+str(i)].copy()#.reset_index(drop=True)
            imp_ic['IC'] = 1

            imp_df = pd.concat([ori_cc, imp_ic], axis=0).sort_index()#.reset_index(drop=True)

            file = open(write_pickles_path + path_add + 'IMPUTED_DATAFRAME_'+outcome+'__'+str(i)+'_' + version + '.pickle','wb')
            pickle.dump(imp_df,file)
            file.close()


        # aki incidence rate
        temp_adm = imp_df.iloc[enc_set[enc_set['ed_disposition_aki']=='Admit'].index].copy()
        temp_dis = imp_df.iloc[enc_set[enc_set['ed_disposition_aki']=='Discharge'].index].copy()

        print(temp_adm[temp_adm['IC']==0][outcome].sum() * 100 / len(temp_adm[temp_adm['IC']==0]))
        # Complete Case, Admit

        print(temp_dis[temp_dis['IC']==0][outcome].sum() * 100 / len(temp_dis[temp_dis['IC']==0]))
        # Complete Case, Discharge

        print(imp_df[imp_df['IC']==1][outcome].sum() * 100 / len(imp_df[imp_df['IC']==1]))
        # Incomplete Case 

        print(imp_df[outcome].sum() * 100 / len(imp_df))
        # Imputed (CC + IC) 
#%%
# STEP 3: TRAIN MODELS ON EACH DATASET
# For each imputed dataset. train a XGBoost model on the dataset using 10-Fold Cross-Validation 
# Use external validation set to assess each model's performance.

import warnings
warnings.filterwarnings("ignore")

build_model = 'no'
write_analytics_pickles = 'no'
write_pickles_path = '\\'
read_analytics_pickles = 'no'
read_pickles_path = '\\'

n_imputations = 20
cv= 10

if build_model == 'yes':


    model_store = {}
    auc_metrics = pd.DataFrame(0, index=range(n_imputations), columns=[['train','train_ci','test','test_ci','cv','cv_ci','cvev','cvev_ci']])
    for i in range(n_imputations):
        print("Processing model " + str(i))

        #read dataframe
        if read_analytics_pickles == 'yes':
            enc_df = pickle.load(open(model_path + 'IMPUTED_DATAFRAME_'+str(i)+'_' + version  + '.pickle', 'rb'))

        # split data into idx_train for cross-validation and idx_test for external validation
        idx_train, idx_test = sklearn.model_selection.train_test_split(enc_df.index, test_size=0.33, random_state=2024+i)

        X_train, X_test = enc_df.iloc[:,:-2].loc[idx_train], enc_df.iloc[:,:-2].loc[idx_test]
        y_train, y_test = enc_df.iloc[:,-2].loc[idx_train], enc_df.iloc[:,-2].loc[idx_test]
        print("data prepared")

        xg = {}
        y_pred = pd.DataFrame(np.zeros(shape=(len(enc_df),1)), columns = ['out_aki'])
        y_pred_cv = pd.DataFrame(np.zeros(shape=(len(enc_df),1)), columns = ['out_aki'])
                
        model_name = 'model_'+str(i)
        best_params = {'subsample': 1,
                        'reg_lambda': 1,
                        'reg_alpha': 0.01,
                        'objective': 'binary:logistic',
                        'n_estimators': 200,
                        'max_depth': 5,
                        'learning_rate': 0.1,
                        'gamma': 0,
                        'eval_metric': 'logloss',
                        'colsample_bytree': 0.3}
        # Train test
        xg['standard'] = XGBClassifier(**best_params)
        xg['standard'].fit(X_train, y_train)
        y_pred['out_aki'].loc[idx_train]= xg['standard'].predict_proba(X_train)[:,1]
        # train eval
        xg['auc_train'] = roc_auc_score(y_train,y_pred['out_aki'].loc[idx_train])
        fpr_rf, tpr_rf, _ = sklearn.metrics.roc_curve(y_train,y_pred['out_aki'].loc[idx_train])
        xg['rocdata_train'] = pd.DataFrame({'fpr':fpr_rf,'tpr':tpr_rf})
        # calculate confidence interval of AUC using delong
        ci_output = auc_ci(y_train,y_pred['out_aki'].loc[idx_train])
        xg['auc_ci_train'] = ci_output


        # External validation - validation set################################################################
        y_pred['out_aki'].loc[idx_test] = xg['standard'].predict_proba(X_test)[:,1]
        xg['auc_test'] = roc_auc_score(y_test,y_pred['out_aki'].loc[idx_test])
        fpr_rf, tpr_rf, _ = sklearn.metrics.roc_curve(y_test,y_pred['out_aki'].loc[idx_test])
        xg['rocdata_test'] = pd.DataFrame({'fpr':fpr_rf,'tpr':tpr_rf})        
        ci_output = auc_ci(y_test,y_pred['out_aki'].loc[idx_test])
        xg['auc_ci_test'] = ci_output 


        # out of sample 10-fold cross-validation  ########################################
        xg['crossval'] = XGBClassifier(**best_params)
        xg['auc_crossval'] = sklearn.model_selection.cross_val_score(xg['crossval'], X_train, y_train, cv=cv, scoring='roc_auc').mean()
        y_pred_cv['out_aki'].loc[idx_train] = sklearn.model_selection.cross_val_predict(xg['crossval'], X_train, y_train, cv=cv, method='predict_proba')[:,1] 
        fpr_rf, tpr_rf, _ = roc_curve(y_train,y_pred_cv['out_aki'].loc[idx_train])
        xg['rocdata_crossval'] = pd.DataFrame({'fpr':fpr_rf,'tpr':tpr_rf})
        ci_output = auc_ci(y_train,y_pred_cv['out_aki'].loc[idx_train])
        xg['auc_ci_crossval'] = ci_output

        # external validation
        xg['crossval'].fit(X_train, y_train) # fit model with all training data
        y_pred_cv['out_aki'].loc[idx_test] = xg['crossval'].predict_proba(X_test)[:,1] 
        xg['auc_crossval_external'] = roc_auc_score(y_test,y_pred_cv['out_aki'].loc[idx_test])
        fpr_rf, tpr_rf, _ = roc_curve(y_test,y_pred_cv['out_aki'].loc[idx_test])
        xg['rocdata_crossval_external'] = pd.DataFrame({'fpr':fpr_rf,'tpr':tpr_rf})
        ci_output = auc_ci(y_test,y_pred_cv['out_aki'].loc[idx_test])
        xg['auc_ci_crossval_external'] = ci_output

        print("finished training and testing, writing back to auc metrics")

        auc_metrics.iloc[i,0] = xg['auc_train']
        auc_metrics.iloc[i,1] = xg['auc_ci_train']
        auc_metrics.iloc[i,2] = xg['auc_test']
        auc_metrics.iloc[i,3] = xg['auc_ci_test']

        auc_metrics.iloc[i,4] = xg['auc_crossval']
        auc_metrics.iloc[i,5] = xg['auc_ci_crossval']
        auc_metrics.iloc[i,6] = xg['auc_crossval_external']
        auc_metrics.iloc[i,7] = xg['auc_ci_crossval_external']
        xg['ypred'] = y_pred
        xg['ypred_cv'] = y_pred_cv
        
        model_store[model_name] = xg 

if write_analytics_pickles == 'yes':
    ## model_store
    file = open(write_pickles_path + 'imputation_model_store_' + version + '.pickle','wb')
    pickle.dump(model_store,file)
    file.close() 

    ## auc_metrics
    file = open(write_pickles_path + 'imputation_aucs_' + version + '.pickle','wb')
    pickle.dump(auc_metrics,file)
    file.close() 


def cal_95ci(auc_scores):
        mean_auc = np.mean(auc_scores)
        sd_auc = np.std(auc_scores, ddof=1)
        se_auc = sd_auc / np.sqrt(len(auc_scores))

        z_score = 1.96

        moe = z_score * se_auc

        ci_lower = mean_auc - moe
        ci_upper = mean_auc + moe

        return mean_auc, ci_lower, ci_upper
       
if build_model == 'yes':
    # calculate AUC
    cal_95ci(auc_metrics['cvev'].values)



#%%
# GENERATE FINAL DATASET 

# 1. read imputed dataset
read_analytics_pickles = 'yes'
read_pickles_path = '\\'

n_imputations = 20


if read_analytics_pickles == 'yes':
    
    for outcome in outcomes:
        # create a column in enc_set called "out_aki_imp", copy original outcome values in
        enc_set[outcome + '_imp'] = enc_set[outcome].copy()
        
        
        if outcome == 'out_aki':
            path_add = 'imputed_123\\'
        elif outcome == 'out_aki_greater_than_2':
            path_add = 'imputed_23\\'
        else:
            print("Error: not valid outcome : " + str(outcome))
    
    
    
    
        # splitting indicies into non-overlapped 20 folds
        enc_df = pickle.load(open(read_pickles_path  + path_add + 'IMPUTED_DATAFRAME_'+outcome+'__'+str(0)+'_' + version  + '.pickle', 'rb'))
        ic_idx = enc_df[enc_df['IC'] == 1].index.values.tolist()
        
        random.seed(3072)
        np.random.shuffle(ic_idx)
        ic_folds = np.array_split(ic_idx, 20)
        
        for i in range(n_imputations):
            print("Processing imputed dataframe " + str(i))
            # read imputed df
            enc_df = pickle.load(open(read_pickles_path  + path_add + 'IMPUTED_DATAFRAME_'+outcome+'__'+str(i)+'_' + version  + '.pickle', 'rb'))
            #assign imputed outcome to "out_aki_imp"
            imp_idx = ic_folds[i]
            enc_set[outcome + '_imp'].iloc[imp_idx] = enc_df[outcome].iloc[imp_idx].copy()
        
        print("imputation finished: final dataset generated. : " + str(outcome))
    
    
    file = open(write_pickles_path + 'enc_set_' + version + '.pickle','wb')
    pickle.dump(enc_set,file)
    file.close() 


