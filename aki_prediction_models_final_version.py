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
from sklearn.metrics import classification_report as cr, brier_score_loss, precision_recall_curve,roc_auc_score
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
read_pickle_data = 'yes'

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
            idx_train = enc_set.loc[enc_set['hosp'].isin(['hosp1','hosp2','hosp3'])].index
            idx_test = enc_set.loc[enc_set['hosp'].isin(['hosp4','hosp5'])].index
        print('train % ' + str(float(len(idx_train))/(len(idx_train)+len(idx_test))))

        return idx_train, idx_test


def specify_predictors(df, data_type='continuous'):
    if data_type == 'categorical': # missing data -> specific category
        pred_library = {}
        pred_library['pred_demographics'] = ['age_raw','sex','race','ethnicity']
        pred_library['pred_comps'] = ['comp_'+str(i) for i in range(1,64)]
        pred_library['pred_probs'] =   ['cevd','hp','mld','msld','diab','diabwc','mrend','srend','canc','metacanc','aids_hiv','ami','cpd','chf','pvd','pud','dementia','rheumd', 'AKI_PHX_1YEAR']
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
        pred_library['pred_probs'] =  ['cevd','hp','mld','msld','diab','diabwc','mrend','srend','canc','metacanc','aids_hiv','ami','cpd','chf','pvd','pud','dementia','rheumd', 'AKI_PHX_1YEAR']
        pred_library['pred_vitals_triage'] = ['dbp_first','sbp_first','spo2_first','pulse_first','resp_first','temp_first']
        pred_library['pred_labs'] = [col for col in df.columns if col.endswith('_last') and not col.endswith('_last_label') and not col.startswith(('sbp','dbp','temp','spo2','pulse','resp'))]
        pred_library['pred_labs_more'] =[col for col in df.columns if col.endswith(('_min','_max')) and not col.endswith(('_min_label','_max_label')) and not col.startswith(('sbp','dbp','temp','spo2','pulse','resp'))]
        pred_library['outcomes'] = ['out_aki', 'out_aki_greater_than_2']
        pred_library['pred_aki'] = ['baseline_cr_exist', 'baseline_cr','aki_init_ed_stage']
        pred_library['missing'] = ['out_aki_missing', 'out_ckd_missing']
    return pred_library




#%% 
# Inverse Propensity Weighting

def get_propensity(enc_set,pred_dummy,missing_type,train_index, ps_model_dir):
            
            # Method: only train the model on discharged patients only
            discharge_index =enc_set[enc_set['ed_disposition_aki']=='Discharge'].index
            admit_index =enc_set[enc_set['ed_disposition_aki']=='Admit'].index
            
            # find discharged and aki missing in training cohort, later does not include large weights for these group of people
            discharge_missing_index = enc_set[(enc_set['ed_disposition_aki']=='Discharge')&(enc_set['out_aki_missing']==1)].index
            not_focus_index = train_index.intersection(discharge_missing_index)
            not_focus_index = not_focus_index.values
            
            #ipw_idx= train_index.intersection(discharge_index)
            ipw_idx=train_index # ipw model gonna train on everypatient inside the training cohort in the prediction model
            ipw_idx=ipw_idx.values
            
            enc_set[missing_type+'_present'] = 1-enc_set[missing_type]
            present_suffix = '_present' # target: being observed
            missingness = enc_set[missing_type+present_suffix].loc[ipw_idx]
            print(f"generating propensity score for {missing_type+present_suffix} ")

            # prepare dataset
            X, y = pred_dummy.loc[ipw_idx], missingness

            bp_ipw = {'subsample': 0.2,
                    'reg_lambda': 1, 
                    'reg_alpha': 0.01, 
                    'objective': 'binary:logistic', 
                    'n_estimators': 300, 
                    'max_depth': 15, 
                    'learning_rate': 0.03, 
                    'gamma': 5, 
                    'eval_metric': 'logloss', 
                    'colsample_bytree': 0.7}

            model = XGBClassifier(**bp_ipw).fit(X, y)
        
            ps_pred = pd.DataFrame(np.ones(shape=(len(train_index),1)), index=train_index, columns = ['propensity_score'])
            
     
            ps_pred['propensity_score'].loc[ipw_idx] = model.predict_proba(X)[:,1]
            
            #exp: admitted and observed patietns with low weight
            ps_pred.loc[train_index & admit_index, 'propensity_score'] = 1
            
            # save IPW model
            model_path =ps_model_dir +  "PS_model.json"
            model.save_model(model_path)
            return ps_pred['propensity_score'].values



def IPW(ipw_type, enc_set, pred_dummy, idx_train, ps_model_dir):
    if ipw_type == 'normal':
        ipw_a = get_propensity(enc_set,pred_dummy,missing_type="out_aki_missing",train_index = idx_train, ps_model_dir=ps_model_dir)
        ipw_aki = 1/ipw_a
        ipw = {"out_aki":ipw_aki}
        propensity_scores = {"out_aki":ipw_a}
    elif ipw_type == 'no':
        
        ipw_aki = np.ones((len(idx_train)))
        ipw = {"out_aki":ipw_aki}
        propensity_scores = {"out_aki":ipw_aki}
    return ipw, propensity_scores
    
    


#%% BUILDING MODELS(XGBOOST)
#####################################################################################
####################################################################################################
#################################### MODEL BUILDING    #############################################
#
#        1. Build using all (missing outcome = NEGATIVE)
#        2. Build using ONLY encounters where outcome is known
#                A. Any repeat sCr between ED departure 72 hours
#                B. Must have stayed in Hospital for 72 hours
####################################################################################################
####################################################################################################
#################################### MODEL EVALIATION    #############################################
#
#        1. Total Pop
#        2. Subset with known outcome 
#                    A & B
#        3. discharged patietns with known outcome
####################################################################################################
####################################################################################################
####################################################################################################
###################################### MODEL Strategy    #############################################     
#          Approach 1: Train 1, Test 1       naming: all
#          Approach 1.2: Train 1, Test 2A      naming: all-repeat
#          Approach 1.3: Train 1, Test 3        naming: all-discharge
#          Approach 2: Train 2A, Test 2A     naming: repeat
#          Approach 2.2: Train 2A, Test 3     naming: repeat-discharge
#              Validation 2.3 Train 2A, Test on admitted naming: repeat-admit
####################################################################################################


import warnings
warnings.filterwarnings("ignore")


build_model = 'yes'
models = ['all', 'imp','repeat','ipw'] 

write_analytics_pickles = 'yes'
write_pickles_path = '\\'

pred_library = specify_predictors(enc_set,'continuous') # continuous
outcomes = ['out_aki', 'out_aki_greater_than_2']
# RF
rf_pred = specify_predictors(enc_set,'categorical')
rf_pred_list = rf_pred['pred_demographics']  + rf_pred['pred_comps'] +rf_pred['pred_aki'] + rf_pred['pred_probs'] + rf_pred['pred_vitals_triage'] + rf_pred['pred_labs'] + rf_pred['pred_labs_more']


if build_model == 'yes':   
    #split_date = str(enc_set['arrdt'].sort_values(ascending=True).reset_index().iloc[round(len(enc_set)*(2/3))]['arrdt'])[:10]
    
    #temporal validation appraoch
    split_date = '2023-01-01'
    
    pred_library['pred_comps'] = remove_zero(enc_set,pred_library['pred_comps'])
    pred_library['pred_probs'] = remove_zero(enc_set,pred_library['pred_probs'])
    
    plist_dict = {}
    plist_dict['totalkey'] = pred_library['pred_demographics'] + pred_library['pred_vitals_triage'] + pred_library['pred_labs'] + pred_library['pred_aki'] + pred_library['pred_probs'] + pred_library['pred_comps'] + pred_library['pred_labs_more']
    model_store = {}
    for model in models:
        model_store_sub = {}
        
        #Primary outcome: serum creatinine present, out_aki_missing indicates sCr missing status
        outcome_1 = 'out_aki'
        # geting repeat sCr within 72 hours population
        idx_not_repeat = enc_set.loc[enc_set[outcome_1+'_missing'] == 1].index
        idx_repeat = enc_set.loc[enc_set[outcome_1+'_missing'] == 0].index
        # getting discharged patietns with known outcomes
        idx_discharge = enc_set.loc[(enc_set[outcome_1+'_missing'] == 0) & (enc_set['ed_disposition_aki'] == 'Discharge')].index
        # getting admitted patients with know outcomes
        idx_admit = enc_set.loc[(enc_set[outcome_1+'_missing'] == 0) & (enc_set['ed_disposition_aki'] == 'Admit')].index
        # initial AKI stage not 3
        #idx_inits = enc_set.loc[enc_set['aki_init_ed_stage'] != 3].index
        # for imputation:
        imp_suffix = ''

        if model == 'all':
            print("Processing model: ALL")
            idx_train, idx_test = cross_split(enc_set,split_date,'time')
            #idx_test = idx_test.intersection(idx_inits)
            idx_test_repeat = idx_test & idx_repeat
            idx_test_discharge = idx_test & idx_discharge
            idx_test_admit = idx_test & idx_admit
            
        elif model == 'repeat':
            print("Processing model: REPEAT")
            idx_train, idx_test = cross_split(enc_set.loc[idx_repeat],split_date,'time')
            #idx_test = idx_test.intersection(idx_inits)
            idx_test_discharge = idx_test & idx_discharge
            idx_test_admit = idx_test & idx_admit
            # getting non observed for external validation
            
        elif model =='ipw':
            idx_train, idx_test = cross_split(enc_set,split_date,'time')
            idx_test_repeat = idx_test & idx_repeat
            
            idx_test_discharge = idx_test & idx_discharge
            idx_test_admit = idx_test & idx_admit

            
        elif model == 'imp':
            idx_train, idx_test = cross_split(enc_set,split_date,'time')
            # test in ipw only evaluted on non-imputed encounters
            idx_test = idx_test.intersection(idx_repeat)
            idx_test_repeat = idx_test & idx_repeat
            idx_test_discharge = idx_test & idx_discharge
            idx_test_admit = idx_test & idx_admit
            
        # EXCLUDING PATIENTS WITH INITIAL STAGE 3 AKI IN TRAINING COHORT
        #idx_train = idx_train.intersection(idx_inits)
        
        for key in plist_dict:
            plist = plist_dict[key]    
            pred_dummy = dummy(enc_set,'sex',ref_var=1)
            pred_dummy = remove_ref_category(plist,pred_dummy,enc_set)
            
            ipw_dummy = dummy(enc_set,'sex',ref_var=1)
            ipw_plist = rf_pred['pred_demographics']  + rf_pred['pred_comps'] +rf_pred['pred_aki'] + rf_pred['pred_probs'] + rf_pred['pred_vitals_triage'] + rf_pred['pred_labs'] + rf_pred['pred_labs_more']
            ipw_dummy = remove_ref_category(ipw_plist,ipw_dummy,enc_set)
            
            ipw_dummy.loc[ipw_dummy['age_raw'] >= 120, 'age_raw'] = 90
            pred_dummy.loc[pred_dummy['age_raw'] >= 120, 'age_raw'] = 90
            #####
            #####
            # IPW USE CONTINUOUSE DATA
            
            ipw_dummy = pred_dummy.copy()
            
            xg = {}
            y_pred = pd.DataFrame(np.zeros(shape=(len(enc_set),len(outcomes))), columns = outcomes)
            y_pred_cv = pd.DataFrame(np.zeros(shape=(len(enc_set),len(outcomes))), columns = outcomes)
            
            if model in ['ipw']:
                ipw, propensity_scores = IPW('normal', enc_set, ipw_dummy, idx_train, ps_model_dir = write_pickles_path)
                print("IPW finished")
            elif model in ['all','repeat','imp']:
                ipw, propensity_scores = IPW('no', enc_set, ipw_dummy, idx_train, ps_model_dir = write_pickles_path)
                
            sample_weight = ipw[outcome_1]
            if model == 'ipw':
                    idx_train_copy = idx_train.copy()
                    idx_train = idx_train & idx_repeat
                    idx_test_copy = idx_test.copy()
                    idx_test = idx_test & idx_repeat
                    sw_df = pd.DataFrame(data=sample_weight, index=idx_train_copy, columns = ['weight'])
                    sw_df = sw_df.loc[idx_train] # only include sample weights from observed pats
                    sample_weight = sw_df['weight'].values

            # GENERATE PROPENSITY SCORE FOR TEST SET (PROPENSITY ANALYSIS)
            # PREDICTED PROBABILITY THAT THE OUTCOME IS OBSERVED
            if model == 'ipw':
                ps_model = xgboost.Booster()
                ps_model.load_model(write_pickles_path+'PS_model.json') 
                propensity_score_test = ps_model.predict(xgboost.DMatrix(pred_dummy.loc[idx_test]))
                ps_test = pd.DataFrame(data=propensity_score_test, index=idx_test, columns = ['ps'])
                file = open(write_pickles_path + 'ps_test_' + version + '.pickle','wb')
                pickle.dump(ps_test,file)
                file.close()    
            
            for outcome in outcomes:
                
                if model == 'imp':
                    imp_suffix = '_imp'
                else:
                    imp_suffix = ''
                    
                print("OUTCOME : "+str(outcome) + str(imp_suffix))
                
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
                xg[outcome] = XGBClassifier(**best_params)
                
                xg[outcome].fit(pred_dummy.loc[idx_train], enc_set[outcome+imp_suffix].loc[idx_train],sample_weight = sample_weight)
                y_pred[outcome].loc[idx_train] = xg[outcome].predict_proba(pred_dummy.loc[idx_train])[:,1]
                #training process eval
                xg[outcome+'_auc_train'] = roc_auc_score(enc_set[outcome+imp_suffix].loc[idx_train],y_pred[outcome].loc[idx_train])
                fpr_rf, tpr_rf, threshold_rf = sklearn.metrics.roc_curve(enc_set[outcome+imp_suffix].loc[idx_train],y_pred[outcome].loc[idx_train])
                xg[outcome+'_rocdata_train'] = pd.DataFrame({'fpr':fpr_rf,'tpr':tpr_rf,'threshold':threshold_rf})
                
                # calculate confidence interval of AUC using delong
                ci_output = auc_ci(enc_set[outcome+imp_suffix].loc[idx_train], y_pred[outcome].loc[idx_train])
                xg[outcome+'_auc_ci_train'] = ci_output
                print('train '+outcome+' auc ci = '+str(xg[outcome+'_auc_ci_train']))

                # Precision-recall curve   # notice: PRC function returns: precision (n_thresholds_precision+element"1")  recall (n_thresholds_recall+element"0"), threshold(n_thresholds)
                pre_prc, rec_prc, threshold_prc = precision_recall_curve(enc_set[outcome+imp_suffix].loc[idx_train], y_pred[outcome].loc[idx_train])
                xg[outcome+'_prcdata_train'] = pd.DataFrame({'pre':pre_prc,'rec':rec_prc, 'threshold':np.append(threshold_prc, np.nan)})
                
                
                # test ################################################################
                y_pred[outcome].loc[idx_test] = xg[outcome].predict_proba(pred_dummy.loc[idx_test])[:,1]
                xg[outcome+'_auc_test'] = roc_auc_score(enc_set[outcome+imp_suffix].loc[idx_test],y_pred[outcome].loc[idx_test])
                fpr_rf, tpr_rf, threshold_rf = sklearn.metrics.roc_curve(enc_set[outcome+imp_suffix].loc[idx_test],y_pred[outcome].loc[idx_test])
                xg[outcome+'_rocdata_test'] = pd.DataFrame({'fpr':fpr_rf,'tpr':tpr_rf,'threshold':threshold_rf})     
                # calculate confidence interval of AUC using delong
                ci_output = auc_ci(enc_set[outcome+imp_suffix].loc[idx_test], y_pred[outcome].loc[idx_test])
                xg[outcome+'_auc_ci_test'] = ci_output
                print('test '+outcome+' auc ci = '+str(xg[outcome+'_auc_ci_test']))
                # Precision-recall curve   # notice: PRC function returns: precision (n_thresholds_precision+element"1")  recall (n_thresholds_recall+element"0"), threshold(n_thresholds)
                pre_prc, rec_prc, threshold_prc = precision_recall_curve(enc_set[outcome+imp_suffix].loc[idx_test], y_pred[outcome].loc[idx_test])
                xg[outcome+'_prcdata_test'] =  pd.DataFrame({'pre':pre_prc,'rec':rec_prc, 'threshold':np.append(threshold_prc, np.nan)})
                
                # # 10-fold Cross Validation with SAMPLE WEIGHT   
                # #######################################################################
                # cross validation w sample weight#######################################
                # initialize cross validation & classifier
                kf = StratifiedKFold(n_splits=10)
                X = pred_dummy.loc[idx_train]
                y = enc_set[outcome+imp_suffix].loc[idx_train]
                xg[outcome+'_cv'] = XGBClassifier(**best_params)
                auroc_scores = []
                print("Cross-validation started")
                for train_index, test_index in kf.split(X, y):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    weights_train = sample_weight[train_index]
                    
                    xg[outcome+'_cv'].fit(X_train, y_train, sample_weight = weights_train) # train the model on K-1 folds
                    # evaluate on tes tset in each iteration
                    y_pred_cv[outcome].loc[idx_train[test_index]] = xg[outcome+'_cv'].predict_proba(X_test)[:,1] 
                    auroc_score = roc_auc_score(y_test, y_pred_cv[outcome].loc[idx_train].iloc[test_index])
                    auroc_scores.append(auroc_score)
                # mean auc across all folds 
                xg[outcome+'_crossval_auc'] = np.mean(auroc_scores)
                # cross-validation process
                fpr_rf, tpr_rf, _ = roc_curve(enc_set[outcome+imp_suffix].loc[idx_train], y_pred_cv[outcome].loc[idx_train])
                xg[outcome+'_rocdata_crossval'] = pd.DataFrame({'fpr':fpr_rf,'tpr':tpr_rf})
                ci_output = auc_ci(enc_set[outcome+imp_suffix].loc[idx_train], y_pred_cv[outcome].loc[idx_train])
                xg[outcome+'_auc_ci_crossval'] = ci_output
                print('Out of sample 10-fold cross-validation '+outcome+' auc = '+str(xg[outcome+'_auc_ci_crossval']))
                pre_prc, rec_prc, threshold_prc  = precision_recall_curve(enc_set[outcome+imp_suffix].loc[idx_train], y_pred_cv[outcome].loc[idx_train])
                xg[outcome+'_prcdata_crossval'] = pd.DataFrame({'pre':pre_prc,'rec':rec_prc, 'threshold':np.append(threshold_prc, np.nan)})
                
                # Validation (external)
                ### xg[outcome+'_cv'].fit(pred_dummy.loc[idx_train], enc_set[outcome+imp_suffix].loc[idx_train], sample_weight = sample_weight)
                y_pred_cv[outcome].loc[idx_test] = xg[outcome+'_cv'].predict_proba(pred_dummy.loc[idx_test])[:,1]
                print("cross-validation ended")
                xg[outcome+'_auc_test_crossval'] = roc_auc_score(enc_set[outcome+imp_suffix].loc[idx_test],y_pred_cv[outcome].loc[idx_test])
                fpr_rf, tpr_rf, threshold_rf = sklearn.metrics.roc_curve(enc_set[outcome+imp_suffix].loc[idx_test],y_pred_cv[outcome].loc[idx_test])
                xg[outcome+'_rocdata_test_crossval'] = pd.DataFrame({'fpr':fpr_rf,'tpr':tpr_rf,'threshold':threshold_rf})    
                ci_output = auc_ci(enc_set[outcome+imp_suffix].loc[idx_test], y_pred_cv[outcome].loc[idx_test])
                xg[outcome+'_auc_ci_test_crossval'] = ci_output
                print('test '+outcome+' auc ci = '+str(xg[outcome+'_auc_ci_test_crossval']))
                # Precision-recall curve   # notice: PRC function returns: precision (n_thresholds_precision+element"1")  recall (n_thresholds_recall+element"0"), threshold(n_thresholds)
                pre_prc, rec_prc, threshold_prc = precision_recall_curve(enc_set[outcome+imp_suffix].loc[idx_test], y_pred_cv[outcome].loc[idx_test])
                xg[outcome+'_prcdata_test_crossval'] =  pd.DataFrame({'pre':pre_prc,'rec':rec_prc, 'threshold':np.append(threshold_prc, np.nan)})
                
                ###########################################################################################################################################################
                ###########################################################################################################################################################
                # EXTERNAL EVALUATION IN SUBGROUP (KNOWN OUTCOME)
                ###########################################################################################################################################################

                if model == "all" or model == 'imp' or model == 'ipw':
                    # approach 1.2 Train 1, Test 2A
                    xg[outcome+'_auc_test_all_repeat'] = roc_auc_score(enc_set[outcome+imp_suffix].loc[idx_test_repeat],y_pred[outcome].loc[idx_test_repeat])
                    fpr_rf, tpr_rf, threshold_rf = sklearn.metrics.roc_curve(enc_set[outcome+imp_suffix].loc[idx_test_repeat],y_pred[outcome].loc[idx_test_repeat])
                    xg[outcome+'_rocdata_test_all_repeat'] = pd.DataFrame({'fpr':fpr_rf,'tpr':tpr_rf,'threshold':threshold_rf})        
                    # print('approach 1.2 all-repeat test '+outcome+' auc = '+str(xg[outcome+'_auc_test_all_repeat']))
                    # calculate confidence interval of AUC using delong
                    ci_output = auc_ci(enc_set[outcome+imp_suffix].loc[idx_test_repeat], y_pred[outcome].loc[idx_test_repeat])
                    xg[outcome+'_auc_ci_test_all_repeat'] = ci_output
                    print('approach 1.2 all-repeat test '+outcome+' auc ci = '+str(xg[outcome+'_auc_ci_test_all_repeat']))
                    # Precision-recall curve   # notice: PRC function returns: precision (n_thresholds_precision+element"1")  recall (n_thresholds_recall+element"0"), threshold(n_thresholds)
                    pre_prc, rec_prc, threshold_prc = precision_recall_curve(enc_set[outcome+imp_suffix].loc[idx_test_repeat], y_pred[outcome].loc[idx_test_repeat])
                    xg[outcome+'_prcdata_test_all_repeat'] =  pd.DataFrame({'pre':pre_prc,'rec':rec_prc, 'threshold':np.append(threshold_prc, np.nan)})
                    

                    # approach 1.3 Train 1, Test 3
                    xg[outcome+'_auc_test_all_discharge'] = roc_auc_score(enc_set[outcome+imp_suffix].loc[idx_test_discharge], y_pred[outcome].loc[idx_test_discharge])
                    fpr_rf, tpr_rf, threshold_rf = sklearn.metrics.roc_curve(enc_set[outcome+imp_suffix].loc[idx_test_discharge],y_pred[outcome].loc[idx_test_discharge])
                    xg[outcome+'_rocdata_test_all_discharge'] = pd.DataFrame({'fpr':fpr_rf,'tpr':tpr_rf,'threshold':threshold_rf})      
                    # print('approach 1.3 all-discharge test '+outcome+' auc = '+str(xg[outcome+'_auc_test_all_discharge']))
                    # calculate confidence interval of AUC using delong
                    ci_output = auc_ci(enc_set[outcome+imp_suffix].loc[idx_test_discharge], y_pred[outcome].loc[idx_test_discharge])
                    xg[outcome+'_auc_ci_test_all_discharge'] = ci_output
                    print('approach 1.3 all-discharge test '+outcome+' auc ci = '+str(xg[outcome+'_auc_ci_test_all_discharge']))
                    # Precision-recall curve   # notice: PRC function returns: precision (n_thresholds_precision+element"1")  recall (n_thresholds_recall+element"0"), threshold(n_thresholds)
                    pre_prc, rec_prc, threshold_prc = precision_recall_curve(enc_set[outcome+imp_suffix].loc[idx_test_discharge], y_pred[outcome].loc[idx_test_discharge])
                    xg[outcome+'_prcdata_test_all_discharge'] =  pd.DataFrame({'pre':pre_prc,'rec':rec_prc, 'threshold':np.append(threshold_prc, np.nan)})
                    
                    
                    # approach 1.4 Train 1, Test 4
                    xg[outcome+'_auc_test_all_admit'] = roc_auc_score(enc_set[outcome+imp_suffix].loc[idx_test_admit], y_pred[outcome].loc[idx_test_admit])
                    fpr_rf, tpr_rf, threshold_rf = sklearn.metrics.roc_curve(enc_set[outcome+imp_suffix].loc[idx_test_admit],y_pred[outcome].loc[idx_test_admit])
                    xg[outcome+'_rocdata_test_all_admit'] = pd.DataFrame({'fpr':fpr_rf,'tpr':tpr_rf,'threshold':threshold_rf})      
                    # calculate confidence interval of AUC using delong
                    ci_output = auc_ci(enc_set[outcome+imp_suffix].loc[idx_test_admit], y_pred[outcome].loc[idx_test_admit])
                    xg[outcome+'_auc_ci_test_all_admit'] = ci_output
                    print('approach 1.3 all-admit test '+outcome+' auc ci = '+str(xg[outcome+'_auc_ci_test_all_admit']))
                    # Precision-recall curve   # notice: PRC function returns: precision (n_thresholds_precision+element"1")  recall (n_thresholds_recall+element"0"), threshold(n_thresholds)
                    pre_prc, rec_prc, threshold_prc = precision_recall_curve(enc_set[outcome+imp_suffix].loc[idx_test_admit], y_pred[outcome].loc[idx_test_admit])
                    xg[outcome+'_prcdata_test_all_admit'] =  pd.DataFrame({'pre':pre_prc,'rec':rec_prc, 'threshold':np.append(threshold_prc, np.nan)})
                    
                    
                    if model == 'ipw' or model == 'imp':
                        # test on not observed patients
                        y_pred[outcome].loc[idx_not_repeat] = xg[outcome].predict_proba(pred_dummy.loc[idx_not_repeat])[:,1]
                        y_pred_cv[outcome].loc[idx_not_repeat] = xg[outcome+'_cv'].predict_proba(pred_dummy.loc[idx_not_repeat])[:,1]  
                    
                    # # cross-validation edition ##############################
                    # # approach 1.2 Train 1, Test 2A
                    xg[outcome+'_auc_test_all_repeat_crossval'] = roc_auc_score(enc_set[outcome+imp_suffix].loc[idx_test_repeat],y_pred_cv[outcome].loc[idx_test_repeat])
                    fpr_rf, tpr_rf, threshold_rf = sklearn.metrics.roc_curve(enc_set[outcome+imp_suffix].loc[idx_test_repeat],y_pred_cv[outcome].loc[idx_test_repeat])
                    xg[outcome+'_rocdata_test_all_repeat_crossval'] = pd.DataFrame({'fpr':fpr_rf,'tpr':tpr_rf,'threshold':threshold_rf})        
                    # print('approach 1.2 all-repeat test '+outcome+' auc = '+str(xg[outcome+'_auc_test_all_repeat']))
                    # calculate confidence interval of AUC using delong
                    ci_output = auc_ci(enc_set[outcome+imp_suffix].loc[idx_test_repeat], y_pred_cv[outcome].loc[idx_test_repeat])
                    xg[outcome+'_auc_ci_test_all_repeat_crossval'] = ci_output
                    print('approach 1.2 all-repeat test '+outcome+' auc ci (CV) = '+str(xg[outcome+'_auc_ci_test_all_repeat_crossval']))
                    # Precision-recall curve   # notice: PRC function returns: precision (n_thresholds_precision+element"1")  recall (n_thresholds_recall+element"0"), threshold(n_thresholds)
                    pre_prc, rec_prc, threshold_prc = precision_recall_curve(enc_set[outcome+imp_suffix].loc[idx_test_repeat], y_pred_cv[outcome].loc[idx_test_repeat])
                    xg[outcome+'_prcdata_test_all_repeat_crossval'] =  pd.DataFrame({'pre':pre_prc,'rec':rec_prc, 'threshold':np.append(threshold_prc, np.nan)})
                    

                    # approach 1.3 Train 1, Test 3
                    xg[outcome+'_auc_test_all_discharge_crossval'] = roc_auc_score(enc_set[outcome+imp_suffix].loc[idx_test_discharge], y_pred_cv[outcome].loc[idx_test_discharge])
                    fpr_rf, tpr_rf, threshold_rf = sklearn.metrics.roc_curve(enc_set[outcome+imp_suffix].loc[idx_test_discharge],y_pred_cv[outcome].loc[idx_test_discharge])
                    xg[outcome+'_rocdata_test_all_discharge_crossval'] = pd.DataFrame({'fpr':fpr_rf,'tpr':tpr_rf,'threshold':threshold_rf})      
                    # print('approach 1.3 all-discharge test '+outcome+' auc = '+str(xg[outcome+'_auc_test_all_discharge']))
                    # calculate confidence interval of AUC using delong
                    ci_output = auc_ci(enc_set[outcome+imp_suffix].loc[idx_test_discharge], y_pred_cv[outcome].loc[idx_test_discharge])
                    xg[outcome+'_auc_ci_test_all_discharge_crossval'] = ci_output
                    print('approach 1.3 all-discharge test '+outcome+' auc ci (CV)= '+str(xg[outcome+'_auc_ci_test_all_discharge_crossval']))
                    # Precision-recall curve   # notice: PRC function returns: precision (n_thresholds_precision+element"1")  recall (n_thresholds_recall+element"0"), threshold(n_thresholds)
                    pre_prc, rec_prc, threshold_prc = precision_recall_curve(enc_set[outcome+imp_suffix].loc[idx_test_discharge], y_pred_cv[outcome].loc[idx_test_discharge])
                    xg[outcome+'_prcdata_test_all_discharge_crossval'] =  pd.DataFrame({'pre':pre_prc,'rec':rec_prc, 'threshold':np.append(threshold_prc, np.nan)})
                    
                    
                    # approach 1.4 Train 1, Test 4
                    xg[outcome+'_auc_test_all_admit_crossval'] = roc_auc_score(enc_set[outcome+imp_suffix].loc[idx_test_admit], y_pred_cv[outcome].loc[idx_test_admit])
                    fpr_rf, tpr_rf, threshold_rf = sklearn.metrics.roc_curve(enc_set[outcome+imp_suffix].loc[idx_test_admit],y_pred_cv[outcome].loc[idx_test_admit])
                    xg[outcome+'_rocdata_test_all_admit_crossval'] = pd.DataFrame({'fpr':fpr_rf,'tpr':tpr_rf,'threshold':threshold_rf})      
                    # calculate confidence interval of AUC using delong
                    ci_output = auc_ci(enc_set[outcome+imp_suffix].loc[idx_test_admit], y_pred_cv[outcome].loc[idx_test_admit])
                    xg[outcome+'_auc_ci_test_all_admit_crossval'] = ci_output
                    print('approach 1.3 all-admit test '+outcome+' auc ci (CV) = '+str(xg[outcome+'_auc_ci_test_all_admit_crossval']))
                    # Precision-recall curve   # notice: PRC function returns: precision (n_thresholds_precision+element"1")  recall (n_thresholds_recall+element"0"), threshold(n_thresholds)
                    pre_prc, rec_prc, threshold_prc = precision_recall_curve(enc_set[outcome+imp_suffix].loc[idx_test_admit], y_pred_cv[outcome].loc[idx_test_admit])
                    xg[outcome+'_prcdata_test_all_admit_crossval'] =  pd.DataFrame({'pre':pre_prc,'rec':rec_prc, 'threshold':np.append(threshold_prc, np.nan)})
                   
                        
                elif model == "repeat":   
                    # approach 2.2 Train 2A, Test 3
                    xg[outcome+'_auc_test_repeat_discharge'] = roc_auc_score(enc_set[outcome].loc[idx_test_discharge],y_pred[outcome].loc[idx_test_discharge])
                    fpr_rf, tpr_rf, threshold_rf = sklearn.metrics.roc_curve(enc_set[outcome].loc[idx_test_discharge],y_pred[outcome].loc[idx_test_discharge])
                    xg[outcome+'_rocdata_test_repeat_discharge'] = pd.DataFrame({'fpr':fpr_rf,'tpr':tpr_rf,'threshold':threshold_rf})    
                    #print('approach 2.2 repeat-discharge test '+outcome+' auc = '+str(xg[outcome+'_auc_test_repeat_discharge']))
                    # calculate confidence interval of AUC using delong
                    ci_output = auc_ci(enc_set[outcome].loc[idx_test_discharge], y_pred[outcome].loc[idx_test_discharge])
                    xg[outcome+'_auc_ci_test_repeat_discharge'] = ci_output
                    print('approach 2.2 repeat-discharge test '+outcome+' auc ci = '+str(xg[outcome+'_auc_ci_test_repeat_discharge']))
                    # Precision-recall curve   # notice: PRC function returns: precision (n_thresholds_precision+element"1")  recall (n_thresholds_recall+element"0"), threshold(n_thresholds)
                    pre_prc, rec_prc, threshold_prc = precision_recall_curve(enc_set[outcome].loc[idx_test_discharge], y_pred[outcome].loc[idx_test_discharge])
                    xg[outcome+'_prcdata_test_repeat_discharge'] =  pd.DataFrame({'pre':pre_prc,'rec':rec_prc, 'threshold':np.append(threshold_prc, np.nan)})
                    


                    xg[outcome+'_auc_test_repeat_admit'] = roc_auc_score(enc_set[outcome].loc[idx_test_admit],y_pred[outcome].loc[idx_test_admit])
                    fpr_rf, tpr_rf, threshold_rf = sklearn.metrics.roc_curve(enc_set[outcome].loc[idx_test_admit],y_pred[outcome].loc[idx_test_admit])
                    xg[outcome+'_rocdata_test_repeat_admit'] = pd.DataFrame({'fpr':fpr_rf,'tpr':tpr_rf,'threshold':threshold_rf})       
                    #print('Validation 2.3 repeat-admit test '+outcome+' auc = '+str(xg[outcome+'_auc_test_repeat_admit']))
                    # calculate confidence interval of AUC using delong
                    ci_output = auc_ci(enc_set[outcome].loc[idx_test_admit], y_pred[outcome].loc[idx_test_admit])
                    xg[outcome+'_auc_ci_test_repeat_admit'] = ci_output
                    print('Validation 2.3 repeat-admit test '+outcome+' auc ci = '+str(xg[outcome+'_auc_ci_test_repeat_admit']))
                    # Precision-recall curve   # notice: PRC function returns: precision (n_thresholds_precision+element"1")  recall (n_thresholds_recall+element"0"), threshold(n_thresholds)
                    pre_prc, rec_prc, threshold_prc = precision_recall_curve(enc_set[outcome].loc[idx_test_admit], y_pred[outcome].loc[idx_test_admit])
                    xg[outcome+'_prcdata_test_repeat_admit'] =  pd.DataFrame({'pre':pre_prc,'rec':rec_prc, 'threshold':np.append(threshold_prc, np.nan)})
                    ##############################
                    # test on not observed patients
                    y_pred[outcome].loc[idx_not_repeat] = xg[outcome].predict_proba(pred_dummy.loc[idx_not_repeat])[:,1]
                    
                    
                    # # cross-validation edition ##############################
                    # # approach 2.2 Train 2A, Test 3
                    xg[outcome+'_auc_test_repeat_discharge_crossval'] = roc_auc_score(enc_set[outcome].loc[idx_test_discharge],y_pred_cv[outcome].loc[idx_test_discharge])
                    fpr_rf, tpr_rf, threshold_rf = sklearn.metrics.roc_curve(enc_set[outcome].loc[idx_test_discharge],y_pred_cv[outcome].loc[idx_test_discharge])
                    xg[outcome+'_rocdata_test_repeat_discharge_crossval'] = pd.DataFrame({'fpr':fpr_rf,'tpr':tpr_rf,'threshold':threshold_rf})    
                    #print('approach 2.2 repeat-discharge test '+outcome+' auc = '+str(xg[outcome+'_auc_test_repeat_discharge']))
                    # calculate confidence interval of AUC using delong
                    ci_output = auc_ci(enc_set[outcome].loc[idx_test_discharge], y_pred_cv[outcome].loc[idx_test_discharge])
                    xg[outcome+'_auc_ci_test_repeat_discharge_crossval'] = ci_output
                    print('approach 2.2 repeat-discharge test '+outcome+' auc ci (CV) = '+str(xg[outcome+'_auc_ci_test_repeat_discharge_crossval']))
                    # Precision-recall curve   # notice: PRC function returns: precision (n_thresholds_precision+element"1")  recall (n_thresholds_recall+element"0"), threshold(n_thresholds)
                    pre_prc, rec_prc, threshold_prc = precision_recall_curve(enc_set[outcome].loc[idx_test_discharge], y_pred_cv[outcome].loc[idx_test_discharge])
                    xg[outcome+'_prcdata_test_repeat_discharge_crossval'] =  pd.DataFrame({'pre':pre_prc,'rec':rec_prc, 'threshold':np.append(threshold_prc, np.nan)})
                    


                    xg[outcome+'_auc_test_repeat_admit_crossval'] = roc_auc_score(enc_set[outcome].loc[idx_test_admit],y_pred_cv[outcome].loc[idx_test_admit])
                    fpr_rf, tpr_rf, threshold_rf = sklearn.metrics.roc_curve(enc_set[outcome].loc[idx_test_admit],y_pred_cv[outcome].loc[idx_test_admit])
                    xg[outcome+'_rocdata_test_repeat_admit_crossval'] = pd.DataFrame({'fpr':fpr_rf,'tpr':tpr_rf,'threshold':threshold_rf})       
                    #print('Validation 2.3 repeat-admit test '+outcome+' auc = '+str(xg[outcome+'_auc_test_repeat_admit']))
                    # calculate confidence interval of AUC using delong
                    ci_output = auc_ci(enc_set[outcome].loc[idx_test_admit], y_pred_cv[outcome].loc[idx_test_admit])
                    xg[outcome+'_auc_ci_test_repeat_admit_crossval'] = ci_output
                    print('Validation 2.3 repeat-admit test '+outcome+' auc ci (CV) = '+str(xg[outcome+'_auc_ci_test_repeat_admit_crossval']))
                    # Precision-recall curve   # notice: PRC function returns: precision (n_thresholds_precision+element"1")  recall (n_thresholds_recall+element"0"), threshold(n_thresholds)
                    pre_prc, rec_prc, threshold_prc = precision_recall_curve(enc_set[outcome].loc[idx_test_admit], y_pred_cv[outcome].loc[idx_test_admit])
                    xg[outcome+'_prcdata_test_repeat_admit_crossval'] =  pd.DataFrame({'pre':pre_prc,'rec':rec_prc, 'threshold':np.append(threshold_prc, np.nan)})
                    y_pred_cv[outcome].loc[idx_not_repeat] = xg[outcome+'_cv'].predict_proba(pred_dummy.loc[idx_not_repeat])[:,1]
                if write_analytics_pickles == 'yes':
                    # SAVE XGBOOST MODEL ###############
                    ### After xgboost 2.0.1, please export the model by calling 'Booster.save_model` from that version first, then load it back in current version.
                    ### https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html
                    model_path =write_pickles_path +  "xgb_model_"+str(model)+"_"+str(outcome)+".json"
                    xg[outcome].save_model(model_path)
                    model_cv_path =write_pickles_path +  "xgb_model_"+str(model)+"_cv_"+str(outcome)+".json"
                    xg[outcome+'_cv'].save_model(model_cv_path)
                
            data = pd.merge(enc_set.loc[idx_test],y_pred.loc[idx_test].rename(columns={'out_aki':'prob_aki','out_aki_greater_than_2':'prob_aki2'}),how='left',left_index=True,right_index=True,copy=False)  
            data_cv = pd.merge(enc_set.loc[idx_test],y_pred_cv.loc[idx_test].rename(columns={'out_aki':'prob_aki','out_aki_greater_than_2':'prob_aki2'}),how='left',left_index=True,right_index=True,copy=False)  
            
            xg['ypred'] = y_pred
            xg['ypred_cv'] = y_pred_cv
            xg['propensity_scores'] = propensity_scores
            model_store_sub[key] = xg
            
            if write_analytics_pickles == 'yes':
                file = open(write_pickles_path + 'xg_'+model+ '_' + version + '.pickle','wb')
                pickle.dump(xg,file)
                file.close()
                ## idx_train
                file = open(write_pickles_path + 'idxtrain_'+model+ '_' + version + '.pickle','wb')
                pickle.dump(idx_train,file)
                file.close()
                if model == 'ipw':
                    file = open(write_pickles_path + 'idxtrain_'+model+ '_original_' + version + '.pickle','wb')
                    pickle.dump(idx_train_copy,file)
                    file.close()
                ## idx_test
                file = open(write_pickles_path + 'idxtest_'+model+ '_' + version + '.pickle','wb')
                pickle.dump(idx_test,file)
                file.close()
                ## data
                file = open(write_pickles_path + 'data_'+model+ '_' + version + '.pickle','wb')
                pickle.dump(data,file)
                file.close()
                ## data cv
                file = open(write_pickles_path + 'data_cv_'+model+ '_' + version + '.pickle','wb')
                pickle.dump(data_cv,file)
                file.close()

        model_store[model] = model_store_sub
if write_analytics_pickles == 'yes':
    # save pred_dummy and tspecs
    file = open(write_pickles_path + 'tspecs_' + version + '.pickle','wb')
    pickle.dump(tspecs,file)
    file.close()    
    ## pred_dummy
    file = open(write_pickles_path + 'pred_dummy_' + version + '.pickle','wb')
    pickle.dump(pred_dummy,file)
    file.close()
    ## model_store
    file = open(write_pickles_path + 'model_store_' + version + '.pickle','wb')
    pickle.dump(model_store,file)
    file.close()  

