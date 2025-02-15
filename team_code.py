#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import pandas as pd
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

from catboost import CatBoostRegressor,CatBoostClassifier,Pool
from xgboost import XGBRegressor,XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

def preprocess_data(df,target_columns):

    data_binned = df.copy()

    data_binned['agecalc_adm'] = data_binned['agecalc_adm'].round()
    mean_weights = data_binned.groupby('agecalc_adm')['weight_kg_adm'].transform('mean')
    std_weights = data_binned.groupby('agecalc_adm')['weight_kg_adm'].transform('std')

    data_binned['weight_kg_adm'] = data_binned['weight_kg_adm'].fillna(mean_weights)
    df['weight_kg_adm'] = df['weight_kg_adm'].fillna(mean_weights)

    for idx, row in data_binned.iterrows():
        age = row['agecalc_adm']
        weight = row['weight_kg_adm']
        age_mean = mean_weights.loc[idx]
        age_std = std_weights.loc[idx]
        
        if weight > age_mean + 2.5 * age_std:
            weight_in_kg = weight * 0.453592  # Convert pounds to kg
            data_binned.loc[idx, 'weight_kg_adm'] = weight_in_kg
            df.loc[idx, 'weight_kg_adm'] = weight_in_kg

    # ==============
    
    # Add binned age and weight columns
    df['age_bin'] = df['agecalc_adm'].round()
    df['weight_bin'] = df['weight_kg_adm'].round()

    # fill missing values based on means of given group columns
    def fill_missing_with_group_mean(df, group_cols, target_cols):
        for col in target_cols:
            df[col] = df.groupby(group_cols)[col].transform(
                lambda x: x.fillna(x.mean())
            )

    # Fill missing values for features
    group_columns = ['age_bin', 'weight_bin']
    fill_missing_with_group_mean(df, group_columns, target_columns)

    # drop binned columns
    df.drop(columns=['age_bin', 'weight_bin'], inplace=True)

def separate_features(df):
    feats = df.columns
    num_feats = ['agecalc_adm', 'height_cm_adm', 'weight_kg_adm', 'muac_mm_adm',
        'hr_bpm_adm', 'rr_brpm_app_adm', 'sysbp_mmhg_adm',
        'diasbp_mmhg_adm', 'temp_c_adm', 'spo2site1_pc_oxi_adm',
        'spo2site2_pc_oxi_adm', 'spo2other_adm', 'hematocrit_gpdl_adm',
        'lactate_mmolpl_adm', 'lactate2_mmolpl_adm', 'glucose_mmolpl_adm',
        'sqi1_perc_oxi_adm', 'sqi2_perc_oxi_adm']

    cat_feats = ['sex_adm', 'spo2onoxy_adm', 'oxygenavail_adm', 'respdistress_adm',
        'caprefill_adm', 'bcseye_adm', 'bcsmotor_adm', 'bcsverbal_adm',
        'bcgscar_adm', 'vaccmeasles_adm', 'vaccpneumoc_adm', 'vaccdpt_adm',
        'priorweekabx_adm', 'priorweekantimal_adm', 'symptoms_adm___1',
        'symptoms_adm___2', 'symptoms_adm___3', 'symptoms_adm___4',
        'symptoms_adm___5', 'symptoms_adm___6', 'symptoms_adm___7',
        'symptoms_adm___8', 'symptoms_adm___9', 'symptoms_adm___10',
        'symptoms_adm___11', 'symptoms_adm___12', 'symptoms_adm___13',
        'symptoms_adm___14', 'symptoms_adm___15', 'symptoms_adm___16',
        'symptoms_adm___17', 'symptoms_adm___18', 'comorbidity_adm___1',
        'comorbidity_adm___2', 'comorbidity_adm___3',
        'comorbidity_adm___4', 'comorbidity_adm___5',
        'comorbidity_adm___6', 'comorbidity_adm___7',
        'comorbidity_adm___8', 'comorbidity_adm___9',
        'comorbidity_adm___10', 'comorbidity_adm___11',
        'comorbidity_adm___12', 'priorhosp_adm', 'prioryearwheeze_adm',
        'prioryearcough_adm', 'diarrheaoften_adm', 'tbcontact_adm',
        'feedingstatus_adm', 'exclbreastfed_adm', 'nonexclbreastfed_adm',
        'totalbreastfed_adm', 'birthdetail_adm___1', 'birthdetail_adm___2',
        'birthdetail_adm___3', 'birthdetail_adm___4',
        'birthdetail_adm___5', 'birthdetail_adm___6', 'traveldist_adm',
        'badhealthduration_adm', 'hivstatus_adm', 'malariastatuspos_adm']

    num_feats = list(set(feats)&set(num_feats))
    cat_feats = list(set(feats)&set(cat_feats))
    target_col = ['inhospital_mortality']

    return num_feats,cat_feats,target_col

def deal_with_remaining_missing_features(num_feats,cat_feats,df):
    # fill numeric na with mean of column if less than half na else drop, and drop categoric na
    to_remove = []
    for x in num_feats:
        if df[x].isna().sum()*2 < len(df):
            df[x] = df[x].fillna(df[x].mean(),inplace=False)
        else:
            to_remove.append(x)
    num_feats = list(set(num_feats).difference(set(to_remove)))
    cat_feats = list(set(cat_feats).difference(set(df.columns[df.isna().any()].tolist())))

    return num_feats,cat_feats

# Train your model.
def train_challenge_model(data_folder, model_folder, verbose):
    # Find the Challenge data.

    '''
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')
        
    patient_ids, data, label, features = load_challenge_data(data_folder)
    num_patients = len(patient_ids)

    if num_patients == 0:
        raise FileNotFoundError('No data is provided.')
        
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)
    
    # Train the models.
    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')
    '''

    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')

    tent_df = pd.read_csv(data_folder)

    preprocess_data(tent_df,target_columns=['momagefirstpreg_adm'])
    num_feats,cat_feats,target_col = separate_features(tent_df)
    num_feats,cat_feats = deal_with_remaining_missing_features(num_feats,cat_feats,tent_df)

    df = tent_df

    # split features, class
    X,y = df.drop(target_col,axis=1,inplace=False),df[target_col]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

    # xgb on numeric features
    xgb_model = XGBClassifier(n_estimators=100,max_depth=5,learning_rate=0.1)
    xgb_model.fit(X_train[num_feats],y_train)

    # logistic regression
    xgb_train = xgb_model.predict_proba(X_train[num_feats])[:,1]
    xgb_test = xgb_model.predict_proba(X_test[num_feats])[:,1]

    # discretize output as categorical feature
    X_train['xgb_probs'] = pd.cut(xgb_train,bins=5,labels=['A','B','C','D','E'])
    X_test['xgb_probs'] = pd.cut(xgb_test,bins=5,labels=['A','B','C','D','E'])

    # catboost
    cat_feats_with_xgb = cat_feats+['xgb_probs']
    cat_train = Pool(X_train[cat_feats_with_xgb],y_train,cat_features=cat_feats_with_xgb)
    cat_model = CatBoostClassifier(iterations=100,depth=5,learning_rate=0.1)
    cat_model.fit(cat_train)

    # Save the models.
    save_challenge_model(model_folder,(xgb_model,cat_model))

    if verbose >= 1:
        print('Done!')
        
# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_model(model_folder, verbose):
    if verbose >= 1:
        print('Loading the model...')

    xgb_model = XGBClassifier()
    xgb_model.load_model(os.path.join(model_folder,'xgb_model'))

    cat_model = CatBoostClassifier()
    cat_model.load_model(os.path.join(model_folder,'cat_model'))

    return xgb_model,cat_model

def run_challenge_model(model, data_folder, verbose):

    # Load data.
    patient_ids, data, label, features = load_challenge_data(data_folder)
    
    preprocess_data(data,target_columns=['momagefirstpreg_adm'])
    num_feats,cat_feats,target_col = separate_features(data)
    num_feats,cat_feats = deal_with_remaining_missing_features(num_feats,cat_feats,data)

    xgb_model,cat_model = model

    # get features in order
    xgb_feats = xgb_model.get_booster().feature_names
    # cat_feats = cat_model.get_feature_names()

    # logistic regression
    xgb_pred = xgb_model.predict_proba(data[xgb_feats])[:,1]

    # discretize output as categorical feature
    data['xgb_probs'] = pd.cut(xgb_pred,bins=5,labels=['A','B','C','D','E'])

    # catboost
    cat_feats_with_xgb = cat_feats+['xgb_probs']
    cat_preds = cat_model.predict(data[cat_feats_with_xgb])
    cat_probs = np.max(cat_model.predict_proba(data[cat_feats_with_xgb]),axis=1)

    return patient_ids, cat_preds, cat_probs

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder,model):
    xgb_model,cat_model = model
    xgb_model.save_model(os.path.join(model_folder,'xgb_model'))
    cat_model.save_model(os.path.join(model_folder,'cat_model'))
