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
from sklearn.metrics import accuracy_score,recall_score
from sklearn.metrics import  roc_curve
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

val_status = False

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
    mean_heights = data_binned.groupby('agecalc_adm')['height_cm_adm'].transform('mean')
    std_heights = data_binned.groupby('agecalc_adm')['height_cm_adm'].transform('std')
    
    data_binned['weight_kg_adm'] = data_binned['weight_kg_adm'].fillna(mean_weights)
    df['weight_kg_adm'] = df['weight_kg_adm'].fillna(mean_weights)
    data_binned['height_cm_adm'] = data_binned['height_cm_adm'].fillna(mean_heights)
    df['height_cm_adm'] = df['height_cm_adm'].fillna(mean_heights)

    for idx, row in data_binned.iterrows():
        age = row['agecalc_adm']
        weight = row['weight_kg_adm']
        weight_mean = mean_weights.loc[idx]
        weight_std = std_weights.loc[idx]
        height = row['height_cm_adm']
        height_mean = mean_heights.loc[idx]
        height_std = std_heights.loc[idx]
        
        if weight > weight_mean + 2.5 * weight_std:
            weight_in_kg = weight * 0.453592  # Convert pounds to kg
            data_binned.loc[idx, 'weight_kg_adm'] = weight_in_kg
            df.loc[idx, 'weight_kg_adm'] = weight_in_kg
            
        if abs(height - height_mean) > 3 * height_std:
            data_binned.loc[idx, 'height_cm_adm'] = height_mean
            df.loc[idx, 'height_cm_adm'] = height_mean 
    # ==============
    
    # Add binned age and weight/height columns
    df['age_bin'] = df['agecalc_adm'].round()
    df['weight_bin'] = df['weight_kg_adm'].round()
    df['height_bin'] = df['height_cm_adm'].round()

    # fill missing values based on means of given group columns
    def fill_missing_with_group_mean(df, group_cols, target_cols):
        for col in target_cols:
            df[col] = df.groupby(group_cols)[col].transform(
                lambda x: x.fillna(x.mean())
            )

    # Fill missing values for features
    group_columns = ['age_bin', 'weight_bin', 'height_bin']
    fill_missing_with_group_mean(df, group_columns, target_columns)

    # drop binned columns
    df.drop(columns=['age_bin', 'weight_bin', 'height_bin'], inplace=True)

def separate_features(df):
    feats = df.columns
    num_feats = ['agecalc_adm', 'height_cm_adm', 'weight_kg_adm', 'muac_mm_adm',
        'hr_bpm_adm', 'rr_brpm_app_adm', 'sysbp_mmhg_adm', 'blantyre_score',
        'diasbp_mmhg_adm', 'temp_c_adm', 'overall_spo2_mean', 'hematocrit_gpdl_adm',
        'lactate_mmolpl_adm', 'lactate2_mmolpl_adm', 'glucose_mmolpl_adm',
        'sqi1_perc_oxi_adm', 'sqi2_perc_oxi_adm']

    cat_feats = ['sex_adm', 'oxygenavail_adm', 'respdistress_adm',
        'caprefill_adm',
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

def merge_oxygen(df):
    df['spo2_combined_adm'] = df[['spo2site1_pc_oxi_adm', 'spo2site2_pc_oxi_adm', 'spo2other_adm']].mean(axis=1, skipna=True)
    overall_spo2_mean = df['spo2_combined_adm'].mean()
    df['spo2_combined_adm'] = df['spo2_combined_adm'].fillna(overall_spo2_mean)

    df.drop(columns=['spo2site1_pc_oxi_adm', 'spo2site2_pc_oxi_adm', 'spo2other_adm'], inplace=True)

def bcs_score(df):
    # Define the mappings
    eye_map = {
        'Watches or follows': 1,
        'Fails to watch or follow': 0
    }

    motor_map = {
        'Localizes painful stimulus': 2,
        'Withdraws limb from painful stimulus': 1,
        'No response or inappropriate response': 0
    }

    verbal_map = {
        'Cries appropriately with pain, or, if verbal, speaks': 2,
        'Moan or abnormal cry with pain': 1,
        'No vocal response to pain': 0
    }

    # Map textual values to scores
    df['eye_score'] = df['bcseye_adm'].map(eye_map)
    df['motor_score'] = df['bcsmotor_adm'].map(motor_map)
    df['verbal_score'] = df['bcsverbal_adm'].map(verbal_map)

    # Define function to calculate the BCS per row
    def calc_bcs(row):
        eye = row['eye_score']
        motor = row['motor_score']
        verbal = row['verbal_score']

        components = [eye, motor, verbal]
        present = [x for x in components if not np.isnan(x)]

        if len(present) == 0:
            return 5
        if len(present) == 1:
            return present[0] * 3
        if len(present) == 2:
            avg = sum(present) / 2
            total = sum(present) + avg
            return round(total)
        return eye + motor + verbal

    df['blantyre_score'] = df.apply(calc_bcs, axis=1)
    df['blantyre_score'] = df['blantyre_score'].clip(0, 5).astype('Int64')

    df.drop(columns=['eye_score', 'motor_score', 'verbal_score', 'bcseye_adm', 'motor_score', 'verbal_score'], inplace=True)

    return df

def combine_lactate(df):
    overall_mean = pd.concat([df['lactate_mmolpl_adm'], df['lactate2_mmolpl_adm']]).mean()

    def combine_row(row):
        a = row['lactate_mmolpl_adm']
        b = row['lactate2_mmolpl_adm']

        if pd.notna(a) and pd.notna(b):
            return (a + b) / 2
        elif pd.notna(a):
            return a
        elif pd.notna(b):
            return b
        else:
            return overall_mean

    df['lactate_combined'] = df.apply(combine_row, axis=1)

    df.drop(columns=['lactate_mmolpl_adm', 'lactate2_mmolpl_adm'],inplace=True)

    return df

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

def deal_with_testing_data(num_feats,cat_feats,df):
    # TODO: make sure data types are correct
    # fill numeric na with mean, fill categoric na with mode
    for x in num_feats:
        df[x] = df[x].fillna(df[x].mean(),inplace=False)
    for x in cat_feats:
        df[x] = df[x].fillna(df[x].mode()[0],inplace=False)

def check_class_balance(target_col,df):
    return df[target_col].value_counts().tolist()

def discretize_probs(probs,num_bins=5):
    binned,edges = pd.qcut(probs,q=num_bins,retbins=True,labels=False)
    edges[0],edges[-1] = 0,1
    return binned.astype(np.int64),edges

def sensitivity_metric(y_pred,dtrain):
    y_true = dtrain.get_label()
    # Convert predicted probabilities to binary class using threshold
    y_pred_labels = (y_pred>=0.5).astype(int)
    sensitivity = recall_score(y_true,y_pred_labels)
    return 'sensitivity',sensitivity

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

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    tent_df = pd.read_csv(data_folder)
    class_weights = check_class_balance('inhospital_mortality',tent_df)

    # preprocess
    unwanted_features = ['symptoms_adm___18','symptoms_adm___17','comorbidity_adm___4','comorbidity_adm___3',
                         'comorbidity_adm___9','comorbidity_adm___8','comorbidity_adm___4','symptoms_adm___16',
                         'birthdetail_adm___4','birthdetail_adm___2','height_cm_adm','hematocrit_gpdl_adm',
                         'symptoms_adm___15','birthdetail_adm___6','comorbidity_adm___2','comorbidity_adm___10',
                         'comorbidity_adm___6']
    preprocess_data(tent_df,target_columns=['momagefirstpreg_adm'])
    merge_oxygen(tent_df)
    bcs_score(tent_df)
    combine_lactate(tent_df)
    num_feats,cat_feats,target_col = separate_features(tent_df)
    num_feats,cat_feats = deal_with_remaining_missing_features(num_feats,cat_feats,tent_df)
    selected_features = [col for col in tent_df.columns if col not in unwanted_features]
    num_feats = [col for col in num_feats if col not in unwanted_features]
    cat_feats = [col for col in cat_feats if col not in unwanted_features]
    df = tent_df[selected_features]

    # for validation only
    if val_status:
        df_train,df_test = train_test_split(df,stratify=df[target_col],test_size=0.2)
        df_train.to_csv('train_out.csv',index=False)
        df_test.to_csv('test_out.csv',index=False)
        df = df_train

    # split features, class
    X_train,y_train = df.drop(target_col,axis=1,inplace=False),df[target_col]

    if verbose >= 1:
        print('Training the Challenge models on the Challenge data...')

    # xgb on numeric features
    xgb_model = XGBClassifier(n_estimators=900,max_depth=13,learning_rate=0.1,scale_pos_weight=10)
    xgb_model.fit(X_train[num_feats],y_train)

    # logistic regression
    xgb_train = xgb_model.predict_proba(X_train[num_feats])[:,1]

    # discretize output as categorical feature
    num_bins = 2
    train_bins,bin_edges = discretize_probs(xgb_train,num_bins)
    if verbose >= 1:
        print('Splitting xgb output into bins: ',bin_edges)
    X_train['xgb_probs'] = train_bins

    # catboost
    cat_feats_with_xgb = cat_feats+['xgb_probs']
    cat_train = Pool(X_train[cat_feats_with_xgb],y_train,cat_features=cat_feats_with_xgb)
    cat_model = CatBoostClassifier(iterations=900,depth=9,learning_rate=0.1,class_weights=[1,20])
    cat_model.fit(cat_train,early_stopping_rounds=10)

    # prob threshold for catboost
    threshold = 3.0000333429776503e-05

    # Save the models.
    save_challenge_model(model_folder,(xgb_model,cat_model,bin_edges,threshold,selected_features))

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

    bin_edges = np.load(os.path.join(model_folder,'bin_edges.npy'))

    threshold = np.load(os.path.join(model_folder,'cat_threshold.npy'))

    selected_features = np.load(os.path.join(model_folder,'selected_features.npy')).tolist()

    return xgb_model,cat_model,bin_edges,threshold,selected_features

def find_threshold_for_sensitivity(y, p, thr=None, min_sens=0.8):
    if thr is not None:
        return thr
    fpr,tpr,ths=roc_curve(y,p)
    valid=ths[tpr>=min_sens]
    return float(valid.max()) if len(valid)>0 else 0.5

def run_challenge_model(model, data_folder, verbose):

    xgb_model,cat_model,bin_edges,threshold,selected_features = model

    # Load data.
    if val_status:
        patient_ids, data, label, features = load_challenge_data(data_folder)
    else:
        patient_ids, data, features = load_challenge_testdata(data_folder,selected_columns=selected_features)
        label = None

    # get features in order
    xgb_feats = xgb_model.get_booster().feature_names
    cat_feats_with_xgb = cat_model.feature_names_

    # process data
    deal_with_testing_data(xgb_feats,[x for x in cat_feats_with_xgb if x != 'xgb_probs'],data)

    # logistic regression
    xgb_pred = xgb_model.predict_proba(data[xgb_feats])[:,1]

    # discretize output as categorical feature
    if verbose >= 1:
        print('Using bins: ',bin_edges)
    data['xgb_probs'] = pd.cut(xgb_pred,bins=bin_edges,labels=False,include_lowest=True).astype(np.int64)

    # catboost
    if verbose:
        print('Loaded threshold:',threshold)
    cat_model.set_probability_threshold(threshold.tolist())
    cat_preds = cat_model.predict(data[cat_feats_with_xgb])
    cat_probs = cat_model.predict_proba(data[cat_feats_with_xgb])[:,1]

    # custom threshold for min sensitivity, used in validation only
    if val_status:
        threshold = find_threshold_for_sensitivity(label, cat_probs, min_sens=0.85)
        if verbose:
            print('Using threshold:',threshold)
        cat_preds = (cat_probs >= threshold).astype(int)

    return patient_ids, cat_preds, cat_probs

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder,model):
    xgb_model,cat_model,bin_edges,threshold,selected_features = model
    xgb_model.save_model(os.path.join(model_folder,'xgb_model'))
    cat_model.save_model(os.path.join(model_folder,'cat_model'))

    # save bin data
    np.save(os.path.join(model_folder,'bin_edges.npy'),bin_edges)

    # save prob threshold
    np.save(os.path.join(model_folder,'cat_threshold.npy'),threshold)

    # save features selection
    np.save(os.path.join(model_folder,'selected_features.npy'),np.array(selected_features))
