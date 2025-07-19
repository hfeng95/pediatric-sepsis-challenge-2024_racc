import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,ParameterGrid
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,fbeta_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier,Pool
from team_code import preprocess_data,separate_features,deal_with_remaining_missing_features,check_class_balance,discretize_probs
import os

def run_experiment(df,target_col,num_feats,cat_feats,xgb_params,cat_params,num_bins=5):
    X,y = df.drop(columns=target_col,axis=1,inplace=False),df[target_col]

    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,stratify=y)
    class_weights = check_class_balance(target_col,df)

    # Train XGBoost
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_train[num_feats],y_train)

    # Get train + val probabilities
    train_probs = xgb.predict_proba(X_train[num_feats])[:,1]
    val_probs = xgb.predict_proba(X_val[num_feats])[:,1]

    # Bin them
    train_bins,bin_edges = discretize_probs(train_probs,num_bins)
    # print(np.unique(train_bins,return_counts=True)[1])
    val_bins = pd.cut(val_probs,bins=bin_edges,labels=False,include_lowest=True).astype(np.int64)
    # print(np.unique(val_bins,return_counts=True)[1])

    X_train['xgb_probs'] = train_bins
    X_val['xgb_probs'] = val_bins

    full_cat_feats = cat_feats+['xgb_probs']
    train_pool = Pool(X_train[full_cat_feats],y_train,cat_features=full_cat_feats)
    val_pool = Pool(X_val[full_cat_feats],y_val,cat_features=full_cat_feats)

    cat = CatBoostClassifier(**cat_params,verbose=0)
    cat.fit(train_pool)

    val_preds = cat.predict(val_pool)
    val_proba = cat.predict_proba(val_pool)[:,1]

    return {
        'f1': f1_score(y_val,val_preds),
        'fb': fbeta_score(y_val,val_preds,beta=2),
        'auc': roc_auc_score(y_val,val_proba),
        'auprc': average_precision_score(y_val,val_proba),
        'xgb_params': xgb_params,
        'cat_params': cat_params
    }

def hyperparam_search(df,target_col,num_feats,cat_feats,xgb_grid,cat_grid,num_bins=5,num_runs=10):
    results = []
    for xgb_params in ParameterGrid(xgb_grid):
        for cat_params in ParameterGrid(cat_grid):
            print(f"Testing XGB {xgb_params} + CAT {cat_params} over {num_runs} runs...")
            aucs,f1s,auprcs,fbs = [],[],[],[]
            for run in range(num_runs):
                try:
                    result = run_experiment(df.copy(),target_col,num_feats,cat_feats,xgb_params,cat_params,num_bins=num_bins)
                    aucs.append(result['auc'])
                    f1s.append(result['f1'])
                    auprcs.append(result['auprc'])
                    fbs.append(result['fb'])
                except Exception as e:
                    print("Error during experiment:",e)

            if aucs:  # skip if all runs failed for some reason
                mean_auc = np.mean(aucs)
                std_auc = np.std(aucs)
                mean_f1 = np.mean(f1s)
                std_f1 = np.std(f1s)
                mean_auprc = np.mean(auprcs)
                std_auprc = np.std(auprcs)
                mean_fb = np.mean(fbs)
                std_fb = np.std(fbs)

                print(f"Avg AUC: {mean_auc:.4f} ± {std_auc:.4f}, Avg F1: {mean_f1:.4f} ± {std_f1:.4f}, Avg AUPRC: {mean_auprc:.4f} ± {std_auprc:.4f}, Avg Fβ: {mean_fb:.4f} ± {std_fb:.4f}")

                results.append({
                    'xgb_params': xgb_params,
                    'cat_params': cat_params,
                    'mean_auc': mean_auc,
                    'std_auc': std_auc,
                    'mean_f1': mean_f1,
                    'std_f1': std_f1,
                    'mean_auprc': mean_auprc,
                    'std_auprc': std_auprc,
                    'mean_fb': mean_fb,
                    'std_fb': std_fb
                })

    return pd.DataFrame(results)

if __name__ == '__main__':
    df = pd.read_csv('SyntheticData_Training.csv')
    target_col = 'inhospital_mortality'
    xgb_grid = {
        'n_estimators': [200,500,900],
        'max_depth': [5,9,13],
        'learning_rate': [0.1],
        'scale_pos_weight': [10]
    }

    cat_grid = {
        'iterations': [150,300,450,900],
        'depth': [5,7,9],
        'learning_rate': [0.1],
        'class_weights': [[1,20]],
        'early_stopping_rounds': [10]
    }

    preprocess_data(df,target_columns=['momagefirstpreg_adm'])
    num_feats,cat_feats,target_col = separate_features(df)
    num_feats,cat_feats = deal_with_remaining_missing_features(num_feats,cat_feats,df)
    results = hyperparam_search(df,target_col,num_feats,cat_feats,xgb_grid,cat_grid,num_bins=2,num_runs=10)
    results.sort_values(by='mean_fb',ascending=False).to_csv('model_results.csv',index=False)