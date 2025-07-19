import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import Booster
from catboost import CatBoostClassifier

# Load models
from team_code import load_challenge_model

# Path to saved models
model_path = 'model'
xgb_model, cat_model,_,_ = load_challenge_model(model_path,verbose=1)

# === XGBoost Feature Importances ===
print("\n🔷 XGBoost Feature Importances")
xgb_importance = xgb_model.get_booster().get_score(importance_type='gain')
xgb_importance_df = pd.DataFrame.from_dict(xgb_importance, orient='index', columns=['Importance'])
xgb_importance_df.index.name = 'Feature'
xgb_importance_df = xgb_importance_df.sort_values(by='Importance', ascending=False)
print(xgb_importance_df)

# === CatBoost Feature Importances ===
print("\n🟣 CatBoost Feature Importances")
cat_importances = cat_model.get_feature_importance(prettified=True)
print(cat_importances[['Feature Id', 'Importances']].sort_values('Importances', ascending=False))

# === Plot ===
def plot_importances(importance_df, model_name, top_n=20):
    importance_df = importance_df.sort_values(by=importance_df.columns[-1], ascending=True).tail(top_n)
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df.iloc[:,0] if model_name == 'CatBoost' else importance_df.index, importance_df.iloc[:,-1])
    plt.xlabel('Importance')
    plt.title(f'{model_name} Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.show()

# Plot
# plot_importances(xgb_importance_df.reset_index(), 'XGBoost')
# plot_importances(cat_importances[['Feature Id', 'Importances']], 'CatBoost')
