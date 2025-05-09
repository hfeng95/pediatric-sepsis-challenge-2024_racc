import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

def plot_auprc(results_df, param_key='xgb_params', param_name='learning_rate', metric='mean_f1'):
    """
    Plot AUPRC vs a specific hyperparameter across trials.

    :param results_df: DataFrame from hyperparam_search()
    :param param_key: Key to the dictionary of parameters ('xgb_params' or 'cat_params')
    :param param_name: Name of the parameter to plot (e.g. 'learning_rate')
    """
    # Extract the desired parameter and AUPRC
    plot_df = results_df.copy()
    plot_df[param_name] = plot_df[param_key].apply(lambda d: json.loads(d.replace('\'','\"'))[param_name])
    
    if plot_df[param_name].isnull().all():
        print(f"⚠️ Parameter '{param_name}' not found in '{param_key}'.")
        return
    if not metric in plot_df.columns:
        print(f"⚠️ Metric '{metric}' not found.")
        return

    plt.figure(figsize=(8, 5))
    sns.pointplot(data=plot_df, x=param_name, y=metric, marker='o')
    plt.title(f"{metric} vs {param_name}")
    plt.xlabel(param_name)
    plt.ylabel(metric)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    results_df = pd.read_csv('model_results.csv')
    plot_auprc(results_df, param_key='cat_params', param_name='iterations',metric='mean_fb')

