#!/usr/bin/env python

# Do *not* edit this script.
# These are helper functions that you can use with your code.
# Check the example code to see how to import these functions to your code.

import os, numpy as np, scipy as sp, scipy.io
import pandas as pd
import psutil

### Challenge data I/O functions

def load_challenge_labels(data_folder):
  
        data = pd.read_csv(data_folder)
        label = data['inhospital_mortality']
        patient_ids = data['studyid_adm']
        
        return patient_ids, label


def load_challenge_data(data_folder):
    data = pd.read_csv(data_folder)
    
    # Check if the expected columns are present; if not, assign NA.
    label = data['inhospital_mortality'] if 'inhospital_mortality' in data.columns else pd.NA
    patient_ids = data['studyid_adm'] if 'studyid_adm' in data.columns else pd.NA
    
    # Drop the columns if they exist in the DataFrame.
    cols_to_drop = [col for col in ['studyid_adm', 'inhospital_mortality'] if col in data.columns]
    data = data.drop(columns=cols_to_drop)
    
    features = data.columns
    return patient_ids, data, label, features
'''
def load_challenge_data(data_folder):
  
        data = pd.read_csv(data_folder)

        label = data['inhospital_mortality']
        patient_ids = data['studyid_adm']
        data = data.drop(['studyid_adm','inhospital_mortality'], axis=1)
        features = data.columns
        
        return patient_ids, data, label, features
'''
def load_challenge_testdata(data_folder, selected_columns=None):
    """
    Loads test data from a CSV file.

    Parameters:
      - data_folder (str): Path to the CSV file.
      - selected_columns (list of str, optional): List of column names to load. If provided, only these columns 
        will be returned (the 'studyid_adm' column is always used for patient IDs).

    Returns:
      - patient_ids: Series containing patient IDs.
      - data: DataFrame with the selected feature columns.
      - features: The columns of the returned DataFrame.
    """
    data = pd.read_csv(data_folder)
    patient_ids = data['studyid_adm']
    
    # Remove the patient ID column.
    data = data.drop(['studyid_adm'], axis=1)
    
    # If a list of selected columns is provided, subset the data accordingly.
    if selected_columns is not None:
        # Ensure only columns present in the data are used.
        selected_columns = [col for col in selected_columns if col in data.columns]
        data = data[selected_columns]
        
    features = data.columns
    return patient_ids, data, features

 
# Save the Challenge outputs for one file.
def save_challenge_outputs(output_folder, patient_ids, prediction_binary, prediction_probability):
    
    if output_folder is not None:
      with open(output_folder, 'w') as f:
          f.write('PatientID|PredictedProbability|PredictedBinary\n')
          for (i, p, b) in zip(patient_ids, prediction_probability, prediction_binary):
              f.write('%d|%g|%d\n' % (i, p, b))
              
    return None
  

# Load the Challenge predictions for all of the patients.
def load_challenge_predictions(folder):
    with open(folder, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        predictions = np.genfromtxt(f, delimiter='|')
    
    patient_ids = predictions[:,0].astype(int)
    prediction_probability = predictions[:,1]
    prediction_binary = predictions[:,2].astype(int)
    
    return patient_ids, prediction_probability, prediction_binary
  
  
### Other helper functions

# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False

# Check if a variable is a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# Check if a variable is a NaN (not a number) or represents a NaN.
def is_nan(x):
    if is_number(x):
        return np.isnan(float(x))
    else:
        return False

# Remove any quotes, brackets (for singleton arrays), and/or invisible characters.
def remove_extra_characters(x):
    return str(x).replace('"', '').replace("'", "").replace('[', '').replace(']', '').replace(' ', '').strip()

# Sanitize boolean values, e.g., from the Challenge outputs.
def sanitize_boolean_value(x):
    x = remove_extra_characters(x)
    if (is_finite_number(x) and float(x)==0) or (x in ('False', 'false', 'F', 'f')):
        return 0
    elif (is_finite_number(x) and float(x)==1) or (x in ('True', 'true', 'T', 't')):
        return 1
    else:
        return float('nan')

# Santize scalar values, e.g., from the Challenge outputs.
def sanitize_scalar_value(x):
    x = remove_extra_characters(x)
    if is_number(x):
        return float(x)
    else:
        return float('nan')

# Function to compute normalized compute resource usage
def compute_resource():
    """
    Compute normalized compute resource usage as a penalty metric.

    Uses psutil to track memory and CPU usage dynamically.
    
    Returns:
    - float: Normalized compute resource usage.
    """
    process = psutil.Process(os.getpid())  # Get current process
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
    cpu_time = process.cpu_times().user + process.cpu_times().system  # Total CPU time in seconds
    return memory_usage, cpu_time

def read_selected_variables(model, model_folder):
    """
    Returns the list of selected variables used during training.

    It first checks if the model dictionary contains the key 'selected_variables'.
    If not, it attempts to read the list from 'selected_features.txt' in the model folder.
    If both methods fail, it returns None (which indicates that all features should be used).

    Parameters:
      - model (dict): The model dictionary which may contain 'selected_variables'.
      - model_folder (str): The path to the folder where 'selected_features.txt' is stored.

    Returns:
      - list or None: A list of selected variable names if available; otherwise, None.
    """
    # Try to get from the model dictionary.
    try:
        if model is not None and "selected_variables" in model:
            return model["selected_variables"]
    except TypeError:
        print("Warning: Model does not have 'selected_variables' key. Attempting to read from file.")
    
    # Otherwise, try to read from the file.
    file_path = os.path.join(model_folder, 'selected_variables.txt')
    try:
        with open(file_path, 'r') as f:
            selected_vars = f.read().splitlines()
        return selected_vars
    except Exception as e:
        print(f"Warning: Could not read selected variables from {file_path}. Using all features. Error: {e}")
        return None


# Function to compute threshold from predictions
def compute_threshold_from_predictions(prediction_probability, prediction_binary):
    """
    Computes a threshold from the predicted probabilities and binary predictions.
    
    The threshold is computed as the average of:
      - the minimum probability among instances predicted as 1, and
      - the maximum probability among instances predicted as 0.
    
    If there are no predicted positives, returns 1.0; if no predicted negatives, returns 0.0.
    """
    pred_probability = np.array(prediction_probability)
    pred_binary = np.array(prediction_binary)
    pos_probs = pred_probability[pred_binary == 1]
    neg_probs = pred_probability[pred_binary == 0]
    
    if len(pos_probs) == 0:
        return 1.0
    if len(neg_probs) == 0:
        return 0.0
    return (pos_probs.min() + neg_probs.max()) / 2.0    
# Function to read threshold from the file, or compute from predictions if not available.
def read_or_compute_threshold(threshold_file, prediction_probability, prediction_binary):
    """
    Attempts to read the classification threshold from the given file.
    If not available, computes an approximate threshold from prediction probabilities and binary predictions.
    
    Parameters:
      - threshold_file (str or None): Path to the threshold file.
      - prediction_probability (array-like): Predicted probabilities.
      - prediction_binary (array-like): Predicted binary outcomes.
    
    Returns:
      - float: The threshold.
    """
    if threshold_file is None:
        print("No threshold file provided; computing threshold from predictions.")
        threshold =compute_threshold_from_predictions(prediction_probability, prediction_binary)
        print(threshold)
        return threshold
    try:
        with open(threshold_file, 'r') as f:
            lines = f.readlines()
            threshold = float(lines[0].strip())
        return threshold
    except Exception as e:
        print(f"Warning: Could not read threshold_file '{threshold_file}', cannot compute using 0.5. Error: {e}")
        return 0.5
