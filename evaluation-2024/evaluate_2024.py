#!/usr/bin/env python

import os
import sys
import numpy as np
import argparse
import time
import json
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
from helper_code import *
import psutil


# Function to calculate Net Benefit
def calculate_net_benefit(y_true, y_pred, threshold=0.5):
    """
    Calculate the net benefit of predictions.
    
    Parameters:
    - y_true (array-like): Ground truth binary labels (0 or 1).
    - y_pred (array-like): Predicted probabilities.
    - threshold (float): Threshold probability for classification (default is 0.5).
    
    Returns:
    - float: Net benefit score.
    """
    tp = np.sum((y_true == 1) & (y_pred >= threshold))
    fp = np.sum((y_true == 0) & (y_pred >= threshold))
    n = len(y_true)
    
    # Calculate Net Benefit
    net_benefit = (tp / n) - ((threshold / (1 - threshold)) * (fp / n))
    return net_benefit


# Function to calculate ECE (Estimated Calibration Error)
def calculate_ece(probs, labels, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        bin_mask = (probs > bin_edges[i]) & (probs <= bin_edges[i + 1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(labels[bin_mask])
            bin_conf = np.mean(probs[bin_mask])
            ece += bin_size * np.abs(bin_acc - bin_conf) / len(probs)
    return ece


# Function to read inference time from the output folder
def read_inference_time(output_folder):
    """
    Reads the recorded inference time from the output folder.

    Parameters:
    - output_folder (str): Path to the folder containing inference time file.

    Returns:
    - float: Recorded inference time in seconds.
    """
    inference_time_file = os.path.join(output_folder, 'inference_time.txt')
    if not os.path.exists(inference_time_file):
        raise FileNotFoundError(f"Inference time file not found in {output_folder}")
    
    with open(inference_time_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Inference time:"):
                return float(line.split(":")[1].strip().split()[0])  # Extracts the time in seconds

    raise ValueError("Inference time not found in the file.")


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


# Helper functions
def compute_confusion_matrix(labels, predictions):
    labels = np.array(labels).astype(int)
    predictions = np.array(predictions).astype(int)
    cm = np.zeros((2, 2))
    for i in range(len(labels)):
        cm[labels[i]][predictions[i]] += 1
    return cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]


def compute_accuracy(tn, fp, fn, tp):
    return (tp + tn) / (tn + fp + fn + tp)


# Function to evaluate model
def evaluate_model(label_folder, output_folder, inference_time_file):
    # Load labels and model outputs.
    _, _, label, _ = load_challenge_data(label_folder)
    patient_ids, prediction_probability, prediction_binary = load_challenge_predictions(output_folder)

    # Read inference time from the file
    with open(inference_time_file, 'r') as f:
        lines = f.readlines()
        inference_time = float(lines[0].split(":")[1].strip().split()[0])  # Extract inference time

    num_predictions = len(prediction_binary)

    # Compute confusion matrix and metrics
    tn, fp, fn, tp = compute_confusion_matrix(label, prediction_binary)
    accuracy = compute_accuracy(tn, fp, fn, tp)

    # Additional metrics
    auc_score = roc_auc_score(label, prediction_probability)
    precision, recall, _ = precision_recall_curve(label, prediction_probability)
    auprc = auc(recall, precision)
    net_benefit = calculate_net_benefit(label, prediction_probability)
    ece = calculate_ece(prediction_probability, label)

    # Compute normalized metrics
    inference_time = inference_time / 1200
    compute = compute_resource()  # Dynamically tracked using psutil

    # Scores 
    return {
        'AUC': auc_score,
        'AUPRC': auprc,
        'Net Benefit': net_benefit,
        'ECE': ece,
        'Inference Time': inference_time,
        'Compute': compute
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance.")
    parser.add_argument("label_folder", type=str, help="Folder containing ground truth labels.")
    parser.add_argument("output_folder", type=str, help="Folder containing model predictions.")
    parser.add_argument("inference_time_file", type=str, help="File containing inference time information.")
    parser.add_argument("output_file", type=str, nargs="?", help="File to save the evaluation results.")

    args = parser.parse_args()

    # Perform evaluation
    metrics = evaluate_model(args.label_folder, args.output_folder, args.inference_time_file)

    # Create score dictionary
    submission_result = {
        'score': {
            'AUC': metrics['AUC'],
            'AUPRC': metrics['AUPRC'],
            'Net Benefit': metrics['Net Benefit'],
            'ECE': metrics['ECE'],
            'Inference Time': metrics['Inference Time'],
            'Compute': metrics['Compute']
        },
        'completion_time': time.strftime('%Y-%m-%dT%H:%M:%SZ')
    }

    # Print or save the results
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(json.dumps(submission_result)) 
    else:
        print(json.dumps(submission_result))  
