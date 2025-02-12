# Scoring code for The 2024 Pediatric Sepsis Challenge

This repository contains the Python evaluation code for The 2024 Pediatric Sepsis Challenge.

The `evaluate_2024` script evaluates the outputs of your models using the evaluation metric that is described on the [README](https://github.com/Kamaleswaran-Lab/The-2024-Pediatric-Sepsis-Challenge/blob/main/README.md) for the 2024 Challenge. This script reports multiple evaluation metrics, so check the [leaderboard scoring app](https://leaderboard-scoring-peds-sepsis-data-challenge-2024.streamlit.app) to see how we evaluate and rank your models.

## Python

You can run the Python evaluation code by installing the NumPy package and running the following command in your terminal:

    python evaluate_2024.py test_data/file_with_labels test_outputs/outputs.txt test_outputs/inference_time.txt scores.json

where
- `test_data` (input; required) is a folder containing file_with_labels.
- `file_with_labels` (input; required) is a file with labels for the data, such as a dummy test set;
- `test_outputs` (input; required) is a folder containing files with your model's outputs as outputs.txt and inference time as inference_time.txt; and
- `scores.json` (output; optional) is a collection of scores for your model.


## Troubleshooting

Unable to run this code with your code? Try one of the [example codes](https://github.com/Kamaleswaran-Lab/The-2024-Pediatric-Sepsis-Challenge/tree/main/python-example-2023) on the [training data](https://github.com/Kamaleswaran-Lab/The-2024-Pediatric-Sepsis-Challenge/tree/main/SyntheticData_Training.csv). Unable to install or run Python? Try [Python](https://www.python.org/downloads/), [Anaconda](https://www.anaconda.com/products/individual), or your package manager.

## How do I learn more?

Please post questions and concerns on the [Challenge discussion forum](https://groups.google.com/g/2024-pediatric-sepsis-data-challenge).

