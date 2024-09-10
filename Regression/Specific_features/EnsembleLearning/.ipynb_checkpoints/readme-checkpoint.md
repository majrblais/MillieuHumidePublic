# EnsembleLearning

This directory contains scripts and resources for performing ensemble learning on the regression models trained on all features from the `Training` folder. It includes methods for loading and retraining models, as well as grouping and classifying results.

## Contents

### Subfolders

1. **graphs**
   - Stores the outputs of the ensemble learning processes, including graphs and saved model predictions.

### Scripts

1. **load_best_ensemble.ipynb**
   - **Purpose**: Loads the best models for an ecosystem function using the `best_modelxxx_info.csv` file, where `xxx` is the function. Combines the models using ensemble learning (average) and saves the results on a graph with actual vs. predicted values.

2. **retrain_model_ensemble.ipynb**
   - **Purpose**: Retrains the best models for an ecosystem function rather than loading them. Combines the retrained models using ensemble learning (average) and saves the results on a graph with actual vs. predicted values.



## Usage

### Performing Ensemble Learning
1. Use `load_best_ensemble.ipynb` to load the best models and perform ensemble learning using the average method. The results will be saved as graphs in the `results` folder.
2. Alternatively, use `retrain_model_ensemble.ipynb` to retrain the best models before performing ensemble learning. The results will be saved as graphs in the `results` folder.

## Future Enhancements
- Exploring additional ensemble learning techniques.
- Enhancing the methods for grouping and classifying results.
- Improving visualization methods for better analysis of ensemble learning performance.
- Fix path errors

For any issues or contributions, please open an issue or submit a pull request on GitHub.
