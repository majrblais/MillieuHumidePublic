# FeatureReduction

This directory contains scripts and resources for applying feature reduction techniques to the regression models. The goal is to identify the most important features while maintaining or improving model performance.

## Contents

### Subfolders

1. **results**
   - Contains subfolders for each ecosystem function, storing the output results from the feature reduction processes.

### Scripts

1. **feature_selection_graphs_save_csv_results.ipynb**
   - **Purpose**: Generates feature reduction algorithms.
   - **Inputs**: Takes the `best_modelsXXX_Benefit_info.csv` file for each ecosystem function.
   - **Outputs**: 
     - Figures showing the results of the three feature reduction techniques for each data CSV.
     - CSV files named `test_results_XXX.csv` containing columns for actual, predicted, RMSE, number of features, and features used.

2. **statistics_best_features.ipynb**
   - **Purpose**: Reads the `test_results_XXX.csv` file and calculates statistics on the most important features and their occurrences.
   - **Outputs**: 
     - Graphs showing the statistics of the most important features.
     - Saved graphs in the `results` subfolders.

3. **best_models_ensemble_learning.ipynb**
   - **Purpose**: Takes the top 3 models (lowest RMSE) for each reduction technique from `test_results_XXX.csv` and plots the validation results.
   - **Outputs**: 
     - Graphs showing the validation results for the top models and an ensemble approach with each model having 1/9 weight.
     - Saved graphs in the `results` subfolders.

4. **grouping.ipynb**
   - **Purpose**: Reads the `test_results_XXX.csv` file and calculates the accuracy of classifying the ratings into lower, moderate, and higher for each feature count and reduction technique.
   - **Outputs**: 
     - CSV files with the accuracy results.
     - Printed global accuracy and accuracies for lower, moderate, and higher classes.

5. **grouping_ensemble.ipynb**
   - **Purpose**: Similar to `grouping.ipynb`, but performs ensemble learning using the 5 best models from the `test_results_XXX.csv` before grouping.
   - **Outputs**: 
     - CSV files with the accuracy results.
     - Printed global accuracy and accuracies for lower, moderate, and higher classes.

## Usage

### Feature Reduction
1. Run `feature_selection_graphs_save_csv_results.ipynb` to generate feature reduction results for each ecosystem function. This will create figures and CSV files in the `results` subfolders.
2. Use `statistics_best_features.ipynb` to calculate and visualize the most important features from the `test_results_XXX.csv` files.

### Ensemble Learning and Grouping
1. Use `best_models_ensemble_learning.ipynb` to plot validation results for the top models and apply ensemble learning.
2. Use `grouping.ipynb` to calculate and save the accuracy of classifying ratings based on the feature reduction results.
3. Use `grouping_ensemble.ipynb` to perform ensemble learning before grouping and calculating accuracy.

## Future Enhancements
- Implementing more advanced feature reduction techniques.
- Improving ensemble learning methods and their visualization.
- Extending the grouping and classification accuracy analysis.

For any issues or contributions, please open an issue or submit a pull request on GitHub.
