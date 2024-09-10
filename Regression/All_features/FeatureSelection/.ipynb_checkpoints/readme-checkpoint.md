# Feature Reduction

This folder is used for performing feature reduction, which involves training the models from the `Training` folder with fewer features while achieving similar RMSE.

## Directory Structure

- `csvs`: Contains multiple CSV files (one for each Function/Benefit, e.g., NR/PR...) based on the results from `Training`. Each CSV has a row for each fill method, along with the best performing model and its hyperparameters (based on the lowest RMSE).

- `feature_selection_graphs_save_csv_results.ipynb`: This script loads the CSV files from `csvs`, retrains the models, and applies multiple feature reduction techniques with 2 to the maximum number of features. It saves the graphs and CSV files of the results. This is the main script of this section.

- `statistics_best_features.ipynb`: Uses the CSV files from `feature_selection_graphs_save_csv_results.ipynb` to display various statistics.

- `best_models_ensemble_learning.ipynb`: Uses the CSV files from `feature_selection_graphs_save_csv_results.ipynb` and performs ensemble learning on the best 3 models from each feature reduction technique.

## Unsupported and Discontinued Code

- `feature_selection_graphs_print.ipynb`: Similar to `feature_selection_graphs_save_csv_results.ipynb` but prints the results. This script is discontinued and unsupported.

- `feature_selection_static.ipynb` and `feature_selection_graphs.ipynb`: Early iterations of `feature_selection_graphs_save_csv_results.ipynb`. Both are discontinued and unsupported.
