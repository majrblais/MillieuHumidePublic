# Results

This directory contains the outputs and analysis scripts for the training processes. Each ecosystem function has its own subfolder containing the saved models, validation results, and optimized parameters.

## Subfolders

Each subfolder corresponds to a specific ecosystem function (10 in total) and includes the following:

### Saved Models
Each subfolder contains saved weights for the algorithms trained in the `Training_Code` folder. Each algorithm was trained on each data file in the `All_data` folder using grid search for parameter optimization.

### CSV Files
Each subfolder also contains 9 CSV files (one for each `All_data` file). These CSV files contain three columns: 
- **Actual**: The actual validation data point.
- **Predicted**: The predicted validation data point by the model.
- **Model**: The model used for the prediction.

## Best Models Information

In the current `Results` folder, there is a CSV file for each ecosystem function named `best_models_infoXXX.csv`, where `XXX` is the ecosystem function name. These CSV files contain the following columns:
- **csv_file**: The data file used from `All_data`.
- **model_name**: The name of the model.
- **hyperparameter**: The optimized hyperparameters for the model.
- **accuracy**: The accuracy achieved by the model.

## Analysis Scripts

The `Results` folder also contains four Jupyter notebooks for analyzing and visualizing the results:

### make_accuracy_table.ipynb
Creates a table for a specific ecosystem function in LaTeX format, showing the validation accuracy for each model and each data file.

### print_best_model.ipynb
Prints the best model based on accuracy for each data file for a specific ecosystem function.

### save_csv_graph_by_model.ipynb
Generates confusion matrices for each saved model for a specific ecosystem function.

### save_csv_graph_by_model_best.ipynb
Generates confusion matrices only for the best model for each ecosystem function.

## Usage

### Viewing Saved Models
Navigate to the subfolder for the specific ecosystem function you are interested in. Load the saved models from the respective files for further analysis or prediction.

### Analyzing CSV Files
Each CSV file within the subfolders provides the actual and predicted values for validation data points. Use these CSV files to evaluate model performance.

### Best Models Information
Open the `best_models_infoXXX.csv` file to view the best-performing models, their hyperparameters, and accuracy for each data file.

### Running Analysis Scripts
1. **make_accuracy_table.ipynb**: Run this notebook to generate LaTeX tables for model accuracies.
2. **print_best_model.ipynb**: Run this notebook to print the best model for each data file.
3. **save_csv_graph_by_model.ipynb**: Run this notebook to create confusion matrices for all models.
4. **save_csv_graph_by_model_best.ipynb**: Run this notebook to create confusion matrices for the best models only.

## Future Enhancements
- Including more advanced analysis and visualization techniques.
- Improving the structure and performance of the existing analysis scripts.

For any issues or contributions, please open an issue or submit a pull request on GitHub.
