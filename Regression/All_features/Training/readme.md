# Training

This directory contains all the necessary components for training regression models using all available features. It includes scripts for training models, organizing data, and storing results.

## Contents

### Subfolders

#### Training_Code
This subfolder includes Python scripts and Jupyter notebooks for training regression models on individual or all ecosystem functions.

#### All_data
This subfolder contains the classification data, copied from the `Data_ML` folder in the main directory for easier accessibility.

#### Results
This subfolder stores the outputs of the training processes, including saved models and validation predictions for each data file, organized in subfolders by ecosystem function.

### Scripts in Results

1. **save_csv_graph_by_fill_method.ipynb**
   - **Purpose**: Generates graphs for each ecosystem function, separated by the fill method used.
   - **Outputs**: Graphs showing validation results for each algorithm, organized by the fill method.

2. **save_csv_graph_by_model.ipynb**
   - **Purpose**: Generates graphs for each ecosystem function, separated by the algorithm used.
   - **Outputs**: Graphs showing validation results for each fill method, organized by algorithm.

## Usage

### Training Models
1. Navigate to the `Training_Code` subfolder and run the desired script or Jupyter notebook to train the regression models.
2. Use the data files in the `All_data` subfolder as input for training.

### Viewing Results
1. Check the `Results` subfolder for saved models and validation predictions.
2. Use `save_csv_graph_by_fill_method.ipynb` to generate graphs organized by fill method.
3. Use `save_csv_graph_by_model.ipynb` to generate graphs organized by algorithm.

## Future Enhancements
- Implementing more advanced regression techniques.
- Enhancing visualization methods for better analysis.
- Extending the analysis to include more sophisticated evaluation metrics.

For any issues or contributions, please open an issue or submit a pull request on GitHub.
