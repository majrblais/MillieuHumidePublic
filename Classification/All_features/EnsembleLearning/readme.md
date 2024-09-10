# EnsembleLearning

This directory contains resources and scripts for performing ensemble learning using the best models identified for each ecosystem function.

## Contents

### Script

1. **best_models_ensemble_learning.ipynb**
    - **Purpose**: Performs ensemble learning on the best models for each ecosystem function.
    - **Inputs**: 
        - `best_model_infoXXX.csv` files for each ecosystem function.
    - **Outputs**: 
        - Confusion matrices for the top 2 individual models.
        - Confusion matrices for ensemble models using the top 5 models for each function.
    - **Description**: This script reads the `best_model_infoXXX.csv` files for each function, similar to the feature selection process. It performs ensemble learning using the top 2 models and the top 5 models without applying any feature reduction techniques.

### Folder

- **output**
    - **Purpose**: Stores the output results from the ensemble learning script.
    - **Contents**: Confusion matrices and other output data from the ensemble learning processes.

- **ensemble_plots**
    - **Purpose**: Stores the best results from the ensemble learning script for each function, used in report.
    - **Contents**: Confusion matrices and other output data from the ensemble learning processes.
## Usage

### best_models_ensemble_learning.ipynb
1. Ensure the `best_model_infoXXX.csv` files for each ecosystem function are available.
2. Run the script to perform ensemble learning using the top 2 individual models and the top 5 models for each function.
3. The script generates confusion matrices to compare the performance of individual models and ensemble models.
4. The results are saved in the `output` folder for further analysis.

## Future Enhancements
- Exploring additional ensemble learning techniques.
- Improving the visualization of ensemble model performance.
- Extending the ensemble methods to include more diverse combinations of models.

For any issues or contributions, please open an issue or submit a pull request on GitHub.
