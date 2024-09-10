# Training

This directory contains all the necessary components for training classification models on all available features. 

## Subfolders

### Training_Code
This subfolder includes Python scripts for training models on each ecosystem function and a comprehensive Jupyter notebook for all training tasks.

- **Python Scripts**: Separate scripts for each ecosystem function.
- **Jupyter Notebook**: A notebook (`all.ipynb`) that contains all training processes.

### All_data
This subfolder contains nine CSV files representing different data filling methods. These were generated using the `Data_ML` directory at the beginning of the repository. For more information, refer to the `Data_ML` directory.

### Results
This subfolder stores the outputs of the training processes, including saved models, CSV files containing the best models, and various codes for data visualization. There is a separate README within this folder detailing the visualization codes.

## Usage

### Training_Code
1. **Python Scripts**: Navigate to the `Training_Code` subfolder and run the desired script for training a model on a specific ecosystem function.
2. **Jupyter Notebook**: Open `all.ipynb` to execute all training processes in a unified environment.

### All_data
Use the CSV files in this folder as input data for training models. Refer to the `Data_ML` directory for details on how these CSVs were generated.

### Results
Check the `Results` subfolder for:
- Saved models after training.
- CSV files containing the best performing models.
- Visualization codes for analyzing the training outcomes.

For detailed instructions on the visualization codes, refer to the README in the `Results` folder.

## Future Enhancements
- Adding more scripts for different ecosystem functions.
- Updating data filling methods and their respective CSV files.
- Improving visualization codes for better analysis.

For any issues or contributions, please open an issue or submit a pull request on GitHub.
