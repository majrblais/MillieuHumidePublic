# All_features

This directory contains scripts and resources for various regression tasks, organized into several subfolders. Each subfolder is dedicated to a specific aspect of regression model training, feature selection, and ensemble learning.

## Contents

### Active Folders

1. **Training**
   - **Purpose**: Contains all necessary components for training regression models using all available features for either with normalization or without. It includes scripts for training models, organizing data, and storing results.
   - **Details**:  
	To switch between normalized and non-normalized data, modify line 243 in the script:

	- For normalized data:
	```python
	csv_files = glob.glob('../../../Data_ML/4_out_csvs_regression_norm/*.csv')
	```

	- For non-normalized data:
	```python
	csv_files = glob.glob('../../../Data_ML/4_out_csvs_regression/*.csv')
	```
	
2. **TrainingResults**
   - **Purpose**: Contains the `non_norm` and `norm` results from `Training` in two seperate folders.
   - **Details**: Each folder contains a csv for each function (e.g PR) which shows the best algorithm for each data filling method. Each folder also contains a folder for each function that contains the saved models.
	
	
3. **EnsembleLearning**
   - **Purpose**: Performs ensemble learning on the models trained on all features from the `norm` and `non_norm` folders. It includes methods for loading, retraining models, and classifying results.
   - **Details**: Contains a single `ipynb` file that loads the models and creates 10 figures (one for function and benefit). More information can be found [here](./EnsembleLearning/readme.md)

4. **ClassGrouping**
   - **Purpose**: Performs class grouping on the results for both `norm` and `non_norm`
   - **Details**: More information can be found within the `ipynb` file
   
5. **FeatureReduction**
   - **Purpose**: Applies feature reduction techniques to the regression models. The goal is to identify the most important features while maintaining or improving model performance.
   - **Details**: Scripts for feature reduction, statistical analysis of features, ensemble learning, and grouping results, more information can be found [here](./FeatureReduction/readme.md))


### Discontinued Folder

1. **Dimension_Reduction (Discontinued)**
   - **Note**: This folder is uncommented, uncleaned, and unsupported. It is included for reference but is not actively maintained or recommended for use.

## Readmes
Each folder contains its own README file with detailed information and usage guidelines. Please refer to the respective READMEs for specific instructions on running the code and reproducing results.

## Usage
1. Train the algorithms using the scripts and python files in `Training`
2. Seperate the training results into `norm` and `non_norm` similar to our `TrainingResults` folder.
3. Use the code in `EnsembleLearning` and `ClassGrouping` to get the results using all features.
4. Use the code in `FeatureSelection` to perform various feature reduction techniques, ensemble learning and class grouping using the reduced features.

## Future Work
- Expanding and refining regression methodologies.
- Enhancing ensemble learning and feature reduction techniques.
- Improving documentation and usability of the existing codes.

For any issues or contributions, please open an issue or submit a pull request on GitHub.
