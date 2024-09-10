# All_features

This directory contains scripts and resources for various classification tasks, organized into several subfolders. Each subfolder is dedicated to a specific aspect of classification model training, feature selection, and ensemble learning.

## Contents

### Active Folders

1. **Training**
   - **Purpose**: Contains all necessary components for training classification models using all available features for either with normalization or without. It includes scripts for training models, organizing data, and storing results.
   - **Details**: Compared to regression, classification only uses normalized data
	
2. **TrainingResults**
   - **Purpose**: Contains the `non_norm` and `norm` results from `Training` in two seperate folders.
   - **Details**: Each folder contains a csv for each function (e.g PR) which shows the best algorithm for each data filling method. Each folder also contains a folder for each function that contains the saved models.
	
	
3. **EnsembleLearning**
   - **Purpose**: Performs ensemble learning on the models trained on all features from the `norm` and `non_norm` folders. It includes methods for loading, retraining models, and classifying results.
   - **Details**: Contains a single `ipynb` file that loads the models and creates 10 figures (one for function and benefit). More information can be found [here](./EnsembleLearning/readme.md)


5. **FeatureReduction**
   - **Purpose**: Applies feature reduction techniques to the classification models. The goal is to identify the most important features while maintaining or improving model performance.
   - **Details**: Scripts for feature reduction, statistical analysis of features, ensemble learning, more information can be found [here](./FeatureReduction/readme.md))



## Readmes
Each folder contains its own README file with detailed information and usage guidelines. Please refer to the respective READMEs for specific instructions on running the code and reproducing results.

## Usage
1. Train the algorithms using the scripts and python files in `Training`
2. Seperate the training results into `norm` and `non_norm` similar to our `TrainingResults` folder.
3. Use the code in `EnsembleLearning`  to get the results using all features.
4. Use the code in `FeatureSelection` to perform various feature reduction techniques, ensemble learning and class grouping using the reduced features.

## Future Work
- Expanding and refining classification methodologies.
- Enhancing ensemble learning and feature reduction techniques.
- Improving documentation and usability of the existing codes.

For any issues or contributions, please open an issue or submit a pull request on GitHub.
