# Project Repository

Welcome to the project repository. This repository contains various codes and resources for classification, regression, data preprocessing, and analyses conducted during our project.
The repository is organized into several main folders. Most of the code base and results were achieved after the month of may-june due to errors in the WESP-AC being corrected around these dates.
Earlier results are not included since they include errors in the files.


For more information about this code and the results, contact **emb9357@umoncton.ca**.

## Usage
Navigate to the respective folders for detailed instructions on running the code and reproducing the results. Each folder contains its own README file with specific information and usage guidelines.
In general, use this repo as such:

1. Generate the data using the folder `Data_ML`
2. Using the generated data, train the `Classification` or `Regression` algorithms in the respective folders.
3. Using the data, the `Tetrad` or `CausalML` models can also be trained.

## Contents
Here are the specific folders and their contents.
### Data_ML
This folder is dedicated to data preprocessing and manipulation tasks. It includes scripts for converting and adapting raw WESP-AC data, compiling data for classification and regression tasks, and filling missing data using various methods. The folder also contains subfolders for storing intermediate and final processed data.

For more detailed information, refer to the [Data_ML README](./Data_ML/readme.md).


### Classification
This folder contains code and resources for classification tasks. The classification algorithms are organized based on the features they utilize.

- **All_features**: Contains code for training classification models using all available features, as well as methods for feature selection and ensemble learning.
- **Specific_features**: Designated for classification algorithms trained on specific features, focusing on particular ecosystem functions derived from the WESP-AC. Also includes feature selection and ensemble learning.
- **Xtra_features (Removed)**: Contains code for training classification models using newly added features not found in the WESP-AC.
- **Specific_Xtra_features (Removed)**: Combines the newly added features with specific features from the WESP-AC for classification tasks.

For more detailed information, refer to the [Classification README](./Classification/readme.md).

### Regression
This folder includes codes and resources for regression tasks. The regression models and analyses are organized based on different methodologies and feature sets used.

- **All_features**: Contains code for training regression models using all available features. It includes methods for training models, feature reduction, class grouping and ensemble learning.
- **Specific_features**: This folder is designated for regression models trained on specific features. Contains the code to train, feature reduction, class grouping and ensemble learning.
- **Xtra_features (Removed)**: Contains code for training regression models using newly added features not found in the WESP-AC.
- **Specific_Xtra_features (Removed)**: Combines the newly added features with specific features from the WESP-AC for regression tasks.

For more detailed information, refer to the [Regression README](./Regression/readme.md).

### CausalML
This folder is used to generate information about the importance of specific features for each ecosystemic function.
CausalML was briefly explored but was only used to confirm our classification and regression results.


### Tetrad
This folder focuses on tetrad causal inference. It contains scripts and resources for conducting causal inference analyses using the tetrad approach.

- **Data_Tetrad**: This folder contains data specific to the tetrad causal inference analyses. It includes raw data, intermediate data, and processed data files necessary for the analyses.

### Report
This folder is used to store the reports generated from various analyses and projects within the repository.
It is compiled with 'pdflatex' on linux.



### FeatureImpact (Discontinued)
This folder contains code related to the feature impact analysis which was used for classification and regression.
It was used to analyse the effect of modifying the feature values to better understand their impact.


## Future Work
- Enhancing documentation and usability of the existing codes.
- Developing more extensive causal inference analyses in the `Tetrad` folder.


For any issues or contributions, please open an issue or submit a pull request on GitHub.
