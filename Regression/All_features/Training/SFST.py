import pandas as pd
import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge, LinearRegression, RANSACRegressor, TheilSenRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA  # Import PCA
import warnings
import joblib
import os

# Define the data columns and results columns
data_columns = [ 'Provincial_Class','Federal_Class','Regime','Vegetation_Type','Vegetation_Cover','Woody_Canopy_Cover','Moss_Cover','Phragmites','Soil_Type',
'Surface_Water_Present','Saturation_Depth','Living_Moss_Depth','Organic_Depth','Hydrogeomorphic_Class',
    'OF2', 'OF3', 'OF4', 'OF5', 'OF6', 'OF7', 'OF8', 'OF9', 'OF10', 'OF11', 'OF13', 'OF14', 'OF15', 'OF16', 'OF17',
    'OF18', 'OF19', 'OF20', 'OF21', 'OF22', 'OF23', 'OF24', 'OF25', 'OF26', 'OF27', 'OF28',  'OF30', 'OF31',
    'OF33', 'OF34', 'OF37', 'OF38', 'F1', 'F2', 'F3_a', 'F3_b', 'F3_c', 'F3_d', 'F3_e', 'F3_f', 'F3_g', 'F4', 'F5', 'F6',
    'F7', 'F8', 'F9', 'F10', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 'F21', 'F22', 'F23',
    'F24', 'F25',  'F28', 'F29', 'F30', 'F31', 'F32', 'F33', 'F34', 'F35', 'F36', 'F37', 'F38', 'F39', 'F40',
    'F41',  'F43', 'F44', 'F45', 'F46', 'F47', 'F48', 'F49', 'F50', 'F51', 'F52', 'F53', 'F54', 'F55', 'F56', 'F57',
    'F58', 'F59', 'F62', 'F63', 'F64', 'F65', 'F67', 'F68', 'S1', 'S2', 'S4', 'S5'
]

# Directory where you want to save your models
model_directory = "SFST"
results_columns = [model_directory]

# Define the parameter grid for GridSearchCV
param_grid = {
    'Ridge': {
        'ridge__alpha': [0.1, 0.5, 1.0],
        'ridge__solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
    },
    'DecisionTreeRegressor': {
        'decisiontreeregressor__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'decisiontreeregressor__splitter': ['best', 'random'],
        'decisiontreeregressor__min_samples_split': [1, 2, 3, 4, 5],
        'decisiontreeregressor__max_features': [0, 1, 2, 3, 'sqrt', 'log2']
    },
    'RandomForestRegressor': {
        'randomforestregressor__n_estimators': [1, 50, 100],
        'randomforestregressor__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'randomforestregressor__min_samples_split': [2, 5],
        'randomforestregressor__max_features': [1, 3, 'sqrt', 'log2'],
    },
    'GradientBoostingRegressor': {
        'gradientboostingregressor__loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
        'gradientboostingregressor__learning_rate': [0.001, 0.01],
        'gradientboostingregressor__n_estimators': [25, 50, 100],
        'gradientboostingregressor__warm_start': [True, False],
    },
    'AdaBoostRegressor': {
        'adaboostregressor__n_estimators': [1, 20, 50, 100],
        'adaboostregressor__learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0, 10],
        'adaboostregressor__loss': ['linear', 'square', 'exponential']
    },
    'KNeighborsRegressor': {
        'kneighborsregressor__n_neighbors': [2, 5, 10, 25],
        'kneighborsregressor__weights': ['uniform', 'distance'],
        'kneighborsregressor__algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'kneighborsregressor__leaf_size': [5, 30, 50],
        'kneighborsregressor__metric': ['cityblock', 'cosine', 'euclidean', 'haversine', 'l1', 'l2', 'manhattan', 'nan_euclidean']
    },
    'MLPRegressor': {
        'mlpregressor__hidden_layer_sizes': [(50, 50, 50), (100, 100, 100), (100, 100, 100, 100)],
        'mlpregressor__activation': ['identity', 'logistic', 'tanh', 'relu'],
        'mlpregressor__solver': ['lbfgs', 'sgd', 'adam'],
        'mlpregressor__learning_rate': ['constant', 'invscaling', 'adaptive'],
    },
    'ElasticNet': {
        'elasticnet__l1_ratio': [0.25, 0.5, 0.75],
        'elasticnet__fit_intercept': [True, False],
        'elasticnet__precompute': [True, False],
        'elasticnet__copy_X': [True, False],
        'elasticnet__warm_start': [True, False],
        'elasticnet__positive': [True, False],
        'elasticnet__selection': ['cyclic', 'random']
    },
    'SGDRegressor': {
        'sgdregressor__loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
        'sgdregressor__penalty': ['l2', 'l1', 'elasticnet', None],
        'sgdregressor__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'sgdregressor__warm_start': [True, False],
    },
    'SVR': {
        'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        'svr__degree': [1, 3, 5],
        'svr__gamma': ['scale', 'auto', 1.0, 5.0],
        'svr__shrinking': [True, False]
    },
    'BayesianRidge': {
        'bayesianridge__alpha_1': [1e-7, 1e-6, 1e-5],
        'bayesianridge__alpha_2': [1e-7, 1e-6, 1e-5],
        'bayesianridge__lambda_1': [1e-7, 1e-6, 1e-5],
        'bayesianridge__lambda_2': [1e-7, 1e-6, 1e-5],
    },
    'KernelRidge': {
        'kernelridge__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
        'kernelridge__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        'kernelridge__degree': [1, 2, 3, 5, 10],
        'kernelridge__coef0': [0.0, 0.5, 1.0]
    },
    'LinearRegression': {
        'linearregression__fit_intercept': [True, False],
        'linearregression__copy_X': [True, False],
        'linearregression__positive': [True, False]
    },
    'RANSACRegressor': {
        'ransacregressor__min_samples': [None, 1, 2, 5, 10, 50],
        'ransacregressor__max_trials': [1, 10, 50, 100, 150],
        'ransacregressor__loss': ['absolute_error', 'squared_error']
    },
    'TheilSenRegressor': {
        'theilsenregressor__max_subpopulation': [1, 10, 100, 1000],
        'theilsenregressor__n_subsamples': [None, 1, 5, 10, 25],
    }
}

models = [
    Ridge(), DecisionTreeRegressor(), GradientBoostingRegressor(), RandomForestRegressor(), AdaBoostRegressor(),
    KNeighborsRegressor(), MLPRegressor(max_iter=1000), ElasticNet(max_iter=1000), SGDRegressor(max_iter=1000),
    BayesianRidge(max_iter=1000), KernelRidge(), LinearRegression(), RANSACRegressor(),TheilSenRegressor()
]
warnings.filterwarnings("ignore")



# Create the directory if it doesn't exist
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Function to process each CSV file
def process_csv(file_path):
    data = pd.read_csv(file_path)
    X = data[data_columns]
    y = data[results_columns[0]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    best_model_info = {
        'csv_file': os.path.basename(file_path),
        'model_name': None,
        'hyperparameters': None,
        'rmse': float('inf')
    }

    results = []

    for model in models + ['TensorFlow']:  # Add TensorFlow model to the loop
        print(f"Processing {model} for {file_path}")
        if model == 'TensorFlow':
            # Define the TensorFlow model
            model_tf = tf.keras.models.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1)
            ])

            # Compile the TensorFlow model
            model_tf.compile(optimizer='adam', loss='mean_squared_error')

            # Standardize the data for TensorFlow model
            scaler_tf = StandardScaler()
            X_train_scaled_tf = scaler_tf.fit_transform(X_train)
            X_test_scaled_tf = scaler_tf.transform(X_test)

            # Train the TensorFlow model
            model_tf.fit(X_train_scaled_tf, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

            # Evaluate the TensorFlow model
            y_pred_tf = model_tf.predict(X_test_scaled_tf)
            rmse_tf = mean_squared_error(y_test, y_pred_tf, squared=False)
            print(f"TensorFlow RMSE: {rmse_tf}")

            if rmse_tf < best_model_info['rmse']:
                best_model_info.update({
                    'model_name': 'TensorFlow',
                    'hyperparameters': None,
                    'rmse': rmse_tf
                })

            # Save the TensorFlow model
            model_filename = os.path.join(model_directory, f"{os.path.basename(file_path)}_TensorFlow_model.h5")
            model_tf.save(model_filename)

            # Save the predictions and actual values
            results.append(pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred_tf.flatten(), 'Model': 'TensorFlow'}))
        else:
            model_name = model.__class__.__name__
            pipeline = make_pipeline(StandardScaler(), model)
            # Perform grid search for hyperparameters
            if model_name in param_grid:
                grid_search = GridSearchCV(pipeline, param_grid[model_name], cv=5, scoring='neg_mean_squared_error')
                grid_search.fit(X_train, y_train)
                best_estimator = grid_search.best_estimator_
                best_params = grid_search.best_params_
                print(f"Best hyperparameters for {model_name}: {best_params}")
            else:
                pipeline.fit(X_train, y_train)
                best_estimator = pipeline
                best_params = None

            # Make predictions
            y_pred = best_estimator.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            print(f"{model_name} RMSE: {rmse}")

            if rmse < best_model_info['rmse']:
                best_model_info.update({
                    'model_name': model_name,
                    'hyperparameters': best_params,
                    'rmse': rmse
                })

            # Save the model
            model_filename = os.path.join(model_directory, f"{os.path.basename(file_path)}_{model_name}_model.pkl")
            joblib.dump(best_estimator, model_filename)

            # Save the predictions and actual values
            results.append(pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten(), 'Model': model_name}))

    # Save the predictions and actual values to a CSV file
    results_df = pd.concat(results, axis=0)
    results_filename = f"output_{os.path.basename(file_path)}_{results_columns[0]}.csv"
    results_df.to_csv(results_filename, index=False)

    return best_model_info

# Get the list of CSV files in the directory
csv_files = glob.glob('../../../Data_ML/4_out_csvs_regression_norm/*.csv')

# Initialize a list to store the best model information for each CSV file
best_models_info = []

# Process each CSV file
for csv_file in csv_files:
    best_model_info = process_csv(csv_file)
    best_models_info.append(best_model_info)

# Save the best model information for each CSV file to a CSV file
best_models_df = pd.DataFrame(best_models_info)
best_models_df.to_csv("best_models"+results_columns[0]+"_info.csv", index=False)

print("Best models information saved to best_models_info.csv")
