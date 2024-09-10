import pandas as pd
import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA  # Import PCA
import warnings
import joblib
import os

# Define the data columns and results columns
data_columns = ['Provincial_Class','Federal_Class','Regime','Vegetation_Type','Vegetation_Cover','Woody_Canopy_Cover','Moss_Cover','Phragmites','Soil_Type','Surface_Water_Present','Saturation_Depth','Living_Moss_Depth','Organic_Depth','Hydrogeomorphic_Class',
    'OF2', 'OF3', 'OF4', 'OF5', 'OF6', 'OF7', 'OF8', 'OF9', 'OF10', 'OF11', 'OF13', 'OF14', 'OF15', 'OF16', 'OF17',
    'OF18', 'OF19', 'OF20', 'OF21', 'OF22', 'OF23', 'OF24', 'OF25', 'OF26', 'OF27', 'OF28',  'OF31',
    'OF33', 'OF34', 'OF37', 'OF38', 'F1', 'F2', 'F3_a', 'F3_b', 'F3_c', 'F3_d', 'F3_e', 'F3_f', 'F3_g', 'F4', 'F5', 'F6',
    'F7', 'F8', 'F9', 'F10',  'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 'F21', 'F22', 'F23',
    'F24', 'F25',  'F28', 'F29', 'F30', 'F31', 'F32', 'F33', 'F34', 'F35', 'F36', 'F37', 'F38', 'F39', 'F40',
    'F41',  'F43', 'F44', 'F45', 'F46', 'F47', 'F48', 'F49', 'F50', 'F51', 'F52', 'F53', 'F54', 'F55', 'F56', 'F57',
    'F58', 'F59', 'F62', 'F63', 'F64', 'F65', 'F67', 'F68', 'S1', 'S2', 'S4', 'S5'
]

results_columns = ['SFST']
model_directory = "SFST"

# Define the parameter grid for GridSearchCV, including class_weight for relevant classifiers
param_grid = {
    'RidgeClassifier': {
        'ridgeclassifier__alpha': [0.1, 0.5, 1.0],
        'ridgeclassifier__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
        'ridgeclassifier__class_weight': [None, 'balanced']
    },
    'DecisionTreeClassifier': {
        'decisiontreeclassifier__criterion': ['gini', 'entropy', 'log_loss'],
        'decisiontreeclassifier__splitter': ['best', 'random'],
        'decisiontreeclassifier__min_samples_split': [2, 3, 4, 5],
        'decisiontreeclassifier__max_features': [None, 'sqrt', 'log2'],
        'decisiontreeclassifier__class_weight': [None, 'balanced']
    },
    'RandomForestClassifier': {
        'randomforestclassifier__n_estimators': [50, 100, 200],
        'randomforestclassifier__criterion': ['gini', 'entropy', 'log_loss'],
        'randomforestclassifier__min_samples_split': [2, 5],
        'randomforestclassifier__max_features': ['sqrt', 'log2'],
        'randomforestclassifier__class_weight': [None, 'balanced']
    },
    'GradientBoostingClassifier': {
        'gradientboostingclassifier__loss': ['log_loss', 'deviance', 'exponential'],
        'gradientboostingclassifier__learning_rate': [0.001, 0.01, 0.1],
        'gradientboostingclassifier__n_estimators': [50, 100, 200],
        'gradientboostingclassifier__warm_start': [True, False],
    },
    'AdaBoostClassifier': {
        'adaboostclassifier__n_estimators': [50, 100, 200],
        'adaboostclassifier__learning_rate': [0.001, 0.01, 0.1, 1.0],
        'adaboostclassifier__algorithm': ['SAMME', 'SAMME.R']
    },
    'KNeighborsClassifier': {
        'kneighborsclassifier__n_neighbors': [5, 10, 15, 20],
        'kneighborsclassifier__weights': ['uniform', 'distance'],
        'kneighborsclassifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'kneighborsclassifier__leaf_size': [30, 50, 70],
        'kneighborsclassifier__metric': ['euclidean', 'manhattan', 'minkowski']
    },
    'MLPClassifier': {
        'mlpclassifier__hidden_layer_sizes': [(50, 50, 50), (100, 100, 100), (100, 100, 100, 100)],
        'mlpclassifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
        'mlpclassifier__solver': ['lbfgs', 'sgd', 'adam'],
        'mlpclassifier__learning_rate': ['constant', 'invscaling', 'adaptive'],
    },
    'LogisticRegression': {
        'logisticregression__penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'logisticregression__C': [0.1, 0.5, 1.0, 5.0, 10.0],
        'logisticregression__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'logisticregression__max_iter': [100, 200, 300],
        'logisticregression__class_weight': [None, 'balanced']
    },
    'SGDClassifier': {
        'sgdclassifier__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
        'sgdclassifier__penalty': ['l2', 'l1', 'elasticnet'],
        'sgdclassifier__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'sgdclassifier__warm_start': [True, False],
        'sgdclassifier__class_weight': [None, 'balanced']
    },
    'SVC': {
        'svc__C': [0.1, 1.0, 10.0],
        'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'svc__degree': [1, 3, 5],
        'svc__gamma': ['scale', 'auto'],
        'svc__class_weight': [None, 'balanced']
    },
    'GaussianNB': {
        'gaussiannb__var_smoothing': [1e-9, 1e-8, 1e-7]
    },
    'LinearDiscriminantAnalysis': {
        'lineardiscriminantanalysis__solver': ['svd', 'lsqr', 'eigen'],
        'lineardiscriminantanalysis__shrinkage': [None, 'auto', 0.1, 0.5, 1.0]
    }
}

models = [
    RidgeClassifier(), DecisionTreeClassifier(), GradientBoostingClassifier(), RandomForestClassifier(), AdaBoostClassifier(),
    KNeighborsClassifier(), MLPClassifier(max_iter=1000), LogisticRegression(max_iter=1000), SGDClassifier(max_iter=1000),
    SVC(), GaussianNB(), LinearDiscriminantAnalysis()
]
warnings.filterwarnings("ignore")

# Directory where you want to save your models

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
        'accuracy': 0
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
                tf.keras.layers.Dense(3, activation='softmax')
            ])

            # Compile the TensorFlow model
            model_tf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Standardize the data for TensorFlow model
            scaler_tf = StandardScaler()
            X_train_scaled_tf = scaler_tf.fit_transform(X_train)
            X_test_scaled_tf = scaler_tf.transform(X_test)

            # Train the TensorFlow model
            model_tf.fit(X_train_scaled_tf, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

            # Evaluate the TensorFlow model
            y_pred_tf = model_tf.predict(X_test_scaled_tf)
            y_pred_tf_classes = tf.argmax(y_pred_tf, axis=1).numpy()
            accuracy_tf = accuracy_score(y_test, y_pred_tf_classes)
            print(f"TensorFlow Accuracy: {accuracy_tf}")

            if accuracy_tf > best_model_info['accuracy']:
                best_model_info.update({
                    'model_name': 'TensorFlow',
                    'hyperparameters': None,
                    'accuracy': accuracy_tf
                })

            # Save the TensorFlow model
            model_filename = os.path.join(model_directory, f"{os.path.basename(file_path)}_TensorFlow_model.h5")
            model_tf.save(model_filename)

            # Save the predictions and actual values
            results.append(pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred_tf_classes, 'Model': 'TensorFlow'}))
        else:
            model_name = model.__class__.__name__
            pipeline = make_pipeline(StandardScaler(), model)
            # Perform grid search for hyperparameters
            if model_name in param_grid:
                grid_search = GridSearchCV(pipeline, param_grid[model_name], cv=5, scoring='accuracy')
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
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{model_name} Accuracy: {accuracy}")

            if accuracy > best_model_info['accuracy']:
                best_model_info.update({
                    'model_name': model_name,
                    'hyperparameters': best_params,
                    'accuracy': accuracy
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
csv_files = glob.glob('../All_data/*.csv')

# Initialize a list to store the best model information for each CSV file
best_models_info = []

# Process each CSV file
for csv_file in csv_files:
    best_model_info = process_csv(csv_file)
    best_models_info.append(best_model_info)

# Save the best model information for each CSV file to a CSV file
best_models_df = pd.DataFrame(best_models_info)
best_models_df.to_csv("best_models_info"+results_columns[0]+".csv", index=False)

print("Best models information saved to best_models_info.csv")
