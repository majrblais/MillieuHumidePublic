# Dimension Reduction (Discontinued and unsupported)
This subfolder is related to using various dimension reduction algorithms to achieve similar results as the ML upfolder with fewer features.

# Setup
To use the code in this folder, we need multiple results from `ML`. Currently only WS is available.
The folder `model_all_data` from `ML` must be included in this folder to read the data.
Folder `WS` is the same as the one found in `ML`, moved for clarity but will be moved ion the future.
The file `best_modelsWS_info.csv` must also be included which consists of the best prediction algorithm for eahc filling method (and its hyperparameter).
The codes will load the models with the hyperparameters and train each reduction algorithm using that model.
# Code

`best_model_DimRed_RMSE.ipynb` calculates the test RMSE for each model in `best_modelsWS_info.csv` file using various reduction techniques.

`all_csv_DimRed_2csv.ipynb` takes the models from `best_modelsWS_info.csv` similar to `best_model_DimRed_RMSE.ipynb`. However, the reduction algorithms dont use a static number of components (10) as the previous code. This code uses components ranging from 2 to 25 and saves them in a csv.

`best_csv_DimRed_2csv.ipynb` does the same as `all_csv_DimRed_2csv.ipynb` but only the best model from `best_modelsWS_info.csv` based on the rmse. 


`all_csv_DimRed_2graph.ipynb` does the same as `all_csv_DimRed_2csv.ipynb` but saves it as a graph instead.

`all_csv_DimRed_2graph_trasnparent.ipynb` does the same as `all_csv_DimRed_2csv.ipynb` but saves it as a graph instead with any curve with a high (10) value transparent and the y limit from 0.5 to 4.


`best_csv_DimRed_2graph.ipynb` does the same as `best_csv_DimRed_2csv.ipynb` but saves graphs.

`DimRed_BestModels.ipynb` reads from `best_modelsWS_info.csv` and saves the results (to `dimensionality_reduction_rmses_WS.csv`) from each DimRed approaches and info about the model.

`all_csv_DimRed_ShowAll.ipynb` reads from `dimensionality_reduction_rmses_WS.csv` and retrains the model and show the RMSE (used temporarily)