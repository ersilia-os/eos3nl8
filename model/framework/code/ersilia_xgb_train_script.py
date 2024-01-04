'''Script to train and fine-tune the XGBoost model to predict the DRKG embeddings of compounds
using the ersilia descriptor as input.'''

# import libraries
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import optuna

# # Specify the directory
directory = 'data/ersilia_embeddings/'
# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)

# Load the variables as NumPy arrays
print("Loading training and test data...")
X_train = np.loadtxt(os.path.join(directory, 'X_train_ersilia.txt'))
y_train = np.loadtxt(os.path.join(directory, 'y_train_ersilia.txt'))
X_test = np.loadtxt(os.path.join(directory, 'X_test_ersilia.txt'))
y_test = np.loadtxt(os.path.join(directory, 'y_test_ersilia.txt'))


def objective(trial):
    '''Function to fine-tune XGBoost model with Optuna'''
    params = {
        "objective": "reg:squarederror",
        "verbosity": 0,
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "n_jobs": -1,  # Set to -1 to use all available CPU cores
        "tree_method": "hist",  # Enable GPU acceleration
        "device": "cuda",  # Specify GPU as the device
    }

    # Create an XGBoost regressor with the suggested hyperparameters
    xgb_regressor = xgb.XGBRegressor(**params)
    xgb_regressor.fit(X_train, y_train)
    # Make predictions on the validation set
    y_pred = xgb_regressor.predict(X_test)
    # Calculate mean squared error for each target variable
    mse_per_target = np.mean((y_test - y_pred)**2, axis=0)
    # print("mse per target", mse_per_target)
    # Average the mean squared errors across all target variables
    mse = np.mean(mse_per_target)
    return mse


# Create an Optuna study and optimize the objective function
print("Fine-tuning XGBoost model...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
# Print the best hyperparameters
print('Best trial:')
trial = study.best_trial
print(f'Params: {trial.params}')
print(f'Mean Squared Error: {trial.value}')


# Extract the best hyperparameters
print("Training model with the best hyperparameters...")
best_params = study.best_trial.params
# Create an XGBoost regressor with the best hyperparameters
best_xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', **best_params)
# Train the final model using the full training set
best_xgb_regressor.fit(X_train, y_train)
# Make predictions on your test set or perform any further evaluation
y_test_pred = best_xgb_regressor.predict(X_test)

# Save the trained model to a file
print("Saving the model to a file...")
model_filename = 'xgboost_ersilia_model.bin'
joblib.dump(best_xgb_regressor, model_filename)
print(f"File saved as {model_filename}")
print("Task completed.")
