'''Script to train and fine-tune the XGBoost model to predict the DRKG embeddings of compounds
using the morgan fingerprint input embeddings.'''

# import libraries
import os
import numpy as np
import pandas as pd
import joblib
import optuna
from rdkit.Chem import AllChem, MolFromSmiles
import xgboost as xgb


# Loading training and test data
print("Loading training and test data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

RADIUS = 2
N_BITS = 2048

# Function to calculate Morgan fingerprints
def calculate_morgan_fingerprint(smiles, radius=RADIUS, n_bits=N_BITS):
    '''Function to convert SMILES to fingerprint using the Morgan Fingerprint.

    Parameters
    -----------
    smiles (str): SMILES of the compound.
    radius (int): controls the radius of the fingerprint.
    n_bits (int): controls the length of the fingerprint bit vector.

    Returns
    -------
    arr (NumPy Array): fingerprint of SMILES
    '''
    # Convert the input SMILES string into an RDKit molecule object.
    mol = MolFromSmiles(smiles)
    # If the molecule conversion is successful, then generate the fingerprint
    if mol is not None:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((1,))
        AllChem.DataStructs.ConvertToNumpyArray(fingerprint, arr)
        return arr
    else:
        return None


# Function to preprocess data and create embeddings
def preprocess_data(df):
    '''Function to convert SMILES to morgan fingerprint embeddings'''
    # Get the target embeddings
    embeddings = df.iloc[:, 2:].values
    # create fingerprint column
    df['morgan_fingerprint'] = df['SMILES'].apply(calculate_morgan_fingerprint)
    df = df.dropna()
    # Extract the fingerprints as a NumPy array
    morgan_fingerprints = np.array(df['morgan_fingerprint'].tolist())
    return morgan_fingerprints, embeddings


# Pre-process the data
print("Pre-processing the training and testing data...")
X_train, y_train = preprocess_data(train_df)
X_test, y_test = preprocess_data(test_df)
print("The length of X_train is:", len(X_train))
print("The length of X_test is:", len(X_test))


def objective(trial):
    '''Function to fine-tune XGBoost model with Optuna'''
    params = {
        "objective": "reg:squarederror",
        "verbosity": 0,
        "n_estimators": trial.suggest_int("n_estimators", 200, 400),
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
    xgb_regressor.fit(X_train, y_train, verbose=True)
    # Make predictions on the validation set
    y_pred = xgb_regressor.predict(X_test)
    # Calculate mean squared error for each target variable
    mse_per_target = np.mean((y_test - y_pred)**2, axis=0)
    print("mse per target", mse_per_target)
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
model_filename = 'xgboost_mf_model.bin'
joblib.dump(best_xgb_regressor, model_filename)
print(f"File saved as {model_filename}")
print("Task completed.")
