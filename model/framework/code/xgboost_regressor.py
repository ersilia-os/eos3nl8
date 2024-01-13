import numpy as np
import xgboost as xgb
import optuna

class XGBRegressor():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def _objective(self, trial):
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
        xgb_regressor.fit(self.X_train, self.y_train, verbose=True)
        # Make predictions on the validation set
        y_pred = xgb_regressor.predict(self.X_test)
        # Calculate mean squared error for each target variable
        mse_per_target = np.mean((self.y_test - y_pred)**2, axis=0)
        print("mse per target", mse_per_target)
        # Average the mean squared errors across all target variables
        mse = np.mean(mse_per_target)
        return mse

    def _optimize_hyperpar(self):
        # Create an Optuna study and optimize the objective function
        study = optuna.create_study(direction='minimize')
        study.optimize(self._objective, n_trials=100)
        # Print the best hyperparameters
        print('Best trial:')
        trial = study.best_trial
        print(f'Params: {trial.params}')
        print(f'Mean Squared Error: {trial.value}')
        # Extract the best hyperparameters
        print("Training model with the best hyperparameters...")
        best_params = study.best_trial.params
        return best_params
    
    def train_model(self):
        # Create an XGBoost regressor with the best hyperparameters
        best_params = self._optimize_hyperpar()
        best_xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', **best_params)
        # Train the final model using the full training set
        best_xgb_regressor.fit(self.X_train, self.y_train)
        return best_xgb_regressor
