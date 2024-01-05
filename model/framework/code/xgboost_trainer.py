from xgboost_regressor import XGBRegressor
from descriptors import MorganFingerprinter, ErsiliaCompoundEmbedder
import pandas as pd
import joblib


# Loading training and test data
print("Loading training and test data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Morgan Counts
print("Morgan Counts")
fps = MorganFingerprinter()
X_train, y_train = fps.preprocess_data_morgan_counts(train_df)
X_test, y_test = fps.preprocess_data_morgan_counts(test_df)

mdl = XGBRegressor(X_train, y_train, X_test, y_test)
best_xgb_regressor= mdl.train_model()

# Save the trained model to a file
print("Saving the model to a file...")
model_path= '../../checkpoints/models/xgboost_morgan_counts.bin'
joblib.dump(best_xgb_regressor, model_path)

# Morgan FPS
print("Morgan FPS")
fps = MorganFingerprinter()
X_train, y_train = fps.preprocess_data_morgan_fps(train_df)
X_test, y_test = fps.preprocess_data_morgan_fps(test_df)

mdl = XGBRegressor(X_train, y_train, X_test, y_test)
best_xgb_regressor= mdl.train_model()

# Save the trained model to a file
print("Saving the model to a file...")
model_path= '../../checkpoints/models/xgboost_morgan_fps.bin'
joblib.dump(best_xgb_regressor, model_path)

# Ersilia Embedding
print("ErsiliaEmbeddings")
fps = ErsiliaCompoundEmbedder()
X_train, y_train = fps.preprocess_data_eosce(train_df)
X_test, y_test = fps.preprocess_data_eosce(test_df)

mdl = XGBRegressor(X_train, y_train, X_test, y_test)
best_xgb_regressor= mdl.train_model()

# Save the trained model to a file
print("Saving the model to a file...")
model_path= '../../checkpoints/models/xgboost_eosce.bin'
joblib.dump(best_xgb_regressor, model_path)