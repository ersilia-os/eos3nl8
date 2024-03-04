# Ersilia Model In Progress

This model is work in progress. Please edit the [metadata.json](metadata.json) file to complete the information about the model. This README file will be updated automatically based on the information contained in that folder.

The `COVID-19_drug_repurposing.ipynb` notebook is the original notebook from the authors. They collected a list of disease of Corona-Virus(COV) in DRKG. They used FDA-approved drugs in Drugbank as candidate drugs (we exclude drugs with molecule weight < 250). The drug list is in `infer_drug.tsv.` They focused on two treatment relations, `Hetionet::CtD::Compound:Disease` and `GNBR::T::Compound:Disease`. They extracted the embeddings of the diseases and drugs from the knowledge graph. Finally, they used the metric `edge score` to predict the drugs that can be repurposed for COVID-19.

In the `smiles_embeddings_prediction_morgan_fps.ipynb` notebook we predict the SMILES embeddings of drug molecules. The goal is to predict the 400 DRKG embeddings of a drug molecule. We used the morgan fingerprint to preprocess the SMILES data. Keras tuner was used to fine-tune and train the neural network algorithm.

The `smiles_embeddings_prediction_morgan_count.ipynb` notebook is similar to the `smiles_embeddings_prediction_morgan_fps.ipynb` notebook. The difference is that the `smiles_embeddings_prediction_morgan_count.ipynb` notebook uses the morgan fingerprint count to preprocess the SMILES data.

The `smiles_embeddings_prediction_eosce` notebook is similar to the `smiles_embeddings_prediction_morgan_fps.ipynb` notebook. The difference is that it uses the Ersilia descriptor to preprocess the SMILES data.

The `eosce_evaluation_keras_tuner.ipynb` shows the full training and evaluation of the model. The first stage involves predicting the DRKG embeddings of drug molecules. The second stage involves predicting the edge score of the drug molecules. The model uses the ersilia descriptor to preprocess SMILES input. The keras-tuner is used to fine-tune and train the model. The notebook also evaluates the predicted embeddings visually. We compared the embeddings row-wise, that is, comparing 400 original embeddings with the predicted embedding for a single drug. We also compared the embeddings column-wise, that is, comparing an embedding vector for all the drugs in the test set. These visualizations show us how good the model’s prediction is.
The `eosce_evaluation_xgboost.ipynb` uses the XGBoost algorithm instead. Optuna is used to fine-tune the XGBoost model. The XGBoost model performs better than the Keras model.

The `morgan_fps_evaluation_keras_tuner.ipynb` notebook uses morgan fingerprint to preprocess the SMILES input, and the model is trained using keras tuner. The `morgan_fps_evaluation_xgboost.ipynb` uses the XGBoost model.

The `morgan_counts_evaluation_keras_tuner.ipynb` notebook uses morgan fingerprint count to preprocess the SMILES input, and the model is trained using keras tuner. The `morgan_counts_evaluation_xgboost.ipynb` uses the XGBoost model.

The `extract_all_drug_embeddings.ipynb` notebook contains code to extract the embeddings of all the drugs in the `drugbank_smiles.txt` from the DRKG. This was used to get a larger dataset for training. The `extract_infer_drug_embeddings.ipynb` contains code to extract the embeddings of all the drugs in the `infer_drug.tsv` file. There are 8807 compounds in the `drugbank_smiles.txt` file. The `infer_drug.tsv` only has drugs with a molecular weight >= 250. There are 8104 drugs in the `infer_drug.tsv` file. Not all the drugs in the `infer_drug.tsv` file are in the `drugbank_smiles.txt` file.

The `descriptors.py`, `xgboost_regressor.py` and `xgboost_trainer.py` are scripts to train and save three XGBoost models using the morgan fingerprint, morgan fingerprint count, and ersilia descriptor. The  `descriptors.py` contains the *MorganFingerprinter()* class which has functions to calculate and preprocess SMILES using morgan fingerprint and morgan fingerprint count.
It also has the class *ErsiliaCompoundEmbedder()* which has functions to calculate and preprocess SMILES using the ersilia descriptor. The `xgboost_regressor.py` script has functions to optimize the hyperparameters and train an XGBoost model. The `xgboost_trainer.py` is the main script for loading the training and test data, preprocessing them, training the models and saving them. The `nohup.out` file contains the log of the training process.


The `eosce_knn_1_neighbor.ipynb` and `eosce_knn_3_neighbor.ipynb` notebooks use the K Nearest Neigbor algorithm to find the embeddings in the training set close to the input ersilia embedding. In the `eosce_knn_1_neighbor.ipynb` notebook, we find the closest embedding in the training data, and use the y^ as the output.  In the `eosce_knn_3_neighbor.ipynb` notebook, we find the three closest embedding in the training data, and use take the average of their y^ as the output. 
