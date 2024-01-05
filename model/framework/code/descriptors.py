import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

RADIUS = 2
N_BITS = 2048

# Function to calculate Morgan fingerprints
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors as rd
from eosce.models import ErsiliaCompoundEmbeddings

class MorganFingerprinter():
    def __init__(self):
        self.radius = RADIUS
        self.n_bits= N_BITS

    def _calculate_morgan_fingerprints(self, smiles_list):
        fingerprints = []
        none_indices = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
                arr = np.zeros((1,))
                AllChem.DataStructs.ConvertToNumpyArray(fingerprint, arr)
                fingerprints.append(arr)
            else:
                fingerprints.append(None)
                none_indices.append(i)
        return fingerprints, none_indices

    def _calculate_morgan_fingerprint_count(self, smiles_list):
        fingerprints = []
        none_indices = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fingerprint = rd.GetHashedMorganFingerprint(mol, self.radius, nBits=self.n_bits)
                arr = np.zeros((self.n_bits,), dtype=np.uint8)
                for idx, count in fingerprint.GetNonzeroElements().items():
                    arr[idx] = count if count < 255 else 255
                fingerprints.append(np.array(arr, dtype=np.uint8))
            else:
                fingerprints.append(None)
                none_indices.append(i)
        return fingerprints, none_indices

    def preprocess_data_morgan_fps(self, df):
        embeddings = df.iloc[:, 2:].values
        morgan_fingerprints, none_indices = self._calculate_morgan_fingerprints(df['SMILES'])
        morgan_fingerprints = np.delete(morgan_fingerprints, none_indices, axis=0)
        embeddings = np.delete(embeddings, none_indices, axis=0)
        return morgan_fingerprints, embeddings
    
    def preprocess_data_morgan_counts(self, df):
        embeddings = df.iloc[:, 2:].values
        morgan_fingerprints, none_indices = self._calculate_morgan_fingerprint_count(df['SMILES'])
        morgan_fingerprints = np.delete(morgan_fingerprints, none_indices, axis=0)
        embeddings = np.delete(embeddings, none_indices, axis=0)
        return morgan_fingerprints, embeddings


class ErsiliaCompoundEmbedder():
    def __init__(self):
        pass

    def _calculate_ersilia_embeddings(self,smiles_list):
        mdl = ErsiliaCompoundEmbeddings()
        X = mdl.transform(smiles_list)
        return X

    def preprocess_data_eosce(self, df):
        embeddings = df.iloc[:, 2:].values
        eosce = self._calculate_ersilia_embeddings(df['SMILES'])
        return eosce, embeddings