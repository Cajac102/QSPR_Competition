# train a random forest regressor on data
# save to pLC_model.pickle for inference

from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem

import numpy as np
import pickle

import mordred
from mordred import Calculator, descriptors

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import cross_val_score


def is_organic(smile):
    """
    Function that tests if a smile is organic or not
    """
    # list containing organic atomic numbers
    organic_elements = {5, 6, 7, 8, 9, 15, 16, 17, 35, 53}

    try:
        mol = Chem.MolFromSmiles(smile)
        atom_num_list = set([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        organic_mol = (atom_num_list <= organic_elements)

        if organic_mol:
            return True
        else:
            return False
    except:
        return False


def add_hydrogens(x):
    """
    Add hydrogens to all molecules
    """
    x = Chem.AddHs(x)
    AllChem.EmbedMolecule(x, randomSeed=0xf00d)
    return x


def get_errors(y_true, y_pred, model_name="Model"):
    """
    This function computes three different measurements
    for model validation: Mean absulute error (MAE),
    Root mean squared error (rmse) and R².
    """
    err_mae = mae(y_true, y_pred).round(4)
    err_rmse = np.sqrt(mse(y_true, y_pred)).round(4)
    err_r2 = r2(y_true, y_pred).round(4)

    print(model_name + " MAE:" + str(err_mae) + " RMSE:" + str(err_rmse) + " R2:" + str(err_r2))

    return err_mae, err_rmse, err_r2


# Read Data:
file = '../../data/qspr-dataset-02.sdf'
molecules = PandasTools.LoadSDF(file,
                                smilesName='SMILES',
                                molColName='Molecule',
                                includeFingerprints=True) \
    .reset_index() \
    .set_index("ID") \
    .drop(columns=["index"])

# Filter out inorganic molecules
organic = molecules['SMILES'].apply(is_organic)
molecules = molecules.drop(molecules[-organic].index)

# Add hydrogens to all molecules
molecules["Molecule_processed"] = molecules["Molecule"].apply(lambda x: add_hydrogens(x))

# Set up descriptors with a Lasso coefficient > 0
calc = Calculator([mordred.Autocorrelation.ATSC(4, 'c'),
                   mordred.Autocorrelation.ATSC(3, 'dv'),
                   mordred.Autocorrelation.ATSC(5, 'se'),
                   mordred.Autocorrelation.ATSC(4, 'i'),
                   mordred.Autocorrelation.ATSC(6, 'i'),
                   mordred.Autocorrelation.AATSC(2, 'Z'),
                   mordred.Autocorrelation.MATS(1, 's'),
                   mordred.Autocorrelation.GATS(1, 'p'),
                   mordred.Autocorrelation.GATS(1, 'i'),
                   mordred.CPSA.RNCG,
                   mordred.EState.AtomTypeEState('count', 'dCH2'),
                   mordred.EState.AtomTypeEState('count', 'dsCH'),
                   mordred.EState.AtomTypeEState('count', 'sssN'),
                   mordred.EState.AtomTypeEState('sum', 'dssC'),
                   mordred.EState.AtomTypeEState('sum', 'dssS'),
                   mordred.InformationContent.InformationContent(5),
                   mordred.InformationContent.StructuralIC(5),
                   mordred.InformationContent.ModifiedIC(1),
                   mordred.InformationContent.ZModifiedIC(0),
                   mordred.LogS.LogS,
                   mordred.MoeType.PEOE_VSA(6),
                   mordred.MoeType.PEOE_VSA(9),
                   mordred.MoeType.PEOE_VSA(10),
                   mordred.MoeType.PEOE_VSA(12),
                   mordred.MoeType.SMR_VSA(4),
                   mordred.MoeType.SMR_VSA(9),
                   mordred.MoeType.SlogP_VSA(4),
                   mordred.MoeType.EState_VSA(2),
                   mordred.MoeType.EState_VSA(5),
                   mordred.MoeType.VSA_EState(9),
                   mordred.RingCount.RingCount(6, False, False, None, True),
                   mordred.RingCount.RingCount(9, False, True, None, None),
                   mordred.RingCount.RingCount(9, False, True, None, True),
                   mordred.RingCount.RingCount(9, False, True, False, None),
                   mordred.RingCount.RingCount(9, False, True, False, True),
                   mordred.SLogP.SLogP,
                   mordred.TopologicalCharge.TopologicalCharge('mean', 1)])

# Calculate descriptors
mordred_desc_frame = calc.pandas(molecules["Molecule_processed"])

# Select columns (Mordred sometimes gives back >1 value for a descriptor)
lasso_descriptors = ['ATSC4c', 'ATSC3dv', 'ATSC5se', 'ATSC4i', 'ATSC6i', 'AATSC2Z', 'MATS1s',
                     'GATS1p', 'GATS1i', 'RNCG', 'NdCH2', 'NdsCH', 'NsssN', 'SdssC', 'SdssS', 'IC5',
                     'SIC5', 'MIC1', 'ZMIC0', 'FilterItLogS', 'PEOE_VSA6', 'PEOE_VSA9', 'PEOE_VSA10',
                     'PEOE_VSA12', 'SMR_VSA4', 'SMR_VSA9', 'SlogP_VSA4', 'EState_VSA2',
                     'EState_VSA5', 'VSA_EState9', 'n6HRing', 'n9FRing', 'n9FHRing', 'n9FARing',
                     'n9FAHRing', 'SLogP', 'JGI1']
mordred_desc_frame = mordred_desc_frame[lasso_descriptors]

# Divide features and response
y = molecules[["pLC50"]].values
X = mordred_desc_frame

# train and save
# Perhaps add CV for n_estimators
rf_model = RandomForestRegressor(random_state=0, n_estimators=100)

r_squared_cv = cross_val_score(rf_model, X, y.ravel(), cv=5)
print("%0.2f R² with a standard deviation of %0.2f" % (r_squared_cv.mean(), r_squared_cv.std()))

mae_cv = cross_val_score(rf_model, X, y.ravel(), cv=5, scoring="neg_mean_absolute_error") * -1
print("%0.2f mean absolute error with a standard deviation of %0.2f" % (mae_cv.mean(), mae_cv.std()))

max_error_cv = cross_val_score(rf_model, X, y.ravel(), cv=5, scoring="max_error")
print("%0.2f max error with a standard deviation of %0.2f" % (max_error_cv.mean(), max_error_cv.std()))

rmse_cs = cross_val_score(rf_model, X, y.ravel(), cv=5, scoring="neg_root_mean_squared_error") * -1
print("%0.2f RMSE with a standard deviation of %0.2f" % (rmse_cs.mean(), rmse_cs.std()))

# Save model (use entire dataset for training, because we only have ~350 data points)
rf_model.fit(X, y.ravel())
pickle.dump(rf_model, open('../../models/pLC_model.sav', 'wb'))
