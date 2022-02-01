# gets: sdf file with molecules for which pLC values should be predicted
# e.g. do_predictions.py HEFLib.svg
# output: creates csv in current folder with predictions

import pickle
import argparse
import os

import pandas as pd
from rdkit.Chem import PandasTools
import mordred
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem

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


parser = argparse.ArgumentParser(description='Predict pLC50 values for molecules in an sdf')
parser.add_argument('input', metavar='i', help='an sdf file containing the molecules')
args = parser.parse_args()

file = args.input
sdffilename = args.input.split("/")[-1].split(".")[0]
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
print(molecules)

# read model (get path from script so that it can be used from anywhere)
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../../models/pLC_model.sav')
loaded_model = pickle.load(open(filename, 'rb'))

# compute descriptors

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
mordred_desc_frame = calc.pandas(molecules["Molecule"])

# Select columns (Mordred sometimes gives back >1 value for a descriptor)
lasso_descriptors = ['ATSC4c', 'ATSC3dv', 'ATSC5se', 'ATSC4i', 'ATSC6i', 'AATSC2Z', 'MATS1s',
                     'GATS1p', 'GATS1i', 'RNCG', 'NdCH2', 'NdsCH', 'NsssN', 'SdssC', 'SdssS', 'IC5',
                     'SIC5', 'MIC1', 'ZMIC0', 'FilterItLogS', 'PEOE_VSA6', 'PEOE_VSA9', 'PEOE_VSA10',
                     'PEOE_VSA12', 'SMR_VSA4', 'SMR_VSA9', 'SlogP_VSA4', 'EState_VSA2',
                     'EState_VSA5', 'VSA_EState9', 'n6HRing', 'n9FRing', 'n9FHRing', 'n9FARing',
                     'n9FAHRing', 'SLogP', 'JGI1']
mordred_desc_frame = mordred_desc_frame[lasso_descriptors]

# inference tiiiiiimeeeee <3
pred_rf_test = loaded_model.predict(mordred_desc_frame)

# Add compound names
prediction_df = pd.DataFrame(list(zip(molecules.index.values.tolist(), pred_rf_test)), columns=["ID", "pLC50"])
prediction_df.to_csv("pLC_50_%s.csv"%sdffilename)
