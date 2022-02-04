"""
This script contains preprocessing utilities and feature calculations
"""

from rdkit.Chem import PandasTools
import mordred
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools


def parse_sdf(filename):
    """
    Read in SDF File
    :param filename:
    :return:
    """
    df = PandasTools.LoadSDF(filename, smilesName='SMILES', molColName='Molecule', includeFingerprints=True) \
        .reset_index() \
        .set_index("ID") \
        .drop(columns=["index"])
    return df


def is_organic(smile):
    """
    Function that tests if a molecule is organic
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


def is_uncharged(x):
    """
    :param x: RDKit molecule
    :return: True if uncharged, False if charged
    """
    return Chem.GetFormalCharge(x) == 0


def is_suspicious(x):
    """
    Check if molecule is within applicability domain
    :param x: RDKit molecule
    :return: returns FALSE if molecule x is within applicability domain (organic + uncharged)
             returns TRUE if molecule x is either inorganic or charged
    """
    if is_organic(Chem.MolToSmiles(x)) and is_uncharged(x):
        return False
    return True


def preprocess(x):
    """
    Add hydrogens and 3D structure to all molecules
    """
    x = Chem.AddHs(x)
    AllChem.EmbedMolecule(x, randomSeed=0xf00d)
    return x


def calc_descriptors(molecule_df):
    """
    Calculates molecular features (descriptors)
    :param molecule_df: pandas DataFrame containing molecules
    Molecule column name has to be named "Molecule"
    :return: pandas DataFrame containing feature values
    """
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
    mordred_desc_frame = calc.pandas(molecule_df["Molecule"])

    # Select columns (Mordred sometimes gives back >1 value for a descriptor)
    lasso_descriptors = ['ATSC4c', 'ATSC3dv', 'ATSC5se', 'ATSC4i', 'ATSC6i', 'AATSC2Z', 'MATS1s',
                         'GATS1p', 'GATS1i', 'RNCG', 'NdCH2', 'NdsCH', 'NsssN', 'SdssC', 'SdssS', 'IC5',
                         'SIC5', 'MIC1', 'ZMIC0', 'FilterItLogS', 'PEOE_VSA6', 'PEOE_VSA9', 'PEOE_VSA10',
                         'PEOE_VSA12', 'SMR_VSA4', 'SMR_VSA9', 'SlogP_VSA4', 'EState_VSA2',
                         'EState_VSA5', 'VSA_EState9', 'n6HRing', 'n9FRing', 'n9FHRing', 'n9FARing',
                         'n9FAHRing', 'SLogP', 'JGI1']
    mordred_desc_frame = mordred_desc_frame[lasso_descriptors]

    return mordred_desc_frame
