# gets: sdf file with molecules for which pLC values should be predicted
# e.g. do_predictions.py HEFLib.svg
# output: creates csv in current folder with predictions

import pickle
import argparse
import os
import pandas as pd
import utils


parser = argparse.ArgumentParser(description='Predict pLC50 values for molecules in an sdf')
parser.add_argument('input', metavar='i', help='an sdf file containing the molecules')
args = parser.parse_args()

# Parse SDF
filename = args.input
molecules = utils.parse_sdf(filename)
sdffilename = args.input.split("/")[-1].split(".")[0]

# Filter out inorganic molecules
organic = molecules['SMILES'].apply(utils.is_organic)
molecules = molecules.drop(molecules[-organic].index)

# Add hydrogens and 3D Structure to all molecules
molecules["Molecule_processed"] = molecules["Molecule"].apply(lambda x: utils.preprocess(x))

# read model (get path from script so that it can be used from anywhere)
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../../models/pLC_model.sav')
loaded_model = pickle.load(open(filename, 'rb'))

# compute descriptors
mordred_desc_frame = utils.calc_descriptors(molecules)

# inference tiiiiiimeeeee <3
pred_rf_test = loaded_model.predict(mordred_desc_frame)

# Add compound names
prediction_df = pd.DataFrame(list(zip(molecules.index.values.tolist(), pred_rf_test)), columns=["ID", "pLC50"])
prediction_df.to_csv("pLC_50_predictions_%s.csv" % sdffilename)
