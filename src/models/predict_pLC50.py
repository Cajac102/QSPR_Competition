# gets: sdf file with molecules for which pLC values should be predicted
# e.g. predict_pLC50.py HEFLib.sdf
# output: creates csv in current folder with predictions

import pickle
import argparse
import os
import pandas as pd
import utils

print("""
      Thank you for chosing Karlas and Caros pLC 50 Predictor. 
      Your predictions will be available soon. 
      They will be stored in a .csv File in your current folder.
      """)

parser = argparse.ArgumentParser(description='Predict pLC50 values for molecules in an sdf')
parser.add_argument('input', metavar='i', help='an sdf file containing the molecules')
args = parser.parse_args()

# Parse SDF
filename = args.input
molecules = utils.parse_sdf(filename)
sdffilename = args.input.split("/")[-1].split(".")[0]

# Add hydrogens and 3D Structure to all molecules
molecules["Molecule"] = molecules["Molecule"].apply(lambda x: utils.preprocess(x))

# Remove out-of-application-domain molecules (charged/inorganic)
sus = molecules['Molecule'].apply(utils.is_suspicious)
molecules = molecules.drop(molecules[sus].index)

# compute descriptors
mordred_desc_frame = utils.calc_descriptors(molecules)

# read model (get path from script so that it can be used from anywhere)
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../../models/pLC_model.sav')
loaded_model = pickle.load(open(filename, 'rb'))

# inference tiiiiiimeeeee <3
pred_rf_test = loaded_model.predict(mordred_desc_frame)

# Add compound names
prediction_df = pd.DataFrame(list(zip(molecules.index.values.tolist(), pred_rf_test)), columns=["ID", "pLC50"])
prediction_df.to_csv("pLC_50_predictions_%s.csv" % sdffilename)
