# train a random forest regressor on data
# save to pLC_model.pickle for inference

import utils
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


# Read Data:
file = '../../data/qspr-dataset-02.sdf'
molecules = utils.parse_sdf(file)

# Filter out inorganic molecules
organic = molecules['SMILES'].apply(utils.is_organic)
molecules = molecules.drop(molecules[-organic].index)

# Add hydrogens and 3D structure to all molecules
molecules["Molecule"] = molecules["Molecule"].apply(lambda x: utils.preprocess(x))

# Set up descriptors with a Lasso coefficient > 0
mordred_desc_frame = utils.calc_descriptors(molecules)

# Divide features and response
y = molecules[["pLC50"]].values
X = mordred_desc_frame

# train and save
# Perhaps add CV for n_estimators
rf_model = RandomForestRegressor(random_state=0, n_estimators=100)

print("Quality measures for Random Forest Regressor Model, as calculated by 5-Fold Cross Validation: \n")
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
