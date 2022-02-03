# QSPR_Competition
Predicting compound toxicity for the QSPR Competition in Cheminformatics, WS21, University of Tuebingen

## Initial set up
1. `git clone https://github.com/Cajac102/QSPR_Competition.git`
2. `cd QSPR_Competition`
3. `conda env create -f qspr_comp.yml` \
    OR \
    `pip install -r requirements.txt`

## Making Predictions
( 1. Activate environment: `conda activate qspr_comp`) \
2. Run: `python ~/QSPR_Competition/src/models/predict_pLC50.py /path/to/your/sdf` \
3. Wait a few seconds - the predictions will be stored in a .csv file in your current folder.
