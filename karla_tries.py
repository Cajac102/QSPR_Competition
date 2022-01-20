from chempy.sdf import SDF
from glob import glob
import numpy as np
import pandas as pd
import time



def main():

    sdf_file = '/home/karla/Dokumente/4.WS2022/3.Cheminfo/Assignments/Projekt/Data/qspr-dataset-02.sdf'

    isdf = SDF(sdf_file)
    counter = 1

    id = []
    toxicity = []
    # iterate over all molecules
    while counter:
        r = isdf.read()
        if not r:
            break
        mol = r.get('MOL')
        print('Name:  {}'.format(mol[0]))
        id.append(mol[0])
        toxic = r.get_single('pLC50')
        print('LC50:  {}'.format(toxic))
        toxicity.append(toxic)
        counter += 1

    data = pd.DataFrame([])
    data['Name'] = id
    data['pLC50'] = toxicity
    print(data)

if __name__ == "__main__":
    main()