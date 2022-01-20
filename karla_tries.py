from chempy.sdf import SDF
from glob import glob
import numpy as np
import time



def main():

    sdf_file = '/home/karla/Dokumente/4.WS2022/3.Cheminfo/Assignments/Projekt/Data/qspr-dataset-02.sdf'

    isdf = SDF(sdf_file)
    counter = 1
    
    # iterate over all molecules
    while counter:
        r = isdf.read()
        if not r:
            break
        mol = r.get('MOL')
        energy = r.get_single('pLC50')

        print(energy)
        counter += 1


if __name__ == "__main__":
    main()