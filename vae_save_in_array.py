#!/bin/python

import numpy as np
import pandas as pd
import os
import argparse
import nibabel as nib

'''
Script to save data nii.gz in numpy array
Example:
python vae_save_in_array.py -dirname ./output_1000/vae_demo_features -n 1000 -outdir .
'''

def get_data_in_array(dirname):
    list_files = os.listdir(dirname)
    # load data
    L = []
    for file in list_files:
        try:
            vect = list(nib.load(dirname + "/" + file).get_data()[0,0,0,0,:])
            L.append(vect)
        except:
            print(file)
    # create the input 
    X = np.array(L)
    return (X)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Save VAE encoded features into a numpy array')
    parser.add_argument('-dirname', type=str, required=True, help='directory name of encoded features')
    parser.add_argument('-n', type=str, required=True, help='model number')
    parser.add_argument('-outdir', type=str, required=True, help='directory where to save data')
    args = parser.parse_args()

    X = get_data_in_array(args.dirname)
    np.save(args.outdir + "/X_" + args.n + ".npy", X)
