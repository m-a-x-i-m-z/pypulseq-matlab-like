# this script is based on the recon code / notebooks from the course "MR Physics with Pulseq" 
import numpy as np
import twixtools
from scipy.io import loadmat

import math
import matplotlib.pyplot as plt
import pypulseq as pp
#import mapvbvd
from recon_utils import reconstruct, read_raw_data, read_siemens_raw_data, plot_nd
import glob
import os

import h5py

# directory to be scanned for data and sequence files
data_path = '/home/zaitsev/range_software/pulseq/IceNIH_RawSend/'; 
#data_path='/dev/shm/mr0mat/';
#data_path='/dev/shm/koma_mat/';

files = glob.glob(os.path.join(data_path, "*.seq"))
files.sort(key=os.path.getmtime)

seq_file_path=files[-1]  # take the last modified sequence file

basic_filename = os.path.splitext(seq_file_path)[0]  # remove the extension

# load the sequence file
print(f'Loading sequence file \'{seq_file_path}\'')
seq = pp.Sequence()
seq.read(seq_file_path, detect_rf_use=True) 

seq_name_definition = seq.get_definition('Name');
if seq_name_definition is not None:
    print(f'Sequence name: {seq_name_definition}')
else:
    print('Sequence name is not defined in the sequence file.')

mat_file_path = basic_filename + '.mat'  # try MATLAB file
data_file_path = basic_filename + '.dat'  # try Siemens TWIX file

# Try to load MATLAB .mat file first
data_unsorted = None
if os.path.exists(mat_file_path):
    print(f'Attempting to load raw data from MATLAB file \'{mat_file_path}\'')

    try:    
        mat_data = loadmat(mat_file_path)
    except Exception as e:
        print(f'Falling back to h5py loader')
        mat_data = h5py.File(mat_file_path, 'r')

    # Try common variable names
    if 'data_unsorted' in mat_data:
        data_unsorted = mat_data['data_unsorted']
    elif 'kdata' in mat_data:
        data_unsorted = mat_data['kdata']
    else:
        # Try to find the first non-metadata key
        for key in mat_data.keys():
            if not key.startswith('__'):
                data_unsorted = mat_data[key]
                print(f'Using variable \'{key}\' from .mat file with shape {data_unsorted.shape}')
                break
    
    if data_unsorted is not None:
        data_unsorted = np.squeeze(np.array(data_unsorted['real']) + 1j * np.array(data_unsorted['imag']))  # Convert to numpy complex array
        if np.ndim(data_unsorted) != 3:
            data_unsorted=np.reshape(data_unsorted, (-1, 1, int(seq.adc_library.data[1][0])))  # todo: more correct detection rather than grabing the size of the first ADC event
        print(f'Loaded MATLAB file and restored dimensions: {data_unsorted.shape} ')

# Otherwise load Siemens TWIX file
if data_unsorted is None:
    print(f'Loading raw data file \'{data_file_path}\'')
    data_unsorted = read_siemens_raw_data(data_file_path)  # 3D numpy array [n_column, n_channel, acquisition_counter]

rec = reconstruct(data_unsorted, seq)

plot_nd(rec)
