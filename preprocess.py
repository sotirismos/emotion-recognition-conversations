"""
@author: Sotiris/Kaist-ICLab 
"""

import os
import pandas as pd
import logging
#import argparse

def aggregate_raw(paths, valid_pids):
    
    """
    Aggregate raw data files by participant IDs and return a dict.
    Args:
        paths (dict of str: paths to K-EmoCon dataset files): requires,
               e4_dir (str): path to a directory of raw Empatica E4 data files saved as CSV files
               h7_dir (str): path to a directory of raw Polar H7 data files saved as CSV files
               valid_pids (list of int): a list containing valid participant IDs
    Returns:
               pid_to_raw_df (dict of int: pandas DataFrame): maps participant IDs to DataFrames containing raw data.
    """
    
    logger = logging.getLogger('default')
    e4_dir, h7_dir = paths['e4_dir'], paths['h7_dir']
    pid_to_raw_df = {}

    # store raw e4 data
    for pid in valid_pids:
        # get paths to e4 data files
        user_dir = os.path.join(e4_dir, str(pid))
        filepaths = [os.path.join(user_dir, f) for f in os.listdir(user_dir) if f != '.ipynb_checkpoints']  # if check may be redudant

        # store e4 data files to dict as k = "{uid}/{filetype}" -> v = DataFrame
        for filepath in filepaths:
            try:
                filetype = filepath.split('/')[-1].split('_')[-1].split('.')[0].lower()
                filekey = f'{pid}/{filetype}'
                data = pd.read_csv(filepath)

                # take care of multi-entry issue
                if pid == 31 and filetype == 'ibi':                           # This should be the case for pid == 29(excluded from our VALIDS list)
                    data = data.loc[data.device_serial == 'A01525'] 
                elif pid == 31:
                    data = data.loc[data.device_serial == 'A013E1']
                elif pid == 32:
                    data = data.loc[data.device_serial == 'A01A3A']           # This should be the case for pid == 30(excluded from our VALIDS list)

                pid_to_raw_df[filekey] = data

            except Exception as err:
                logger.warning(f'Following exception occurred while processing {filekey}: {err}')

    # store raw h7 data (stores only the Polar HR raw DataFrame) 
    for pid in valid_pids:
        # get path to h7 data file
        filepath = os.path.join(h7_dir, str(pid), 'Polar_HR.csv')

        # store h7 data files to dict as k = "{uid}/ecg" -> v = DataFrame
        try:
            filekey = f'{pid}/ecg'
            pid_to_raw_df[filekey] = pd.read_csv(filepath)

        except Exception as err:
            logger.warning(f'Following exception occurred while processing {filekey}: {err}')

    return pid_to_raw_df


PATHS = {
        'e4_dir': (r'C:\Users\sotir\Documents\thesis\Dataset\e4_data'),
        'h7_dir': (r'C:\Users\sotir\Documents\thesis\Dataset\neurosky_polar_data'),
        }

VALIDS = [1, 4, 5, 8, 9, 10, 11, 13, 14, 15, 16, 19, 22, 23, 24, 25, 26, 27, 28, 31, 32]
FILETYPES = ['bvp', 'eda', 'hr', 'ibi', 'temp', 'ecg']

pid_to_raw_df = aggregate_raw(PATHS, VALIDS)

