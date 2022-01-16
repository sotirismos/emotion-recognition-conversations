"""
@author: Sotiris/Kaist-ICLab 
"""

import os
import pandas as pd
import logging
from datetime import datetime
import numpy as np
import json
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

    # store raw h7 data (stores only the Polar HR) (as ECG) 
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


def get_baseline_and_debate(paths, valid_pids, filetypes, pid_to_raw_df):
    
    """
    Split aggregated raw data files into baseline and debate dataframes.
    Args:
        paths (dict of str: paths to K-EmoCon dataset files): requires,
                   e4_dir (str): path to a directory of raw Empatica E4 data files saved as CSV files
                   h7_dir (str): path to a directory of raw Polar H7 data files saved as CSV files
                   subjects_info_path (str): csv file containing baseline and debate(start, end) times per pid as csv file
               valid_pids (list of int)
               filetypes (list of str) (3-axis acceleration excluded)
               pid_to_raw_df (dict of int: pandas DataFrame): maps participant IDs to DataFrames containing raw data
    Returns:
        pid_to_baseline_raw (dict of int: (dict of str: pandas Series))
        pid_to_debate_raw (dict of int: (dict of str: pandas Series))
    """
    
    logger = logging.getLogger('default')
    subject_info_table = pd.read_csv(paths['subjects_info_path'], index_col='pid')
    pid_to_baseline_raw = {pid:dict() for pid in valid_pids}
    pid_to_debate_raw = {pid:dict() for pid in valid_pids}

    # for each participant
    for pid in valid_pids:
        print('-' * 80)
        # get session info and timestamps
        subject_info = subject_info_table.loc[subject_info_table.index == pid]
        init_time, start_time, end_time = tuple(subject_info[['initTime', 'startTime', 'endTime']].to_numpy()[0]) # selects row and convets it to a NumPy array

        # get baseline interval
        baseline_td = 2 * 60 * 1e3  # 2 minutes (120s) in milliseconds
        baseline_start, baseline_end = init_time, init_time + baseline_td
        
        # for each filetype
        for filetype in filetypes:
            filekey = f'{pid}/{filetype}'
            try:
                data = pid_to_raw_df[filekey]
            except KeyError as err:
                logger.warning(f'Following exception occurred: {err}')
                continue

            # get baseline and debate portions of data
            baseline = data.loc[lambda x: (x.timestamp >= baseline_start) & (x.timestamp < baseline_end)]
            debate = data.loc[lambda x: (x.timestamp >= start_time) & (x.timestamp < end_time)]

            # skip the process if debate data is missing
            if debate.empty:
                logger.warning(f'Debate data missing for {filekey}, skipped')
                continue
            else:
                # try storing data with corresponding filekeys and printing extra information
                debate_len = datetime.fromtimestamp(max(debate.timestamp) // 1e3) - datetime.fromtimestamp(min(debate.timestamp) // 1e3)
                pid_to_debate_raw[pid][filetype] = debate.set_index('timestamp').value  # is a series

                # however, baseline data might be missing, so take care of that
                try:
                    baseline_len = datetime.fromtimestamp(max(baseline.timestamp) // 1e3) - datetime.fromtimestamp(min(baseline.timestamp) // 1e3)
                    pid_to_baseline_raw[pid][filetype] = baseline.set_index('timestamp').value  # is a series
                    print(f'For {filekey}:\t baseline - {baseline_len}: {len(baseline):5} \t|\t debate - {debate_len}: {len(debate):5}')
                except ValueError:
                    print(f'WARNING - Baseline data missing for {filekey} \t|\t debate - {debate_len}: {len(debate):5}')

    print('-' * 80)
    return pid_to_baseline_raw, pid_to_debate_raw


def baseline_to_json(paths, pid_to_baseline_raw):
    save_dir = paths['baseline_dir']
    # create a new directory if there isn't one already
    os.makedirs(save_dir, exist_ok=True)

    # for each participant
    for pid, baseline in pid_to_baseline_raw.items():

        # resample and interpolate ECG signals as they have duplicate entries while the intended frequency of ECG is 1Hz
        if 'ecg' in baseline.keys():
            ecg = baseline['ecg']
            ecg.index = pd.DatetimeIndex(ecg.index * 1e6)
            ecg = ecg.resample('1S').mean().interpolate(method='time')
            ecg.index = ecg.index.astype(np.int64) // 1e6
            baseline['ecg'] = ecg

        # convert sig values to list
        baseline = {sigtype: sig.values.tolist() for sigtype, sig in baseline.items() if sigtype in ['bvp', 'eda', 'temp', 'ecg']}

        # save baseline as json file
        savepath = os.path.join(save_dir, f'p{pid:02d}.json')
        with open(savepath, 'w') as f:
            json.dump(baseline, f, sort_keys=True, indent=4)

    return


PATHS = {
        'e4_dir': (r'C:\Users\sotir\Documents\thesis\dataset\e4_data'),
        'h7_dir': (r'C:\Users\sotir\Documents\thesis\dataset\neurosky_polar_data'),
        'subjects_info_path':(r'C:\Users\sotir\Documents\thesis\dataset\metadata\subjects.csv'),
        'baseline_dir': (r'C:\Users\sotir\Documents\thesis\baseline')
        }

VALIDS = [1, 4, 5, 8, 9, 10, 11, 13, 14, 15, 16, 19, 22, 23, 24, 25, 26, 27, 28, 31, 32]
FILETYPES = ['bvp', 'eda', 'hr', 'ibi', 'temp', 'ecg']                        # 3-axis acceleration is excluded

pid_to_raw_df = aggregate_raw(PATHS, VALIDS)
pid_to_baseline_raw, pid_to_debate_raw = get_baseline_and_debate(PATHS, VALIDS, FILETYPES, pid_to_raw_df)   
baseline_to_json(PATHS, pid_to_baseline_raw)  
     

