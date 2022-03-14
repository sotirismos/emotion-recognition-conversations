"""
@author: Sotiris/Kaist-ICLab 
"""
import warnings
import os
import json
from tqdm import tqdm
import numpy as np
from utils import LoggingConfig
from collections import OrderedDict
#import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import StratifiedKFold

from pyteap.signals.bvp import acquire_bvp, get_bvp_features
from pyteap.signals.gsr import acquire_gsr, get_gsr_features
from pyteap.signals.hst import acquire_hst, get_hst_features
from pyteap.signals.ecg import get_ecg_features

def load_segments(segments_dir):
    segments = {}

    # for each participant
    for pid in os.listdir(segments_dir):
        segments.setdefault(int(pid), [])
        froot = os.path.join(segments_dir, pid)

        # for segments for a participant
        for fname in os.listdir(froot):
            # get labels, segment index, and path to json file
            labels = fname.split('-')[-1].split('.')[0]
            idx = int(fname.split('-')[1])
            fpath = os.path.join(froot, fname)

            # load json file and save to dict of pid: [segments = (idx, segment, labels)]
            with open(fpath) as f:
                seg = json.load(f)
                segments[int(pid)].append((idx, seg, labels))

    # return dict sorted by pid
    return OrderedDict(sorted(segments.items(), key=lambda x: x[0]))


def get_features(sig, sr, sigtype):
    if sigtype == 'bvp':
        features = get_bvp_features(acquire_bvp(sig, sr), sr)
    elif sigtype == 'eda':
        features = get_gsr_features(acquire_gsr(sig, sr, conversion=1e6), sr)
    elif sigtype == 'temp':
        features = get_hst_features(acquire_hst(sig, sr), sr)
    elif sigtype == 'ecg':
        features = get_ecg_features(sig)
    return features

def get_features_rolling(segments, n, labeltype, majority):

    X, y = {}, {}

    # for each participant
    for pid, segs in segments.items():
        segs = sorted(segs, key=lambda x: x[0])
        pbar = tqdm(range(len(segs) - n), desc=f'Subject {pid:02d}', ascii=True, dynamic_ncols=True)

        curr_X, curr_y = [], []
        for i in pbar:
            # get n consecutive segments from i-th segment
            curr_segs = segs[i:i + n]

            features = []
            # get features
            for sigtype, sr in [('bvp', 64), ('eda', 4), ('temp', 4), ('ecg', 1)]:
                sig = np.concatenate([sigs[sigtype] for _, sigs, _ in curr_segs])
                features.extend(get_features(sig, sr, sigtype))

            # skip if one or more feature is NaN
            if np.isnan(features).any() | np.isinf(features).any():
                logger.warning('One or more feature is NaN or inf, skipped.')
                continue
            
            if labeltype == 's':
                curr_a = [int(labels[0]) for _, _, labels in curr_segs]
                curr_v = [int(labels[1]) for _, _, labels in curr_segs]
            elif labeltype == 'p':
                curr_a = [int(labels[2]) for _, _, labels in curr_segs]
                curr_v = [int(labels[3]) for _, _, labels in curr_segs]
            elif labeltype == 'e':
                curr_a = [int(labels[4]) for _, _, labels in curr_segs]
                curr_v = [int(labels[5]) for _, _, labels in curr_segs]
            elif labeltype == 'sp':
                curr_a = [np.sum([int(labels[0]), int(labels[2])]) for _, _, labels in curr_segs]
                curr_v = [np.sum([int(labels[1]), int(labels[3])]) for _, _, labels in curr_segs]
            
            # take majority label
            if majority:
                a_values, a_counts = np.unique(curr_a, return_counts=True)
                v_values, v_counts = np.unique(curr_v, return_counts=True)
                a_val = a_values[np.argmax(a_counts)]
                v_val = v_values[np.argmax(v_counts)]
            # or take label of the last segment
            else:
                a_val, v_val = curr_a[-1], curr_v[-1]

            curr_X.append(features)
            if labeltype != 'sp':
                #curr_y.append([int(a_val > 2), int(v_val > 2)])              # For binary classification
                curr_y.append([int(a_val), int(v_val)])                      
            else:
                #curr_y.append([int(a_val > 5), int(v_val > 5)])              # For binary classification
                curr_y.append([int(a_val), int(v_val)])

        # stack features for current participant and apply standardization
        X[pid] = StandardScaler().fit_transform(np.stack(curr_X))
        y[pid] = np.stack(curr_y)

    return X, y

def get_features_discrete(segments, n, labeltype, majority):
    X, y = {}, {}

    # for each participant
    for pid, segs in segments.items():
        segs = sorted(segs, key=lambda x: x[0])
        pbar = tqdm(segs, desc=f'For subject {pid:02d}', ascii=True, dynamic_ncols=True)

        curr_X, curr_y, curr_segs = [], [], {}
        # for each segment
        for idx, signals, labels in pbar:
            # get labels and add to buffer
            s_a, s_v = int(labels[0]), int(labels[1])
            p_a, p_v = int(labels[2]), int(labels[3])
            e_a, e_v = int(labels[4]), int(labels[5])

            if labeltype == 's':
                curr_segs.setdefault('a', []).append(s_a)
                curr_segs.setdefault('v', []).append(s_v)
            elif labeltype == 'p':
                curr_segs.setdefault('a', []).append(p_a)
                curr_segs.setdefault('v', []).append(p_v)
            elif labeltype == 'e':
                curr_segs.setdefault('a', []).append(e_a)
                curr_segs.setdefault('v', []).append(e_v)
            elif labeltype == 'sp':
                curr_segs.setdefault('a', []).append(np.sum([s_a, p_a]))
                curr_segs.setdefault('v', []).append(np.sum([s_v, p_v]))

            # get signals and add to buffer
            for sigtype, sr in [('bvp', 64), ('eda', 4), ('temp', 4), ('ecg', 1)]:
                curr_segs.setdefault(sigtype, []).append(signals[sigtype])

                # if n segments are in buffer
                if len(curr_segs[sigtype]) == n:
                    # concat signals and get features
                    sig = np.concatenate(curr_segs.pop(sigtype))
                    features = get_features(sig, sr, sigtype)
                    curr_segs.setdefault('features', []).append(features)

            # if features are in the buffer, pop features and labels
            if 'features' in curr_segs:
                features = np.concatenate(curr_segs.pop('features'))
                # skip if one or more feature is NaN
                if np.isnan(features).any():
                    logger.warning('One or more feature is NaN, skipped.')
                    continue

                # take majority label
                if majority:
                    a_values, a_counts = np.unique(curr_segs.pop('a'), return_counts=True)
                    v_values, v_counts = np.unique(curr_segs.pop('v'), return_counts=True)
                    a_val = a_values[np.argmax(a_counts)]
                    v_val = v_values[np.argmax(v_counts)]
                # or take label of the last segment
                else:
                    a_val = curr_segs.pop('a')[-1]
                    v_val = curr_segs.pop('v')[-1]
                
                curr_X.append(features)
                if labeltype != 'sp':
                    curr_y.append([int(a_val>2), int(v_val>2)])
                else:
                    curr_y.append([int(a_val>5), int(v_val>5)])

                pbar.set_postfix({'processed': idx // n})

        # stack features for current participant and apply standardization
        X[pid] = StandardScaler().fit_transform(np.stack(curr_X))
        y[pid] = np.stack(curr_y)

    return X, y

def prepare_features_kemocon(segments_dir, n, labeltype, majority, rolling):
       
    # load segments
    pid_to_segments = load_segments(segments_dir)

    # extract features and labels
    if rolling:
        X, y = get_features_rolling(pid_to_segments, n, labeltype, majority)
    else:
        X, y = get_features_discrete(pid_to_segments, n, labeltype, majority)

    with open('C:\\Users\\sotir\\Documents\\thesis\\segments\\features.pickle', 'wb') as handle:
         pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)
         
    with open('C:\\Users\\sotir\\Documents\\thesis\\segments\\labels.pickle', 'wb') as handle:
         pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    logger = LoggingConfig('info', handler_type='stream').get_logger()

    INFO = {
            'segments_dir': (r'C:\Users\sotir\Documents\thesis\segments'),
            'label' : 's',       # type of label to use for classification, must be either "s"=self, "p"=partner, "e"=external, or "sp"=self+partner (default="s")
            'length': 5,          # number of consecutive 5s-signals in one segment, default is 5
            'majority': True,     # set majority label for segments, default is last
            'rolling' : False,    # get segments with rolling: e.g., s1=[0:n], s2=[1:n+1], ..., default is no rolling: e.g., s1=[0:n], s2=[n:2n], ...')
            'seed': 2300,         # seed for random number generator, default is 0
            'cv': 'kfold',        # type of cross-validation to perform, must be either ''kfold'' or ''loso''
            'splits': 4,          # number of folds for the k-fold stratified classification
            'shuffle': True,     # shuffle data before splitting to folds, default is no shuffle
            'target' : 'valence', # target label for classification
            'save_dir': (r'C:\Users\sotir\Documents\thesis\data')
            }
    
    # filter these RuntimeWarning messages
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in double_scalars')
    warnings.filterwarnings(action='ignore', message='divide by zero encountered in true_divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in subtract')
    
    prepare_features_kemocon(INFO['segments_dir'], INFO['length'], INFO['label'], INFO['majority'], INFO['rolling'])

