"""
@author: Sotiris/Kaist-ICLab 
"""
import os
import json
#import logging
from collections import OrderedDict


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




if __name__ == "__main__":
    
    PATHS = {
            'segments_dir': (r'C:\Users\sotir\Documents\thesis\segments')
            }
    
    pid_to_segments =  load_segments(PATHS['segments_dir'])