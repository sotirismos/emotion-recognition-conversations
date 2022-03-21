"""
@author: Sotiris
"""

import json

config_xgb = {
        'data':{
            'data_dir': (r'C:\Users\sotir\Documents\thesis\segments'),
            'load_dir': None,
            'save_dir': (r'C:\Users\sotir\Documents\thesis\features\arousal-50.pkl'),
            'label_type': 'self',
            'batch_size': 2000,
            'n_classes': 2,
            'val_size': 0.1,
            'num_segs': 10,
            'resample': False,
            'extract_features': True,
            'standardize': True,
            'fusion': 'stack',
            },
        'exp':{
            'seed': 5,
            'target': 'arousal',
            'pos_label': 'high',
            'model': 'xgboost',
            'type': 'holdout',
            'savedir': (r'C:\Users\sotir\Documents\thesis\results'),
            'tune': False,
            },
        'trainer': None,
        'hparams': {
            'bst':{
                'booster': 'gbtree',
                'verbosity': 1,
                'learning_rate': 0.5,
                'min_split_loss': 0,
                'max_depth': 8,
                'objective': 'binary:logitraw',
                'eval_metric': 'logloss',
                'seed': 1,
            },
            'num_rounds': 100,
            'threshold': 0.5
        }
    }        

with open(r'C:\Users\sotir\Documents\thesis\configs\xgb_holdout.json', 'w') as outfile:
    json.dump(config_xgb, outfile)
