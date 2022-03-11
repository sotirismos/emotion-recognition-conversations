"""
@author: Sotiris
"""

import json

config = {
        'data':{
            'data_dir': (r'C:\Users\sotir\Documents\thesis\segments'),
            'save_dir': (r'C:\Users\sotir\Documents\thesis\features\arousal-25.pkl'),
            'load_dir': None,
            'label_type': 'self',
            'batch_size': 2000,
            'n_classes': 2,
            'val_size': 0.1,
            'num_segs': 5,
            'resample': False,
            'extract_features': True,
            'standardize': True,
            'fusion': 'stack',
            },
        'exp':{
            'seed': 1,
            'target': 'arousal',
            'pos_label': 'high',
            'model': 'xgboost',
            'type': 'holdout',
            'savedir': (r'C:\Users\sotir\Documents\thesis\results'),
            'tune': False,
            },
        'hparams': {
            'bst':{
                'booster': 'xgbtree',
                'verbosity': 1,
                'learning_rate': 0.3,
                'min_split_loss': 0,
                'max_depth': 6,
                'objective': 'binary:logitraw',
                'eval_metric': 'auc',
                'seed': 1,
            },
            'num_rounds': 100,
            'threshold': 0.5
        }
    }        

with open(r'C:\Users\sotir\Documents\thesis\configs\xgb_holdout.json', 'w') as outfile:
    json.dump(config, outfile)
