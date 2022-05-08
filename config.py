"""
@author: Sotiris
"""

import json

config_xgb = {
        'data':{
            'data_dir': (r'C:\Users\sotir\Documents\thesis\segments'),
            'load_dir': (r'C:\Users\sotir\Documents\thesis\features\arousal-60.pkl'),
            'save_dir': None,
            'label_type': 'self',
            'batch_size': 2000,
            'n_classes': 2,
            'val_size': 0.1,
            'num_segs': 12,
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

config_lstm = {
        'logger': {
            'logdir': (r'C:\Users\sotir\Documents\thesis\logs')
            },
        'exp': {
            'seed': 5,
            'model': 'lstm',
            'type': 'holdout', # Hadjileontiadis uses K-fold cross validation
            'target': 'arousal',
            'pos_label': 'high',
            'savedir': (r'C:\Users\sotir\Documents\thesis\results'),
            'tune': True
            },
        'data': {
            'data_dir':  (r'C:\Users\sotir\Documents\thesis\segments'),
            'load_dir': None,
            'save_dir': (r'C:\Users\sotir\Documents\thesis\raw_signal\arousal-25.pkl'),
            'batch_size': 256, # Hadjileontiadis uses 800?
            'label_type': 'self',
            'n_classes': 2,
            'val_size': 0.1,
            'num_segs': 5,
            'resample': True,
            'extract_features': False,
            'standardize': True,
            'fusion': 'stack'
            },
        'early_stop': {
            'monitor': 'valid_acc',
            'min_delta': 0.0,
            'patience': 100,
            'verbose': True,
            'mode': 'max'
            },
        'trainer': {
            'gpus': 1,
            'auto_select_gpus': True,
            'precision': 16,
            'deterministic': True,
            'max_epochs': 500,
            'gradient_clip_val': 0.0
            },
        'hparams': {
            'inp_size': 4, # number of features
            'out_size': 1, # output of fully connected layer
            'hidden_size': 90, # Hadjileontiadis uses 100?
            'n_layers': 1, # the number of hidden layers (Hadjileontiadis uses 1 also)
            'p_drop': 0.0,
            'bidirectional': False,
            'learning_rate': 0.00085, # will be overrider if tune = True
            'scheduler': None
            }
     }

config_att_lstm = {
        'logger': {
            'logdir': (r'C:\Users\sotir\Documents\thesis\logs')
            },
        'exp': {
            'seed': 5,
            'model': 'att_lstm',
            'type': 'holdout', # Hadjileontiadis uses K-fold cross validation
            'target': 'arousal',
            'pos_label': 'high',
            'savedir': (r'C:\Users\sotir\Documents\thesis\results'),
            'tune': True
            },
        'data': {
            'data_dir':  (r'C:\Users\sotir\Documents\thesis\segments'),
            'load_dir': None,
            'save_dir': (r'C:\Users\sotir\Documents\thesis\raw_signal\arousal-30.pkl'),
            'batch_size': 256, # Hadjileontiadis uses 800?
            'label_type': 'self',
            'n_classes': 2,
            'val_size': 0.1,
            'num_segs': 6,
            'resample': True,
            'extract_features': False,
            'standardize': True,
            'fusion': 'stack'
            },
        'early_stop': {
            'monitor': 'valid_acc',
            'min_delta': 0.0,
            'patience': 100,
            'verbose': True,
            'mode': 'max'
            },
        'trainer': {
            'gpus': 1,
            'auto_select_gpus': True,
            'precision': 16,
            'deterministic': True,
            'max_epochs': 500,
            'gradient_clip_val': 0.0
            },
        'hparams': {
            'inp_size': 4, # number of features
            'out_size': 1, # output of fully connected layer
            'hidden_size': 100, # Hadjileontiadis uses 100?
            'n_layers': 1, # the number of hidden layers (Hadjileontiadis uses 1 also)
            'p_drop': 0.0,
            'bidirectional': False,
            'learning_rate': 0.00085, # will be overrider if tune = True
            'scheduler': None
            }
     }

config_svm = {
        'data':{
            'data_dir': (r'C:\Users\sotir\Documents\thesis\segments'),
            'load_dir': (r'C:\Users\sotir\Documents\thesis\features\arousal-30.pkl'),
            'save_dir': None,
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
            'seed': 5,
            'target': 'arousal',
            'pos_label': 'high',
            'model': 'svm',
            'type': 'holdout',
            'savedir': (r'C:\Users\sotir\Documents\thesis\results'),
            'tune': False,
            },
        'trainer': None,
        'hparams':{
            'C': 1,
            'kernel': 'rbf',
            'probability': True,
            'gamma': 0.001
            }
    }


#with open(r'C:\Users\sotir\Documents\thesis\configs\xgb_holdout.json', 'w') as outfile:
#    json.dump(config_xgb, outfile)
    
#with open(r'C:\Users\sotir\Documents\thesis\configs\lstm_holdout.json', 'w') as outfile:
#    json.dump(config_lstm, outfile)

#with open(r'C:\Users\sotir\Documents\thesis\configs\att_lstm_holdout.json', 'w') as outfile:
#    json.dump(config_att_lstm, outfile)

with open(r'C:\Users\sotir\Documents\thesis\configs\svm_holdout.json', 'w') as outfile:
    json.dump(config_svm, outfile)    

        