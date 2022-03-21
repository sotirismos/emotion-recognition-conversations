"""
@author: Kaist-ICLab/Sotiris
"""
#import os
import numpy as np
import xgboost as xgb
from xgboost import DMatrix

#import torch
#import torch.nn.functional as F
#import pytorch_lightning as pl
#from pytorch_lightning.core.decorators import auto_move_data

#from torch import nn
#from torch.utils.data import DataLoader, random_split
#from torch.distributions.bernoulli import Bernoulli

from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score, confusion_matrix
        
class XGBoost(object):

    def __init__(self, hparams):
        self.hparams = hparams

    def train(self, x, y, model=None):
        self.bst = xgb.train(
            params          = (self.hparams['bst']),
            dtrain          = DMatrix(x, label=y),
            num_boost_round = self.hparams['num_rounds'],
            xgb_model       = model
        )

    def predict(self, x):
        logits = self.bst.predict(DMatrix(x))
        return logits

    def classify(self, p):
        if p < 0.5:
            return 0
        elif p > 0.5:
            return 1
        else:
            return np.random.binomial(1, 0.5)

    def test(self, x, y):
        logits = self.predict(x)
        probs = 1 / (1 + np.exp(-logits))  # apply sigmoid to get probabilities
        preds = list(map(lambda p: self.classify(p), probs))

        # get metrics
        acc = accuracy_score(y, preds)
        ap = average_precision_score(y, probs, average='weighted', pos_label=1)
        f1 = f1_score(y, preds, average='weighted', pos_label=1)
        auroc = roc_auc_score(y, probs, average='weighted')
        cm = confusion_matrix(y, preds, normalize=None)

        return {'acc': acc, 'ap': ap, 'f1': f1, 'auroc': auroc}, cm

        
    
    