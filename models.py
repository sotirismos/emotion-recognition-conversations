"""
@author: Kaist-ICLab/Sotiris
"""
#import os
#import numpy as np
#import xgboost as xgb
#from xgboost import DMatrix

import torch
#import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data

from torch import nn
#from torch.utils.data import DataLoader, random_split
from torch.distributions.bernoulli import Bernoulli

from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score, confusion_matrix
#from utils import get_config

class LSTM(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        # save hyperparameters, anything assigned to self.hparams will be saved automatically
        self.hparams = hparams

        # define LSTM and fully-connected layer
        self.lstm = nn.LSTM(
            input_size      = self.hparams['inp_size'],
            hidden_size     = self.hparams['hidden_size'],
            num_layers      = self.hparams['n_layers'],
            dropout         = self.hparams['p_drop'],
            bidirectional   = self.hparams['bidirectional'],
            batch_first     = True
        )
        if self.hparams['bidirectional'] is True:
            self.fc = nn.Linear(self.hparams['hidden_size'] * 2, self.hparams['out_size'])
        else:
            self.fc = nn.Linear(self.hparams['hidden_size'], self.hparams['out_size'])
        
        # define loss
        self.loss = nn.BCEWithLogitsLoss()
    
    @auto_move_data
    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.fc(out)[:, -1]  # if batch_first=True
        return logits

    def classify(self, p):
        be = Bernoulli(torch.tensor([0.5]))
        if p < 0.5:
            return 0
        elif p > 0.5:
            return 1
        else:
            return be.sample()

    def log_metrics(self, loss, logits, y, stage):
        # see https://discuss.pytorch.org/t/should-it-really-be-necessary-to-do-var-detach-cpu-numpy/35489
        # and https://stackoverflow.com/questions/63582590/why-do-we-call-detach-before-calling-numpy-on-a-pytorch-tensor
        # for converting tensors to numpy for calculating metrics
        # https://pytorch-lightning.readthedocs.io/en/latest/performance.html
        probas = torch.sigmoid(logits).detach().cpu().numpy()
        preds = torch.tensor(list(map(lambda p: self.classify(p), probas)))
        y = y.detach().cpu().numpy()

        # for the choice of metrics, see https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc
        acc = accuracy_score(y, preds)
        ap = average_precision_score(y, probas, average='weighted', pos_label=1)
        f1 = f1_score(y, preds, average='weighted', pos_label=1)
        auroc = roc_auc_score(y, probas, average='weighted')

        # converting scalars to tensors to prevent errors
        # see https://github.com/PyTorchLightning/pytorch-lightning/issues/3276
        self.log_dict({
            f'{stage}_loss': loss,
            f'{stage}_acc': torch.tensor(acc),
            f'{stage}_ap': torch.tensor(ap),
            f'{stage}_f1': torch.tensor(f1),
            f'{stage}_auroc': torch.tensor(auroc)
        }, on_step=False, on_epoch=True, logger=True)

        if stage == 'test':
            cm = confusion_matrix(y, preds, normalize=None)
            return cm

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log_metrics(loss, logits, y, stage='train')
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log_metrics(loss, logits, y, stage='valid')
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        cm = self.log_metrics(loss, logits, y, stage='test')
        return cm

    def test_epoch_end(self, outputs):
        # save test confusion matrix
        self.cm = sum(outputs)

    def configure_optimizers(self):