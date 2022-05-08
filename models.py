"""
@author: Kaist-ICLab/Sotiris
"""
#import os
import numpy as np
import xgboost as xgb
from xgboost import DMatrix
from sklearn.svm import SVC

import torch
import pytorch_lightning as pl

from torch import nn
from torch.distributions.bernoulli import Bernoulli

from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score, confusion_matrix

class Att_LSTM(pl.LightningModule):

    def __init__(self, hparams):
        
        super().__init__()
        # save hyperparameters, anything assigned to self.hparams will be saved automatically
        self._hparams = hparams

        # define LSTM and fully-connected layer
        self.lstm = nn.LSTM(
            input_size      = self.hparams['inp_size'],
            hidden_size     = self.hparams['hidden_size'],
            num_layers      = self.hparams['n_layers'],
            dropout         = self.hparams['p_drop'],
            bidirectional   = self.hparams['bidirectional'],
            batch_first     = True
        )
        
        # define stuff for attention layer
        self.tanh = nn.Tanh()
        if self.hparams['bidirectional'] is True:
            self.context_vector = nn.Parameter(torch.randn(1, self.hparams['hidden_size'] * 2, 1))
        else:
            self.context_vector = nn.Parameter(torch.randn(1, self.hparams['hidden_size'], 1))    
        
        # define final fc layer
        if self.hparams['bidirectional'] is True:
            self.fc = nn.Linear(self.hparams['hidden_size'] * 2, self.hparams['out_size'])
        else:
            self.fc = nn.Linear(self.hparams['hidden_size'], self.hparams['out_size'])
        
        # define loss
        self.loss = nn.BCEWithLogitsLoss()
        
    
    def attention_layer(self, h):
        context_vector = self.context_vector.expand(h.shape[0], -1, -1) # B*H*1
        att_score = torch.bmm(self.tanh(h), context_vector) # B*L*H * B*H*1 -> B*L*1
        att_weight = torch.sigmoid(att_score) # B*L*1
        
        reps = torch.bmm(h.transpose(1, 2), att_weight).squeeze(dim=-1) # B*H*L * B*L*1 -> B*H*1 -> B*H
        reps = self.tanh(reps) # B*H
        
        return reps
        
    def forward(self, x):
        out, _ = self.lstm(x) # B*L*H
        reps = self.attention_layer(out) # B*reps
        logits = self.fc(reps)
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
        
    # if i want to do a scheduler, i need to configure the following function
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'])

        # configure learning rate scheduler if needed
        if self.hparams['scheduler'] is not None:
            if self.hparams['scheduler'].type == 'CosineAnnealingWarmRestarts':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.hparams['scheduler']['params'])
                return [optimizer], [scheduler]

            elif self.hparams['scheduler'].type == 'ReduceLROnPlateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, self.hparams['scheduler']['params'])
                return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'valid_loss'}
            
        else:
            return optimizer

class LSTM(pl.LightningModule):

    def __init__(self, hparams):
        
        super().__init__()
        # save hyperparameters, anything assigned to self.hparams will be saved automatically
        self._hparams = hparams

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
        
    # if i want to do a scheduler, i need to configure the following function
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'])

        # configure learning rate scheduler if needed
        if self.hparams['scheduler'] is not None:
            if self.hparams['scheduler'].type == 'CosineAnnealingWarmRestarts':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.hparams['scheduler']['params'])
                return [optimizer], [scheduler]

            elif self.hparams['scheduler'].type == 'ReduceLROnPlateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, self.hparams['scheduler']['params'])
                return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'valid_loss'}
            
        else:
            return optimizer

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

class SVM(object):
    
    def __init__(self, hparams):
        self.hparams = hparams
    
        self.model_init = SVC(
                C = self.hparams['C'],
                kernel = self.hparams['kernel'],
                probability = self.hparams['probability'],
                gamma = hparams['gamma']
                )
        
    def train(self, x, y, model=None):
        self.model = self.model_init.fit(x, np.ravel(y))
    
    def predict(self, x):
        probs = self.model.predict_proba(x)
        return probs
    
    def classify(self, p):
        if p < 0.5:
            return 0
        elif p > 0.5:
            return 1
        else:
            return np.random.binomial(1, 0.5)
    
    def test(self, x, y):
        probs = self.predict(x)
        preds = list(map(lambda p: self.classify(p), probs[:,-1]))

        # get metrics
        acc = accuracy_score(y, preds)
        ap = average_precision_score(y, probs[:,-1], average='weighted', pos_label=1)
        f1 = f1_score(y, preds, average='weighted', pos_label=1)
        auroc = roc_auc_score(y, probs[:,-1], average='weighted')
        cm = confusion_matrix(y, preds, normalize=None)
    
        return {'acc': acc, 'ap': ap, 'f1': f1, 'auroc': auroc}, cm
             