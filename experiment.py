"""
@author: Kaist-ICLab/Sotiris
""" 
import os
import json
import pandas as pd
import warnings

# import custom modules
os.chdir(r'C:\Users\sotir\Documents\thesis')
from data_prep import KEMOCONDataModule
from utils import transform_label 
from models import LSTM, SVM, XGBoost, Att_LSTM

# import pytorch related stuff
import torch

# import pytorch-lightning related stuff
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class Experiment(object):
    
    def __init__(self, config):        
        # get configurations
        self.config = config
        # set seed
        pl.seed_everything(config['exp']['seed'])

        # prepare data
        self.dm = KEMOCONDataModule(
            config      = self.config['data'],
            label_fn    = transform_label(self.config['exp']['target'], self.config['exp']['pos_label']),
        )

        # get experiment name
        self.exp_name = self.config['exp']['model'] +'_'+ self.config['exp']['type'] +'_'+ self.config['exp']['target'] +'_'+ self.config['exp']['pos_label'] +'_'+ str(self.config['data']['num_segs'])
        print(f'Experiment: {self.exp_name}')

        # make directory to save results
        os.makedirs(self.config['exp']['savedir'], exist_ok=True)

        # set path to save experiment results
        self.savepath = os.path.join(self.config['exp']['savedir'], f'{self.exp_name}.json')
        
    def init_logger(self, pid):
        # set version number if needed
        version = '' if pid is None else f'_{pid:02d}'

        # make logger
        logger = TensorBoardLogger(
            save_dir    = self.config['logger']['logdir'],
            version     = version,
            name        = f'{self.exp_name}'
        )
        return logger

    def init_model(self, hparams):
        if self.config['exp']['model'] == 'xgboost':
            model = XGBoost(hparams)
        elif self.config['exp']['model'] == 'lstm':
            model = LSTM(hparams)
        elif self.config['exp']['model'] == 'svm':
            model = SVM(hparams)
        elif self.config['exp']['model'] == 'att_lstm':
            model = Att_LSTM(hparams)
            
        return model

    def _body(self, pid=None):
        # init model
        self.model = self.init_model(self.config['hparams'])

        # setup datamodule
        self.dm.setup(stage=None, test_id=None)

        # init training with pl.LightningModule models
        if self.config['trainer'] is not None:
            # init logger
            if self.config['logger'] is not None:
                logger = self.init_logger(pid=None)

            # init lr monitor and callbacks
            callbacks = list()
            if self.config['hparams']['scheduler'] is not None:
                callbacks.append(LearningRateMonitor(logging_interval='epoch'))

            # init early stopping
            if self.config['early_stop'] is not None:
                callbacks.append(EarlyStopping(**self.config['early_stop']))

            # make trainer
            trainer_args = self.config['trainer']
            trainer_args.update({
                'logger': logger,
                'callbacks': callbacks,
                'auto_lr_find': True if self.config['exp']['tune'] else False
            })
            trainer = pl.Trainer(**trainer_args)

            # find optimal lr
            if self.config['exp']['tune']:
                trainer.tune(self.model, datamodule=self.dm)
            
            # train model
            trainer.fit(self.model, self.dm)

            # test model and get results
            [results] = trainer.test(self.model, self.dm)

            # return metrics and confusion matrices
            metr = {
                'pid': pid,
                'acc': results['test_acc'],
                'ap': results['test_ap'],
                'f1': results['test_f1'],
                'auroc': results['test_auroc'],
                'num_epochs': self.model.current_epoch,
            }
            cm = self.model.cm
        
        else:
            # train model: concat train and valid inputs and labels and convert torch tensors to numpy arrays
            X_train, y_train = map(lambda x: torch.cat(x, dim=0).numpy(), zip(self.dm.kemocon_train[:], self.dm.kemocon_val[:]))
            self.model.train(X_train, y_train)

            # test model
            X_test, y_test = map(lambda x: x.numpy(), self.dm.kemocon_test[:])
            metr, cm = self.model.test(X_test, y_test)

        return metr, cm

    def run(self):
        # run holdout validation
        if self.config['exp']['type'] == 'holdout':
            metr, cm = self._body()
            results = {
                'config': self.config,
                'metrics': metr,
                'confmats': cm.tolist()
            }
            print(metr)
            print(cm)

        # run loso cv
        if self.config['exp']['type'] == 'loso':
            metrics, confmats = list(), dict()

            # for each participant
            for pid in self.dm.ids:
                # run loso cv and get results
                metr, cm = self._body(pid=pid)
                metrics.append(metr)
                confmats[pid] = cm.tolist()
                print(f'pid: {pid},\n{cm}')

            # convert metrics for each participant to json string
            metrics = pd.DataFrame(metrics).set_index('pid').to_json(orient='index')

            # make results dict
            results = {
                'config': self.config,
                'metrics': metrics,
                'confmats': confmats
            }
      
        # save results
        with open(self.savepath, 'w') as f:
            json.dump(results, f, indent=4)    
    
if __name__ == "__main__":
    
    # Opening JSON file
    with open(r'C:\Users\sotir\Documents\thesis\configs\svm_holdout.json', 'r') as f:
        config = json.load(f)
    
    # filter these RuntimeWarning messages
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in double_scalars')
    warnings.filterwarnings(action='ignore', message='divide by zero encountered in true_divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in subtract')
    
    # run experiment with configuration
    exp = Experiment(config)
    exp.run()    
    