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
from models import XGBoost, LSTM

# import pytorch related stuff
import torch

# import pytorch-lightning related stuff
import pytorch_lightning as pl

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

    def init_model(self, hparams):
        if self.config['exp']['model'] == 'xgboost':
            model = XGBoost(hparams)
        elif self.config['exp']['model'] == 'lstm':
            model = LSTM(hparams)
    
        return model

    def body(self, pid=None):
        # init model
        self.model = self.init_model(self.config['hparams'])

        # setup datamodule
        self.dm.setup(stage=None, test_id=None)

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
            metr, cm = self.body()
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
    with open(r'C:\Users\sotir\Documents\thesis\configs\xgb_holdout.json', 'r') as f:
        config = json.load(f)
      
    # filter these RuntimeWarning messages
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in double_scalars')
    warnings.filterwarnings(action='ignore', message='divide by zero encountered in true_divide')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in subtract')
    
    # run experiment with configuration
    exp = Experiment(config)
    exp.run()    