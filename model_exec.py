import os
import numpy as np
import torch

import models
from utils.Params import Params
from utils.Dataset import Dataset
from utils.Logger import Logger
from utils.Evaluator import Evaluator
from utils.Trainer import Trainer
from Namespace import Namespace

class model_exec:
    def __init__(self, model_name):
        self.model_name = model_name

    def initialize(self,data_name):
        conf = Namespace(model=self.model_name, data_dir="D:\Descargas\RecSys_PyTorch-master/data/", save_dir="D:\Descargas\RecSys_PyTorch-master/saves/", conf_dir="D:\Descargas\RecSys_PyTorch-master/conf/", seed=257,data_name=data_name)
        self.model_conf = Params(os.path.join(conf.conf_dir, conf.model.lower() + '.json'))
        self.model_conf.update_dict('exp_conf', conf.__dict__)
        np.random.seed(conf.seed)
        torch.random.manual_seed(conf.seed)
        self.device = torch.device('cpu')
        self.dataset = Dataset(
            data_dir=conf.data_dir,
            data_name=conf.data_name,
            train_ratio=0.8,
            device=self.device
        )

        self.log_dir = os.path.join('saves', conf.model)
        self.logger = Logger(self.log_dir)
        self.model_conf.save(os.path.join(self.logger.log_dir, 'config.json'))

        self.eval_pos, self.eval_target = self.dataset.eval_data()
        self.item_popularity = self.dataset.item_popularity
        self.evaluator = Evaluator(self.eval_pos, self.eval_target, self.item_popularity, self.model_conf.top_k)

        self.model_base = getattr(models, conf.model)
        self.model = self.model_base(self.model_conf, self.dataset.num_users, self.dataset.num_items, self.device)
        self.logger.info(self.model_conf)
        self.logger.info(self.dataset)

        self.trainer = Trainer(
            dataset=self.dataset,
            model=self.model,
            evaluator=self.evaluator,
            logger=self.logger,
            conf=self.model_conf
        )

    def run(self):
        self.trainer.train()
        