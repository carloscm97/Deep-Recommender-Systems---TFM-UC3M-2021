"""
Yao Wu et al., Collaborative denoising auto-encoders for top-n recommender systems. WSDM 2016.
https://alicezheng.org/papers/wsdm16-cdae.pdf
"""
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.binomial import Binomial
from models.BaseModel import BaseModel
from utils.Tools import apply_activation

class DAE(BaseModel):
    def __init__(self, model_conf, num_users, num_items, device):
        super(DAE, self).__init__()
        self.hidden_dim = model_conf.hidden_dim
        self.act = model_conf.act
        self.corruption_ratio = model_conf.corruption_ratio
        self.num_users = num_users
        self.num_items = num_items
        self.binomial = Binomial(total_count=1, probs=(1 - self.corruption_ratio))
        self.device = device

        self.encoder = nn.Linear(self.num_items, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.num_items)

        self.to(self.device)

    def forward(self, rating_matrix):
        # normalize
        user_degree = torch.norm(rating_matrix, 2, 1).view(-1, 1)   # user, 1
        item_degree = torch.norm(rating_matrix, 2, 0).view(1, -1)   # 1, item
        normalize = torch.sqrt(user_degree @ item_degree)
        zero_mask = normalize == 0
        normalize = torch.masked_fill(normalize, zero_mask.bool(), 1e-10)

        normalized_rating_matrix = rating_matrix / normalize
        # corruption
        if self.training == True:
            self.normalized_rating_matrix_01 = normalized_rating_matrix
        normalized_rating_matrix = F.dropout(normalized_rating_matrix, self.corruption_ratio, training=self.training)
        if self.training == True:
            self.normalized_rating_matrix = normalized_rating_matrix
        # AE
        enc = self.encoder(normalized_rating_matrix)
        enc = apply_activation(self.act, enc)

        dec = self.decoder(enc)

        return torch.sigmoid(dec)

    def train_one_epoch(self, dataset, optimizer, batch_size, verbose):
        self.train()
        
        # user, item, rating pairs
        train_matrix = dataset.train_matrix

        num_training = train_matrix.shape[0]
        num_batches = int(np.ceil(num_training / batch_size))

        perm = np.random.permutation(num_training)

        loss = 0.0
        for b in range(num_batches):
            optimizer.zero_grad()

            if (b + 1) * batch_size >= num_training:
                batch_idx = perm[b * batch_size:]
            else:
                batch_idx = perm[b * batch_size: (b + 1) * batch_size]

            batch_matrix = torch.FloatTensor(train_matrix[batch_idx].toarray()).to(self.device)
            pred_matrix = self.forward(batch_matrix)
            # cross_entropy
            batch_loss = F.binary_cross_entropy(pred_matrix, batch_matrix, reduction='sum')
            # batch_loss = batch_matrix * (pred_matrix + 1e-10).log() + (1 - batch_matrix) * (1 - pred_matrix + 1e-10).log()
            # batch_loss = -torch.sum(batch_loss)
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

            if verbose and b % 50 == 0:
                print('(%3d / %3d) loss = %.4f' % (b, num_batches, batch_loss))
        return loss

    def predict(self, eval_users, eval_pos, test_batch_size):
        with torch.no_grad():
            input_matrix = torch.FloatTensor(eval_pos.toarray()).to(self.device)
            preds = np.zeros(eval_pos.shape)

            num_data = input_matrix.shape[0]
            num_batches = int(np.ceil(num_data / test_batch_size))
            perm = list(range(num_data))
            for b in range(num_batches):
                if (b + 1) * test_batch_size >= num_data:
                    batch_idx = perm[b * test_batch_size:]
                else:
                    batch_idx = perm[b * test_batch_size: (b + 1) * test_batch_size]
                
                test_batch_matrix = input_matrix[batch_idx]
                batch_pred_matrix = self.forward(test_batch_matrix)
                preds[batch_idx] += batch_pred_matrix.detach().cpu().numpy()
        
        preds[eval_pos.nonzero()] = float('-inf')
        
        return preds