import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import torch
import rpy2.robjects as robjects
from tqdm import tqdm
from rpy2.robjects.packages import importr
from sys import platform
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
import time
from sklearn.linear_model import LinearRegression
import random
import os
import copy
from sklearn.linear_model import LinearRegression

from model.data_split import TrainTestSplit
from model.neuroimg_network import NeuroNet


class ModelTrain:
    """
    input the number of reptitions, the model saving path and file name, and the number of network layers
    example path: "model/neuroimg_prob"
    return the trained model and network weight
    """
    
    def __init__(self, imgs, Y, W, phi, rep, n_hid, n_hid2, n_hid3, n_hid4, path, nb_layer, device, lr=3e-4,    
                 n_epochs=120, 
                 lamb=10, batch_size=128,
                 N_saves=70, test_every=10, n_img=1):
        self.imgs = imgs
        self.Y = Y
        self.W = W
        self.phi = phi
        self.rep = rep
        self.n_hid = n_hid
        self.n_hid2 = n_hid2
        self.n_hid3 = n_hid3
        self.n_hid4 = n_hid4
        self.path = path
        self.nb_layer = nb_layer
        self.lr = lr
        self.n_epochs = n_epochs
        self.lamb = lamb
        self.batch_size = batch_size
        self.N_saves = N_saves
        self.test_every = test_every
        self.n_img = n_img
        self.device= device
        
    def train(self):
        for seed in range(self.rep):
            print("seed:",seed)
            start_epoch = 0

            np.random.seed(seed)
            torch.manual_seed(seed)
            input_dim = self.imgs[0].shape[1]
            train_ratio = 0.8
            
            train_idx = np.random.choice(np.arange(len(self.Y)), int(train_ratio * len(self.Y)), replace = False)
            test_idx = np.ones(len(self.Y), np.bool)
            test_idx[train_idx] = 0
            n_train = int(train_ratio * len(self.Y))
            n_test = len(self.Y)-n_train
            nbatch_train = (n_train+self.batch_size-1) // self.batch_size
            nbatch_test = (n_test+self.batch_size-1) // self.batch_size
            
            reg = LinearRegression().fit(self.W[train_idx,:self.W.shape[1]-1], self.Y[train_idx])
            reg_init = np.append(reg.coef_, reg.intercept_).astype(np.float32)
            n_knots = []
            for i in range(self.n_img):
                n_knots.append(self.phi[i].shape[1])
            
            net = NeuroNet(reg_init=reg_init, 
                           lr = self.lr, lamb = self.lamb, input_dim = input_dim, 
                           N_train=n_train, n_hid=self.n_hid, n_hid2=self.n_hid2,
                           n_hid3=self.n_hid3, n_hid4=self.n_hid4,
                           phi = self.phi, num_layer=self.nb_layer, w_dim = self.W.shape[1],
                           n_knots=n_knots, n_img=self.n_img,
                           step_decay_epoch = 200, step_gamma = 0.2, b_prior_sig = None,
                           langevin=False)

            nsamples0 = (n_train+self.batch_size-1) // self.batch_size
            nsamples1 = (n_test+self.batch_size-1) // self.batch_size

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            path = self.path+"/neuroimg_prob"+str(seed)+".pth"

            if os.path.exists(path):
                # y_train = None
                # y_test = None
                # y_train_pred = None
                # y_test_pred = None
                y_test = None
                y_test_pred = None

                checkpoint = torch.load(path)
                net.load_state_dict(checkpoint['model'])
                start_epoch = checkpoint['epoch']
                best_R2 = checkpoint['best_R2']
                weight_samples = checkpoint['weight_set_samples']
                print('For random split '+str(seed)+', loading epoch {} successfully!'.format(start_epoch))
                print('Best R2:',best_R2)

                # i = 0
                loss_all = np.zeros((nsamples0, len(weight_samples)))
#                 for ((x1, w), y),((x2, w), y) in zip(train_loader1, train_loader2):
#                     x1 = x1.float().to(device)
#                     x2 = x2.float().to(device)
#                     w = w.float().to(device)
#                     y = y.float().to(device).reshape(-1, 1)
#                     loss, out = net.fit(x1, x2, w, y)
                
#                     loss = self.compute_loss(net, weight_samples, x1, x2, w, y)
#                     loss_all[i,:]=loss
                cov_total = []
                MSE_total = []
                indices = list(test_idx)
                random.seed(seed)
                random.shuffle(indices)
                
                batch_n = 0
                while batch_n < nbatch_test:
                    batch_indices = np.asarray(indices[0:self.batch_size])  
                    indices = indices[self.batch_size:] + indices[:self.batch_size] 
                    x = []
                    for j in range(self.n_img):
                        x.append(torch.tensor(self.imgs[j][indices]).float().to(device))
                    w = torch.tensor(self.W[indices])
                    w = w.float().to(device)
                    y = torch.tensor(self.Y[indices])
                    y = y.float().to(device).reshape(-1, 1)
                    loss, out = net.eval(x, w, y)
                    tmp = out.cpu().detach().numpy()
                    
                    if y_test_pred is None:
                        y_test = y.detach().cpu().numpy()
                        # y_test = y
                        y_test_pred = tmp
                    else:
                        y_test = np.concatenate(([y_test, y.detach().cpu().numpy()]))
                        # y_test = np.concatenate(([y_test, y]))
                        y_test_pred = np.concatenate(([y_test_pred, tmp]))           

                    batch_n += 1
                MSE = (y_test_pred.reshape(-1)-y_test.reshape(-1))**2
                MSE_pd = pd.DataFrame(MSE.reshape(-1,1))
                MSE_pd.to_csv('MSE_T1/'+'itr'+str(seed)+'.csv')

        return R2_total
