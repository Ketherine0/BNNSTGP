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
import math
from sklearn.metrics import r2_score

import os
os.environ['R_HOME'] = '/home/ellahe/.conda/envs/bnnstgp/lib/R'

import rpy2.robjects as robjects

from rpy2.robjects.packages import importr


utils = importr('utils')

# options(repos=c('https://repo.miserver.it.umich.edu/cran/'))
# utils.install_packages('BayesGPfit', verbose = 0,repos = 'https://repo.miserver.it.umich.edu/cran/')
GP = importr('BayesGPfit')

from BNNSTGP4.data_split import TrainTestSplit
from BNNSTGP4.neuroimg_network import NeuroNet


class ModelTrain:
    """
    input the number of reptitions, the model saving path and file name, and the number of network layers
    example path: "model/neuroimg_prob"
    return the trained model and network weight
    """
    
    def __init__(self, imgs, Y, W, phi, rep, n_hid, n_hid2, n_hid3, n_hid4, path, nb_layer, device, lr=3e-4,    
                 n_epochs=120, 
                 lamb=10, batch_size=128,
                 N_saves=30, test_every=10, n_img=1,train_ratio=0.8):
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
        self.train_ratio=train_ratio
        
    def train(self):
        torch.cuda.empty_cache()
    
        l = []
        R2_total = np.zeros(self.rep)
        random.seed(2023)
        np.random.seed(2023)
    
        v_list1 = GP.GP_generate_grids(d = 2, num_grids = 112,grids_lim=np.array([-3,3]))
        v_list2 = GP.GP_generate_grids(d = 2, num_grids = 112,grids_lim=np.array([-1,1]))
        true_beta1 = (0.5*v_list1[:,0]**2+v_list1[:,1]**2)<2
        true_beta2 = np.exp(-5*(v_list2[:,0]-1.5*np.sin(math.pi*np.abs(v_list2[:,1]))+1.0)**2)

        for seed in range(self.rep):
            tic = time.time()
            start_epoch = 0

            np.random.seed(seed)
            torch.manual_seed(seed)
            input_dim = self.imgs[0].shape[1]
            if self.rep == 1:
                train_ratio = 1
                # Use all data for training when rep=1
                train_idx = np.arange(len(self.Y))  # Select all indices for training
                test_idx = np.array([])  # No test indices
            else:
                train_ratio = 0.8
                train_idx = np.random.choice(np.arange(len(self.Y)), int(train_ratio * len(self.Y)), replace=False)
                test_idx = np.ones(len(self.Y), dtype=np.bool)
                test_idx[train_idx] = False

            n_train = len(train_idx)  # Adjusted for the case when rep=1
            n_test = len(test_idx)  # Adjusted for the case when rep=1, will be 0
            nbatch_train = (n_train + self.batch_size - 1) // self.batch_size
            nbatch_test = (n_test + self.batch_size - 1) // self.batch_size if n_test > 0 else 0  # Adjust for when n_test is 0

            reg = LinearRegression().fit(self.W[train_idx, :self.W.shape[1]-1].detach().cpu().numpy(), self.Y[train_idx].detach().cpu().numpy())
            reg_init = np.append(reg.coef_, reg.intercept_).astype(np.float32)
            n_knots = [self.phi[i].shape[1] for i in range(self.n_img)]

            net = NeuroNet(reg_init=reg_init, lr=self.lr, lamb=self.lamb, input_dim=input_dim, N_train=n_train, n_hid=self.n_hid, n_hid2=self.n_hid2,
                           n_hid3=self.n_hid3, n_hid4=self.n_hid4, phi=self.phi, num_layer=self.nb_layer, w_dim=self.W.shape[1],
                           n_knots=n_knots, n_img=self.n_img, step_decay_epoch=200, step_gamma=0.2, b_prior_sig=None, langevin=False)

            epoch = 0
            start_save = 30
            save_every = 1
            # N_saves = 70
            # test_every = 10
            print_every = 10
            # batch_size = 128

            loss_train = np.zeros(self.n_epochs)
            R2_train = np.zeros(self.n_epochs)

            loss_val = np.zeros(self.n_epochs)
            R2_val = np.zeros(self.n_epochs)

            # device = 'cuda' if torch.cuda.is_available() else 'cpu'

            best_R2 = 0

            path = self.path+str(seed)+".pth"

            for i in range(start_epoch, self.n_epochs):

                tic = time.time()
                y_train = None
                y_test = None
                y_train_pred = None
                y_test_pred = None

                indices = list(train_idx)
                random.seed(i)
                random.shuffle(indices)
                
                batch_n = 0
                while batch_n < nbatch_train:
                    batch_indices = np.asarray(indices[0:self.batch_size])  
                    indices = indices[self.batch_size:] + indices[:self.batch_size] 
                    x = []
                    for j in range(self.n_img):
                        # x.append(torch.tensor(self.imgs[j][indices]).float().to(self.device))
                        x.append(self.imgs[j][indices])
                    # w = torch.tensor(self.W[indices])
                    # w = w.float().to(self.device)
                    # y = torch.tensor(self.Y[indices])
                    # y = y.float().to(self.device).reshape(-1, 1)
                    w = self.W[indices]
                    y = self.Y[indices].reshape(-1, 1)
                    loss, out = net.fit(x, w, y)
                    loss_train[i] += loss
                    tmp = out.cpu().detach().numpy()
                    if y_train_pred is None:
                        # y_train = y
                        y_train = y.detach().cpu().numpy()
                        y_train_pred = tmp
                    else:
                        y_train = np.concatenate(([y_train, y.detach().cpu().numpy()]))
                        # y_train = np.concatenate(([y_train, y]))
                        y_train_pred = np.concatenate(([y_train_pred, tmp]))              
                    
                    batch_n += 1
                       
                
                net.scheduler.step()
                loss_train[i] /= n_train
                R2_train[i] = r2_score(y_train_pred.reshape(-1), y_train.reshape(-1))
                toc = time.time()
                # print(net.weight_set_samples)

                if i > start_save and i % save_every == 0:
                    net.save_net_weights(max_samples = self.N_saves)

                if i % print_every == 0:
                    toc = time.time()
                    print('Epoch %d, train time %.4f s, train MSE %.4f, train R2 %.3f' % (i, toc-tic, loss_train[i], 
                                                                                          R2_train[i]))
                    # print('  Epoch %d, test time %.4f s, test MSE %.4f, test R2 %.3f' % (i, toc-tic, loss_val[i], 
                                                                                            # R2_val[i]))

                if i % self.test_every == 0 and self.rep != 1:
                    with torch.no_grad():
                        tic = time.time()

                        indices = list(test_idx)
                        random.seed(i)
                        random.shuffle(indices)

                        y_test_concat = np.array([])
                        mean_Y_test_concat = np.array([])

                        batch_n = 0
                        while batch_n < nbatch_test:
                            batch_indices = np.asarray(indices[0:self.batch_size])  
                            indices = indices[self.batch_size:] + indices[:self.batch_size] 
                            x = [self.imgs[j][indices] for j in range(self.n_img)]  # Use test images from self.imgs
                            w = self.W[indices]
                            y = self.Y[indices].reshape(-1, 1)
                            loss, out = net.eval(x, w, y)

                            loss_val[i] += loss
                            tmp = out.cpu().detach().numpy()
                            y_test_batch = y.detach().cpu().numpy().flatten()
                            theta0 = 2
                            mean_Y_test_batch = theta0 + np.dot(self.imgs[0][indices], true_beta1) + np.dot(self.imgs[1][indices], true_beta2)
                            if y_test_pred is None:
                                y_test_pred = tmp
                            else:
                                y_test_pred = np.concatenate((y_test_pred, tmp))


                            if y_test_concat.size == 0:
                                y_test_concat = y_test_batch
                                mean_Y_test_concat = mean_Y_test_batch
                            else:
                                y_test_concat = np.concatenate((y_test_concat, y_test_batch))
                                mean_Y_test_concat = np.concatenate((mean_Y_test_concat, mean_Y_test_batch))

                            batch_n += 1

                        # Calculate real R2 based on the true function
                        real_R2 = np.corrcoef(y_test_concat, mean_Y_test_concat.flatten())[0,1]**2
                        print('real R2', real_R2)

                        # Calculate R2 for the test data
                        R2_val[i] = r2_score(y_test_concat, y_test_pred.flatten())
                        # print('R2 score', np.corrcoef(y_test_pred, y_test_concat))

                        best_R2 = max(best_R2, R2_val[i])
                        toc = time.time()
                        print('  Epoch %d, test time %.4f s, test MSE %.4f, test R2 %.3f, real R2 %.3f' % (i, toc-tic, loss_val[i], 
                                                                                                             R2_val[i], real_R2))

#                     print('  Epoch %d, test time %.4f s, test MSE %.4f, test R2 %.3f' % (i, toc-tic, loss_val[i], 
#                                                                                          R2_val[i]))

                if i==self.n_epochs-1:
                    weight_samples = net.weight_set_samples
                    # nb_hid = n_hid_li[self.nb_layer-1]
                    state = {'model':net.state_dict(), 'epoch':i, 'best_R2':best_R2, 
                             'weight_set_samples':weight_samples,
                            'num_hidden':self.n_hid,'batch_size':self.batch_size}
                    torch.save(state,self.path+"/neuroimg_prob"+str(seed)+".pth")

            l.append(best_R2)
            print(f'{seed}:Best test R2: = {best_R2}') 

            del net, reg_init

        return R2_total
