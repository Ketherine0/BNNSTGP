import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import torch
from tqdm import tqdm
from sys import platform
import time
from sklearn.linear_model import LinearRegression
import random

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

        # for seed in range(self.rep):
        for seed in range(self.rep):
            tic = time.time()
            start_epoch = 0

            np.random.seed(seed)
            torch.manual_seed(seed)
            input_dim = self.imgs[0].shape[1]
            train_ratio = self.train_ratio
            
            train_idx = np.random.choice(np.arange(len(self.Y)), int(train_ratio * len(self.Y)), replace = False)
            test_idx = np.ones(len(self.Y), dtype=np.bool_)
            test_idx[train_idx] = 0
            n_train = int(train_ratio * len(self.Y))
            n_test = len(self.Y)-n_train
            nbatch_train = (n_train+self.batch_size-1) // self.batch_size
            nbatch_test = (n_test+self.batch_size-1) // self.batch_size
            
            reg = LinearRegression().fit(self.W[train_idx,:self.W.shape[1]-1].detach().cpu().numpy(), self.Y[train_idx].detach().cpu().numpy())
            # reg = LinearRegression().fit(self.W[train_idx,:self.W.shape[1]-1], self.Y[train_idx])
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

            epoch = 0
            start_save = 0
            save_every = 1
            # N_saves = 70
            # test_every = 10
            print_every = 10

            loss_train = np.zeros(self.n_epochs)
            R2_train = np.zeros(self.n_epochs)

            loss_val = np.zeros(self.n_epochs)
            R2_val = np.zeros(self.n_epochs)

            # device = 'cuda' if torch.cuda.is_available() else 'cpu'

            best_R2 = 0

            path = self.path+str(seed)+".pth"

            for i in range(start_epoch, self.n_epochs):
                
                # print("Epoch: ",i)
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
                        x.append(self.imgs[j][indices])
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
                        y_train_pred = np.concatenate(([y_train_pred, tmp]))              
                    
                    batch_n += 1
                       
                net.scheduler.step()
                loss_train[i] /= n_train
                R2_train[i] = np.corrcoef(y_train_pred.reshape(-1), y_train.reshape(-1))[0,1]**2
                toc = time.time()
                              
                if i > start_save and i % save_every == 0:
                    net.save_net_weights(max_samples = self.N_saves)
                    
                if i % print_every == 0:
                    toc = time.time()
                    print('Epoch %d, train time %.4f s, train MSE %.4f, train R2 %.3f' % (i, toc-tic, loss_train[i], 
                                                                                          R2_train[i]))
                if train_ratio!=1:
                    if i % self.test_every == 0:
                        with torch.no_grad():
                            tic = time.time()

                            indices = list(test_idx)
                            random.seed(i)
                            random.shuffle(indices)

                            batch_n = 0
                            while batch_n < nbatch_test:
                                batch_indices = np.asarray(indices[0:self.batch_size])  
                                indices = indices[self.batch_size:] + indices[:self.batch_size] 
                                x = []
                                for j in range(self.n_img):
                                    x.append(self.imgs[j][indices])
                                w = self.W[indices]
                                y = self.Y[indices].reshape(-1, 1)
                                loss, out = net.eval(x, w, y)

                                loss_val[i] += loss
                                tmp = out.cpu().detach().numpy()
                                    
                                    
                                if y_test_pred is None:
                                    y_test = y.detach().cpu().numpy()
                                    y_test_pred = tmp
                                else:
                                    y_test = np.concatenate(([y_test, y.detach().cpu().numpy()]))
                                    y_test_pred = np.concatenate(([y_test_pred, tmp]))           

                                batch_n += 1


                            loss_val[i] /= n_test
                            R2_val[i] = np.corrcoef(y_test_pred.reshape(-1), y_test.reshape(-1))[0,1]**2
                            best_R2 = max(best_R2, R2_val[i])
                            toc = time.time()
                        toc = time.time()
                        print('  Epoch %d, test time %.4f s, test MSE %.4f, test R2 %.3f' % (i, toc-tic, loss_val[i], 
                                                                                             R2_val[i]))

                if i==self.n_epochs-1:
                    weight_samples = net.weight_set_samples
                    state = {'model':net.state_dict(), 'epoch':i, 'best_R2':best_R2, 
                             'weight_set_samples':weight_samples,
                            'num_hidden':self.n_hid,'batch_size':self.batch_size}
                    torch.save(state,self.path+"/neuroimg_prob"+str(seed)+".pth")

            l.append(best_R2)
            print(f'{seed}:Best test R2: = {best_R2}') 
            R2_total[seed] = best_R2

            del net, reg_init

        return R2_total
