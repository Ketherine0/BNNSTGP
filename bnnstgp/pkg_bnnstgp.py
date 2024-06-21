# #!/usr/bin/env python
# # coding: utf-8

# # In[1]:

# # from model import config

# ## import R packages
# # import os
# # os.environ['R_HOME'] = '/home/ellahe/.conda/envs/bnnstgp/lib/R'

# # import rpy2.robjects as robjects


# # from rpy2.robjects.packages import importr

# # utils = importr('utils')


# # from rpy2.robjects import r
# # # r.options(repos='https://repo.miserver.it.umich.edu/cran/')
# # # utils.install_packages('BayesGPfit', verbose = 0,repos = 'https://repo.miserver.it.umich.edu/cran/')
# # # GP = importr('BayesGPfit')
# # GP = importr('BayesGPfit')


# # In[2]:

# ## basic python import

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import random

from model.data_split import TrainTestSplit
from model.model_train import ModelTrain
from model.pre_data import LoadData
from model.post_pred import PostPred
from model.fdr_control import BetaFdr
from model.region_select import Selection

class BNN_model(nn.Module):
    """
    The BNN_model allows the creation, training and analysis of a Bayesian Neural Network (BNN) model with sophisticated control parameters for BNN structure, training and fdr control. 
    It includes specific steps for data loading, model training, beta posterior calculation, false discovery rate (FDR) control, and region selection output.
    """
    def __init__(self, coord, imgs, cov, Y, rep, path, nb_layer, thred, bg_image1, bg_image2, 
                 region_info1, region_info2, mni_file1, mni_file2, nii_path, fdr_thred=0.01,
                 lr=1e-3, input_dim=784, n_hid=None, n_hid2=None, n_hid3=None, 
                 n_hid4=None, hid_u=None, output_dim=1, w_dim=2, n_knots=66, 
                 N_train=200, phi=None, langevin=True, step_decay_epoch=100, step_gamma=0.1, 
                 act='relu', b_prior_sig=None, num_layer=None, n_epochs=120, lamb=10, batch_size=128, 
                 N_saves=30, test_every=10, n_img=1, a=0.01, b=20, poly_degree=15, device='cpu',train_ratio=0.8, fdr_path="Voxel"):
        super(BNN_model, self).__init__()
        self.coord = coord
        self.imgs = imgs
        self.cov = cov
        self.Y = Y
        self.rep = rep
        self.path = path
        self.nb_layer = nb_layer
        self.thred = thred
        self.bg_image1 = bg_image1
        self.bg_image2 = bg_image2
        self.region_info1 = region_info1
        self.region_info2 = region_info2
        self.mni_file1 = mni_file1
        self.mni_file2 = mni_file2
        self.nii_path = nii_path
        self.fdr_thred = fdr_thred
        self.a = a
        self.b = b
        self.poly_degree = poly_degree
        self.device = device
        
        self.lr = lr
        self.input_dim = input_dim
        self.n_hid = n_hid
        self.n_hid2 = n_hid2
        self.n_hid3 = n_hid3
        self.n_hid4 = n_hid4
        self.hid_u = hid_u
        self.output_dim = output_dim
        self.w_dim = w_dim
        self.n_knots = n_knots
        self.phi = phi
        self.lamb = lamb
        self.act = act
        if b_prior_sig:
            self.b1_prior_sig = []
            for i in range(n_img):
                self.b1_prior_sig.append(torch.Tensor(b_prior_sig[i]))
        else:
            self.b_prior_sig = None
        self.num_layer = num_layer
        self.n_img = n_img

        self.N_train = N_train
        self.langevin = langevin
        self.step_decay_epoch = step_decay_epoch
        self.step_gamma = step_gamma
        
        self.n_epochs = n_epochs
        self.lamb = lamb
        self.batch_size = batch_size
        self.N_saves = N_saves
        self.test_every = test_every
        self.train_ratio = train_ratio
        self.fdr_path = fdr_path
    
    def load_data(self):
        torch.cuda.empty_cache()
        self.imgs, self.Y, self.W, self.coord, self.phi = LoadData(self.coord, self.cov, self.imgs, self.Y, self.a, self.b, self.poly_degree, self.device).preprocess()
        
    def create_model_train(self):
        return ModelTrain(
            imgs=self.imgs, Y=self.Y, W=self.W, phi=self.phi, 
            n_hid=self.n_hid, n_hid2=self.n_hid2, n_hid3=self.n_hid3, n_hid4=self.n_hid4,
            path=self.path, nb_layer=self.nb_layer, device=self.device, lr=self.lr, 
            n_epochs=self.n_epochs, lamb=self.lamb, batch_size=self.batch_size, 
            N_saves=self.N_saves, test_every=self.test_every, n_img=self.n_img
        )
    
    def post_pred(self):
        PostPred(rep=self.rep, path=self.path, imgs=self.imgs,
                 Y=self.Y, W=self.W, phi=self.phi, n_img=self.n_img,
                 n_hid=self.n_hid, n_hid2=self.n_hid2, 
                 n_hid3=self.n_hid3, n_hid4=self.n_hid4, 
                 lr=self.lr, lamb=self.lamb, batch_size=self.batch_size, nb_layer=self.nb_layer).compute_cov()
        
    def beta_fdr_control(self, rep=None, path=None, train_ratio=None, fdr_path=None):
        if rep is not None:
            self.rep = rep
        if path is not None:
            self.path = path
        if train_ratio is not None:
            self.train_ratio = train_ratio
        if fdr_path is not None:
            self.fdr_path = fdr_path        
        self.beta_all = []
        for i in range(self.n_img):
            self.beta_all.append(BetaFdr(self.rep, self.path, self.lr, self.lamb, self.Y,
                                    n_hid=self.n_hid, n_hid2=self.n_hid2, 
                                    n_hid3=self.n_hid3, n_hid4=self.n_hid4, fdr_thred=self.fdr_thred,
                                    coord=self.coord, imgs=self.imgs, 
                                    W=self.W, hid_u=self.hid_u,
                                    phi=self.phi, nb_layer=self.nb_layer, 
                                    n_img=self.n_img, train_ratio=self.train_ratio).beta_fdr(i,self.fdr_path))
        return self.beta_all
    
    def output_selected(self, original_coord, fdr_path=None):
        if fdr_path is not None:
            self.fdr_path = fdr_path
        self.coord = original_coord
        for i in range(self.n_img):
            Selection(self.imgs, self.coord, self.rep, self.path, self.coord, self.thred, self.bg_image1, self.bg_image2, self.region_info1, self.region_info2, self.mni_file1, self.mni_file1, self.nii_path,self.n_img).output_selected_region(i, self.fdr_path)
