#!/usr/bin/env python
# coding: utf-8

# In[1]:


## import R packages
import os
os.environ['R_HOME'] = '/home/ellahe/.conda/envs/BNNSTGP/lib/R'

import rpy2.robjects as robjects


from rpy2.robjects.packages import importr

utils = importr('utils')



# options(repos=c('https://repo.miserver.it.umich.edu/cran/'))
# utils.install_packages('BayesGPfit', verbose = 0,repos = 'https://repo.miserver.it.umich.edu/cran/')
GP = importr('BayesGPfit')


# In[2]:


## basic python import

import numpy as np
import os
import matplotlib.pyplot as plt
import copy
import time
import random
import pandas as pd
import rpy2.robjects as robjects
import numpy as np
from rpy2.robjects import pandas2ri
readRDS = robjects.r['readRDS']
import h5py


# %matplotlib inline

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.optimizer import Optimizer, required
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# from tqdm.notebook import tqdm
from tqdm import tqdm
import SimpleITK as sitk
import nibabel as nib
import scipy.stats
from scipy.stats import invgamma
import math

import warnings
warnings.filterwarnings("ignore")

print("Start")
if torch.cuda.is_available():
    print('using GPU')
else:
    print('using CPU')

torch.set_default_dtype(torch.float32)
pandas2ri.activate()


# In[3]:
import sys
import os.path
this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir)


from model.neuroimg_network import NeuroNet
from model.data_split import TrainTestSplit
from model.model_train_all import ModelTrain
from model.pre_data import LoadData
from model.beta_posterior import BetaPost
from model.fdr_control2 import BetaFdr
from model.region_select2 import Selection


# In[4]:


class BNN_model(nn.Module):
    
    def __init__(self, coord, imgs, cov, Y, rep, path, nb_layer, thred, bg_image, nii_path, 
                     lr=1e-3, input_dim=784, n_hid = None, n_hid2 = None, n_hid3 = None, 
                     n_hid4 = None, hid_u=None, output_dim = 1, w_dim = 1, n_knots = 66, 
                     N_train=200, phi=None, langevin = True, step_decay_epoch = 100, step_gamma = 0.1, 
                     act = 'relu', b_prior_sig=None, num_layer=None, n_epochs=120, lamb=10, batch_size=128, 
                     N_saves=30, test_every=10, n_img=1, a=0.01, b=20, poly_degree = 15, device='cpu',train_ratio=0.8):
        super(BNN_model, self).__init__()
        self.coord = coord
        self.imgs = imgs
        self.cov = cov
        self.Y = Y
        self.rep = rep
        self.path = path
        self.nb_layer = nb_layer
        self.thred = thred
        self.bg_image = bg_image
        self.nii_path = nii_path
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
    
    def load_data(self):
        torch.cuda.empty_cache()
        self.imgs, self.Y, self.W, self.coord, self.phi = LoadData(self.coord, self.cov, self.imgs, self.Y, self.a, self.b, self.poly_degree, self.device).preprocess()
        
    def train_model(self):
        R2 = ModelTrain(imgs=self.imgs, Y=self.Y, W=self.W, phi=self.phi, rep=self.rep, 
                                 path=self.path, n_hid=self.n_hid, n_hid2=self.n_hid2, 
                                 n_hid3=self.n_hid3, n_hid4=self.n_hid4, nb_layer=self.nb_layer, lr=self.lr, n_epochs=self.n_epochs, 
                                 lamb=self.lamb, batch_size=self.batch_size, N_saves=self.N_saves, 
                                 test_every=self.test_every, n_img=self.n_img, device=self.device, train_ratio=self.train_ratio).train()
        # return BNN_net, R2
    
    def beta_post(self):
        BetaPost(rep=self.rep, path=self.path, imgs=self.imgs,
                 Y=self.Y, W=self.W, phi=self.phi, n_img=self.n_img,
                 n_hid=self.n_hid, n_hid2=self.n_hid2, 
                 n_hid3=self.n_hid3, n_hid4=self.n_hid4, 
                 lr=self.lr, lamb=self.lamb, batch_size=self.batch_size, nb_layer=self.nb_layer).compute_cov()
        
    def beta_fdr_control(self):
#         self.beta_all1, self.beta_all2 = BetaFdr(self.rep, self.path, self.lr, self.lamb, 
#                                                         self.n_hid, self.Y, coord=self.coord, imgs=self.imgs, 
#                                                         W=self.W,
#                                                         phi=self.phi, nb_layer=self.nb_layer, n_img=self.n_img).beta_fdr()
        self.beta_all = []
        for i in range(self.n_img):
            self.beta_all.append(BetaFdr(self.rep, self.path, self.lr, self.lamb, self.Y,
                                    n_hid=self.n_hid, n_hid2=self.n_hid2, 
                                    n_hid3=self.n_hid3, n_hid4=self.n_hid4,  
                                    coord=self.coord, imgs=self.imgs, 
                                    W=self.W, hid_u=self.hid_u,
                                    phi=self.phi, nb_layer=self.nb_layer, 
                                    n_img=self.n_img).beta_fdr(i))
    
    def output_selected(self):
        for i in range(self.n_img):
            # beta_matched = Selection(self.rep, self.path, self.coord[i], self.beta_all[i], self.thred, self.bg_image, self.nii_path, self.n_img).unit_match()
            # Selection(self.rep, self.path, self.coord[i], self.beta_all[i], self.thred, self.bg_image, self.nii_path,self.n_img).output_selected_region(beta_matched,i)
            Selection(self.rep, self.path, self.coord[i], self.beta_all[i], self.thred, self.bg_image, self.nii_path,self.n_img).output_selected_region(self.beta_all[i],i)

 


# Y = pd.read_csv("y1.csv").iloc[:,1].values
# idx = np.invert(np.isnan(Y))
# Y = Y[idx]



# hf = h5py.File('image1.hdf5', 'r')
# img1 = hf.get('img')['img1'][()][idx,:]
# # img2 = hf.get('img')['img2'][()][idx,:]

# h2 = h5py.File('coord1.hdf5', 'r')
# coord1 = h2.get('coord')['coord1'][()]
# # coord2 = h2.get('coord')['coord2'][()]

# hf = h5py.File('image_fMRI2.hdf5', 'r')
# img2 = hf.get('img')['img_fMRI'][()][idx,:]

# h2 = h5py.File('coord_fMRI2.hdf5', 'r')
# coord2 = h2.get('coord')['coord_fMRI'][()]

# coord = [coord1, coord2]
# img_data = [img1, img2]



### train both
# BNN_neuroimg = BNN_model(coord=coord, imgs=img_data, cov=np.zeros((img_data[0].shape[0],1)),
#                          Y=Y,rep=1,a=0.01,b=100, poly_degree=18, N_saves=70,
#                          lamb=10,n_hid=128, n_hid2=16, lr=3e-3,path="/nfs/turbo/jiankanggroup/ellahe/multi_test_resize_all",nb_layer=2, n_epochs=151,
#                         thred = 0.5, bg_image = "../data/neuroimaging/AAL_MNI_2mm.nii", batch_size=128,
#                         nii_path = 'model_sig_nii/select_region_unit', n_img=len(img_data),
#                         device='cuda' if torch.cuda.is_available() else 'cpu')

# def preprocess(coord,img,Y,cov=np.zeros((img_data[0].shape[0],1)),rep=30,a=0.01,b=100,poly_degree=18, N_saves=70,
#                          lamb=10,n_hid=128, n_hid2=16, lr=3e-3,path="/nfs/turbo/jiankanggroup/ellahe/test_pth",nb_layer=2, n_epochs=151,
#                         thred = 0.5, bg_image = "../data/neuroimaging/AAL_MNI_2mm.nii", batch_size=128,
#                         nii_path = 'model_sig_nii/select_region_unit',
#                         device='cuda' if torch.cuda.is_available() else 'cpu'):
    
#     BNN_neuroimg = BNN_model(coord=coord, imgs=img, cov=cov,
#                              Y=Y,rep=rep,a=a,b=b, poly_degree=poly_degree, N_saves=N_saves,
#                              lamb=lamb,n_hid=n_hid, n_hid2=n_hid2, lr=lr,path=path,nb_layer=nb_layer, n_epochs=n_epochs,
#                             thred = thred, bg_image = bg_image, batch_size=batch_size,
#                             nii_path = nii_path, n_img=len(img),
#                             device='cuda' if torch.cuda.is_available() else 'cpu')
    
#     BNN_neuroimg.load_data()

    
# def train():
#     BNN_neuroimg = BNN_model(coord=coord, imgs=img, cov=cov,
#                              Y=Y,rep=rep,a=a,b=b, poly_degree=poly_degree, N_saves=N_saves,
#                              lamb=lamb,n_hid=n_hid, n_hid2=n_hid2, lr=lr,path=path,nb_layer=nb_layer, n_epochs=n_epochs,
#                             thred = thred, bg_image = bg_image, batch_size=batch_size,
#                             nii_path = nii_path, n_img=len(img),
#                             device='cuda' if torch.cuda.is_available() else 'cpu')
#     R2 = BNN_neuroimg.train_model() 


# def cal_post():
#     BNN_neuroimg.beta_post()


# def select_region():
#     BNN_neuroimg.beta_fdr_control()
#     BNN_neuroimg.output_selected()









