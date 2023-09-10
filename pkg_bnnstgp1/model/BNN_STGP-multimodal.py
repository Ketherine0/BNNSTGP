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


get_ipython().run_line_magic('matplotlib', 'inline')

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

if torch.cuda.is_available():
    print('using GPU')
else:
    print('using CPU')

torch.set_default_dtype(torch.float32)
pandas2ri.activate()


# In[3]:


from model.neuroimg_network import NeuroNet
from model.data_split import TrainTestSplit
from model.model_train import ModelTrain
from model.pre_data import LoadData
from model.beta_posterior import BetaPost
from model.fdr_control import BetaFdr
from model.region_select import Selection


# In[4]:


class BNN_model(nn.Module):
    
    def __init__(self, coord, imgs, cov, Y, rep, path, nb_layer, thred, bg_image, nii_path, 
                     lr=1e-3, input_dim=784, n_hid = 32, n_hid2 = 128, n_hid3 = 8, 
                     n_hid4 = 32, output_dim = 1, w_dim = 1, n_knots = 66, 
                     N_train=200, phi=None, langevin = True, step_decay_epoch = 100, step_gamma = 0.1, 
                     act = 'relu', b_prior_sig=None, num_layer=None, n_epochs=120, lamb=10, batch_size=128, 
                     N_saves=30, test_every=10, n_img=1, a=0.01, b=20, poly_degree = 15, device='cpu'):
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
    
    def load_data(self):
        torch.cuda.empty_cache()
        self.imgs, self.Y, self.W, self.coord, self.phi = LoadData(self.coord, self.cov, self.imgs, self.Y, self.a, self.b, self.poly_degree, self.device).preprocess()
        
    def train_model(self):
        R2 = ModelTrain(imgs=self.imgs, Y=self.Y, W=self.W, phi=self.phi, rep=self.rep, 
                                 path=self.path, n_hid=self.n_hid, n_hid2=self.n_hid2, 
                                 n_hid3=self.n_hid3, n_hid4=self.n_hid4, nb_layer=self.nb_layer, lr=self.lr, n_epochs=self.n_epochs, 
                                 lamb=self.lamb, batch_size=self.batch_size, N_saves=self.N_saves, 
                                 test_every=self.test_every, n_img=self.n_img, device=self.device).train()
        # return BNN_net, R2
    
    def beta_post(self):
        BetaPost(rep=self.rep, path=self.path, imgs=self.imgs,
                 Y=self.Y, W=self.W, phi=self.phi, n_img=self.n_img,
                 lr=self.lr, lamb=self.lamb, batch_size=self.batch_size, nb_layer=self.nb_layer, n_hid=self.n_hid).compute_cov()
        
    def beta_fdr_control(self):
#         self.beta_all1, self.beta_all2 = BetaFdr(self.rep, self.path, self.lr, self.lamb, 
#                                                         self.n_hid, self.Y, coord=self.coord, imgs=self.imgs, 
#                                                         W=self.W,
#                                                         phi=self.phi, nb_layer=self.nb_layer, n_img=self.n_img).beta_fdr()
        self.beta_all = []
        for i in range(self.n_img):
            self.beta_all.append(BetaFdr(self.rep, self.path, self.lr, self.lamb, 
                                    self.n_hid, self.Y, coord=self.coord, imgs=self.imgs, 
                                    W=self.W,
                                    phi=self.phi, nb_layer=self.nb_layer, n_img=self.n_img).beta_fdr(i))
    
    def output_selected_region(self):
        for i in range(self.n_img):
            beta_matched = Selection(self.rep, self.n_hid, self.path, self.coord[i], self.beta_all[i], self.thred, self.bg_image, self.nii_path).unit_match()
            Selection(self.rep, self.n_hid, self.path, self.coord[i], self.beta_all[i], self.thred, self.bg_image, self.nii_path).output_selected_region(beta_matched)
#         beta_matched2 = Selection(self.rep, self.n_hid, self.path, self.coord2, self.beta_all2, self.thred, self.bg_image, self.nii_path).unit_match()
#         Selection(self.rep, self.n_hid, self.path, self.coord2, self.beta_all2, self.thred, self.bg_image, self.nii_path).output_selected_region(beta_matched2)
        


# In[5]:


# def compute_bases(v_list,poly_degree=30,a=0.001,b=20):
#     Psi = GP.GP_eigen_funcs_fast(v_list, poly_degree = poly_degree, a = a, b = b)
#     lam = GP.GP_eigen_value(poly_degree = poly_degree, a = a, b = b, d = np.array(v_list).shape[1])
#     lam=list(lam)
#     sqrt_lambda = list(np.sqrt(lam))
#     Psi2 = np.transpose(np.array(Psi))
#     Bases = np.zeros((Psi2.shape[0],Psi2.shape[1]))
#     for i in range(len(sqrt_lambda)):
#         Bases[i,:] = Psi2[i,:]*sqrt_lambda[i]
    
#     return Bases
    

# def simulate_data():
#     random.seed(2023)
#     v_list1 = GP.GP_generate_grids(d = 3, num_grids = 20,grids_lim=np.array([-1,1]))
#     v_list2 = GP.GP_generate_grids(d = 2, num_grids = 20,grids_lim=np.array([-1,1]))
#     true_beta1 = v_list1[:,0]**2+v_list1[:,1]**2+v_list1[:,2]**2<0.9
#     true_beta2 = np.exp(-5*(v_list2[:,0]-1.5*np.sin(math.pi*np.abs(v_list2[:,1]))+1.0)**2)
#     p1 = v_list1.shape[0]
#     p2 = v_list2.shape[0]
#     m = 3000

#     Bases1 = compute_bases(v_list1)
#     Bases2 = compute_bases(v_list2)
#     theta1 = np.random.normal(size=m*Bases1.shape[0],scale=1/np.sqrt(p1))
#     theta1 = theta1.reshape(Bases1.shape[0],m)
#     theta2 = np.random.normal(size=m*Bases2.shape[0],scale=1/np.sqrt(p2))
#     theta2 = theta2.reshape(Bases2.shape[0],m)
#     # simulate an image
#     img1 = np.transpose(np.dot(np.transpose(Bases1),theta1))
#     img2 = np.transpose(np.dot(np.transpose(Bases2),theta2))
#     v_list1 = np.array(v_list1)
#     v_list2 = np.array(v_list2)
    
#     R2 = 0.9
#     theta0 =  2
#     mean_Y = theta0 + np.dot(img1,true_beta1) + np.dot(img2,true_beta2)
#     true_sigma2 = np.var(mean_Y)*(1-R2)/R2
#     Y = mean_Y + np.random.normal(size=m,scale=np.sqrt(true_sigma2))
#     v_list = [v_list1,v_list2]
#     img = [img1, img2]

#     return v_list, img, Y
    
# coord, img_data, Y = simulate_data()


# In[6]:


Y = pd.read_csv("y.csv").iloc[:,1].values


# In[7]:


hf = h5py.File('image.hdf5', 'r')
img1 = hf.get('img')['img1'][()]
img2 = hf.get('img')['img2'][()]

h2 = h5py.File('coord.hdf5', 'r')
coord1 = h2.get('coord')['coord1'][()]
coord2 = h2.get('coord')['coord2'][()]

coord = [coord1, coord2]
img_data = [img1, img2]


# In[8]:


# BNN_neuroimg = BNN_model(coord=coord, imgs=img_data, cov=np.zeros((img_data[0].shape[0],1)),
#                          Y=Y,rep=50,a=0.01,b=100, poly_degree=18,
#                          lamb=10,n_hid=16,lr=3e-3,path="multi_test",nb_layer=1, n_epochs=70,
#                         thred = 0.6, bg_image = "../data/neuroimaging/AAL_MNI_2mm.nii", 
#                         nii_path = 'model_sig_nii/select_region_unit', n_img=len(img_data))
BNN_neuroimg = BNN_model(coord=coord, imgs=img_data, cov=np.zeros((img_data[0].shape[0],1)),
                         Y=Y,rep=50,a=0.01,b=100, poly_degree=18, N_saves=20,
                         lamb=10,n_hid=128, n_hid2=16, lr=3e-3,path="multi_test",nb_layer=1, n_epochs=70,
                        thred = 0.6, bg_image = "../data/neuroimaging/AAL_MNI_2mm.nii", batch_size=32,
                        nii_path = 'model_sig_nii/select_region_unit', n_img=len(img_data),
                        device='cuda' if torch.cuda.is_available() else 'cpu')


# In[9]:


BNN_neuroimg.load_data()


# In[10]:


# import multiprocessing

# # Define the function to run in parallel
# def my_function(arg):
#     # some code here
#     return result

# if __name__ == '__main__':
#     # Set up the pool of workers
#     pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

#     # Submit tasks to the pool
#     results = []
#     for arg in my_args:
#         result = pool.apply_async(my_function, args=(arg,))
#         results.append(result)

#     # Get the results
#     output = [r.get() for r in results]


# In[11]:


# a=0.01,b=100, poly_degree=18, lamb=10, n_hid=128, n_hid2=16
R2 = BNN_neuroimg.train_model() 


# In[ ]:


# a=0.01,b=100, poly_degree=18, lamb=10,n_hid=16
# R2 = BNN_neuroimg.train_model() 


# In[ ]:


BNN_neuroimg.beta_post()


# In[9]:


BNN_neuroimg.beta_fdr_control()


# In[11]:


BNN_neuroimg.output_selected_region()


# In[ ]:




