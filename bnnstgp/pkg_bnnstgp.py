#!/usr/bin/env python
# coding: utf-8

# In[1]:

# from model import config

## import R packages
# import os
# os.environ['R_HOME'] = '/home/ellahe/.conda/envs/BNNSTGP/lib/R'

# import rpy2.robjects as robjects


# from rpy2.robjects.packages import importr

# utils = importr('utils')


# from rpy2.robjects import r
# # r.options(repos='https://repo.miserver.it.umich.edu/cran/')
# # utils.install_packages('BayesGPfit', verbose = 0,repos = 'https://repo.miserver.it.umich.edu/cran/')
# # GP = importr('BayesGPfit')
# GP = importr('BayesGPfit')


# In[2]:


## basic python import

import numpy as np
import os
import copy
import time
import random
import pandas as pd
# import rpy2.robjects as robjects
import numpy as np
# from rpy2.robjects import pandas2ri
# readRDS = robjects.r['readRDS']
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

# torch.set_default_dtype(torch.float32)
# pandas2ri.activate()


# In[3]:
import sys
import os.path
this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir)


from model.data_split import TrainTestSplit
from model.model_train3 import ModelTrain
from model.pre_data import LoadData
from model.post_pred import PostPred
from model.fdr_control import BetaFdr
from model.region_select import Selection


# In[4]:


class BNN_model(nn.Module):
    """
    The BNN_model allows the creation, training and analysis of a Bayesian Neural Network (BNN) model with sophisticated control parameters for BNN structure, training and fdr control. 
    It includes specific steps for data loading, model training, beta posterior calculation, false discovery rate (FDR) control, and region selection output.
    
    Attributes: 
        coord (numpy array): The value of coordinates.
        imgs (numpy array): The images included in model.
        cov(numpy array): Covariates for the images.
        Y(numpy array): The dependent variables for model.
        rep(int): The number of repetitions to be used in model.
        path(str):The relevant path to save trained models, images and other output files.
        nb_layer(int): The number of layers to use in model.
        thred(float): The threshold to use for selection.
        bg_image1(str): The first base image for output.
        bg_image2(str): The second base image for output.
        region_info1(str): The first region information file name for output.
        region_info2(str): The second region information file name for output.
        nii_path(str): The relevant path to save nii image files.
        lr(float): The learning rate to use in model training. (Default: 1e-3)
        n_img(int): number of image (modality) to input.
        a(float), b(float), poly_degree(int): parameters for eigen decomposition using the BayesGPfit package.
        lamb(float): the threshold parameter for BNN-STGP.
        n_hid(int): number of hidden units for the first layer.
        n_hid2(int): number of hidden units for the second layer.
        n_hid3(int): number of hidden units for the third layer.
        n_hid4(int): number of hidden units for the fourth layer.
        N_saves(int): number of saved weight samples per epochs.
    """
    
    def __init__(self, coord, imgs, cov, Y, rep, path, nb_layer, thred, bg_image1, bg_image2, 
                     region_info1, region_info2, mni_file1, mni_file2, nii_path, fdr_thred=0.01,
                     lr=1e-3, input_dim=784, n_hid = None, n_hid2 = None, n_hid3 = None, 
                     n_hid4 = None, hid_u=None, output_dim = 1, w_dim = 2, n_knots = 66, 
                     N_train=200, phi=None, langevin = True, step_decay_epoch = 100, step_gamma = 0.1, 
                     act = 'relu', b_prior_sig=None, num_layer=None, n_epochs=120, lamb=10, batch_size=128, 
                     N_saves=30, test_every=10, n_img=1, a=0.01, b=20, poly_degree = 15, device='cpu',train_ratio=0.8, fdr_path = "Voxel"):
        """
        Constructs a new 'BNN_model' object.

        :param coord (numpy array): An array of coordinates.
        :param imgs (list): A list of image data.
        :param cov (numpy array): An array of covariates.
        :param Y(numpy array): The dependent variables for model.
        :param rep (int): The number of test repetitions.
        :param path (str): A filesystem path for saving and loading models.
        :param nb_layer (int): The number of layers in the neural network.
        :param thred (float): The threshold for FDR control.
        :param bg_image1 (str): A background image file path.
        :param bg_image2 (str): A secondary background image file path.
        :param region_info1 (str): Information about the AAL region of the first image (T1).
        :param region_info2 (str): Information about the AAL region of the second image (fMRI1).
        :param mni_file1 (str): Information file about the matching of coordinates and the region index for the first image (fMRI).
        :param mni_file2 (str): Information file about the matching of coordinates and the region index for the second image (fMRI).
        :param fdr_thred: threshold value for beta fdr control.
        :param lr (float): The learning rate for training the model (default is 1e-3).
        :param input_dim (int): The dimension of the input vector.
        :param n_hid (int): The number of hidden units in the first hidden layer.
        :param n_hid2 (int): The number of hidden units in the second hidden layer.
        :param n_hid3 (int): The number of hidden units in the third hidden layer.
        :param n_hid4 (int): The number of hidden units in the fourth hidden layer.
        :param output_dim (int): The dimensionality of output. e.g. number of classes in multiclass classification.
        :param nii_path(str): The relevant path to save nii image files.
        :param n_img(int): number of image (modality) to input.
        :param a(float), b(float), poly_degree(int): parameters for eigen decomposition using the BayesGPfit package.
        :param lamb(float): the threshold parameter for BNN-STGP.
        :param N_saves(int): number of saved weight samples per epochs.
        """
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
        """
        Load, preprocess and initialize image, coordinate and scalar data.

        This function loads image, coordinate and scalar data into the class attributes. 
        It preprocesses the data by applying normalization, standardization etc, as defined in the LoadData function.

        Returns:
            This function doesn't return a value but updates class attributes with preprocessed data.
        """
        torch.cuda.empty_cache()
        self.imgs, self.Y, self.W, self.coord, self.phi = LoadData(self.coord, self.cov, self.imgs, self.Y, self.a, self.b, self.poly_degree, self.device).preprocess()
        
        
    def train_model(self, rep=None, path=None, train_ratio=None):
        """
        Train the model based on input parameters and data.

        It uses the ModelTrain module to train the model. 
        The repetition count, saving path, layer count, and other parameters are provided as inputs.

        Returns:
            R2(numpy array): Array of R-square values representing fit of the model.
        """   
        if rep is not None:
            self.rep = rep
        if path is not None:
            self.path = path
        if train_ratio is not None:
            self.train_ratio = train_ratio
            
        R2 = ModelTrain(imgs=self.imgs, Y=self.Y, W=self.W, phi=self.phi, rep=self.rep, 
                                 path=self.path, n_hid=self.n_hid, n_hid2=self.n_hid2, 
                                 n_hid3=self.n_hid3, n_hid4=self.n_hid4, nb_layer=self.nb_layer, lr=self.lr, n_epochs=self.n_epochs, 
                                 lamb=self.lamb, batch_size=self.batch_size, N_saves=self.N_saves, 
                                 test_every=self.test_every, n_img=self.n_img, device=self.device).train()
        
    
    def post_pred(self):
        """
        Compute coverage ratio based on the number of repetitions, path and data.

        This function uses BetaPost module to compute the coverage ratio.

        Returns:
            None.
        """
        PostPred(rep=self.rep, path=self.path, imgs=self.imgs,
                 Y=self.Y, W=self.W, phi=self.phi, n_img=self.n_img,
                 n_hid=self.n_hid, n_hid2=self.n_hid2, 
                 n_hid3=self.n_hid3, n_hid4=self.n_hid4, 
                 lr=self.lr, lamb=self.lamb, batch_size=self.batch_size, nb_layer=self.nb_layer).compute_cov()
        
    def beta_fdr_control(self, rep=None, path=None, train_ratio=None, fdr_path=None):
        """
        Compute and return spatially varying function (beta) values after performing False Discovery Rate (FDR) control over all repetitions.

        The function provides a FDR control and compute beta values using BetaFdr module.

        Returns:
            beta_all(list): The list of adjusted beta values after performing FDR control.
        """
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
        """
        Generate a summary of selected regions based on the stability threshold and save this in a nii image.

        Regions are selected if their corresponding value exceeds a given stability threshold. 
        The summary table of selected regions and saved nii image with selected regions annotated will be saved

        Returns:
            None.
        """
        if fdr_path is not None:
            self.fdr_path = fdr_path
        self.coord = original_coord
        for i in range(self.n_img):
            Selection(self.imgs, self.coord, self.rep, self.path, self.coord, self.thred, self.bg_image1, self.bg_image2, self.region_info1, self.region_info2, self.mni_file1, self.mni_file1, self.nii_path,self.n_img).output_selected_region(i, self.fdr_path)

 
