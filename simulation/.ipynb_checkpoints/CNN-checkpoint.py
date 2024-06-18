## import R packages
import os
os.environ['R_HOME'] = '/home/ellahe/.conda/envs/bnnstgp/lib/R'

import rpy2.robjects as robjects

from rpy2.robjects.packages import importr


utils = importr('utils')

# options(repos=c('https://repo.miserver.it.umich.edu/cran/'))
# utils.install_packages('BayesGPfit', verbose = 0,repos = 'https://repo.miserver.it.umich.edu/cran/')
GP = importr('BayesGPfit')

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
import sys

import warnings
warnings.filterwarnings("ignore")

print("Start")
if torch.cuda.is_available():
    print('using GPU')
else:
    print('using CPU')

torch.set_default_dtype(torch.float32)
pandas2ri.activate()

current_dir = os.getcwd()
sub_dir = os.path.join(current_dir, 'pkg_bnnstgp2') 
sys.path.append(sub_dir)

from model.neuroimg_network import NeuroNet
from model.data_split import TrainTestSplit
from model.model_train_all import ModelTrain
from model.pre_data import LoadData
from model.post_pred import PostPred
from model.fdr_control_simulation import BetaFdr
from model.region_select import Selection


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
import matplotlib.pyplot as plt
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
                     n_hid4 = None, hid_u=None, output_dim = 1, w_dim = 1, n_knots = 66, 
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
                                 test_every=self.test_every, n_img=self.n_img, device=self.device, train_ratio=self.train_ratio).train()
        
    
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
                                    n_hid3=self.n_hid3, n_hid4=self.n_hid4, fdr_thred=self.thred,
                                    coord=self.coord, imgs=self.imgs, 
                                    W=self.cov, hid_u=self.hid_u,
                                    phi=self.phi, nb_layer=self.nb_layer, 
                                    n_img=self.n_img,    
                                    train_ratio=self.train_ratio).beta_fdr(i,self.fdr_path))
    
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

 

def compute_bases(v_list,poly_degree=30,a=0.001,b=20):
    Psi = GP.GP_eigen_funcs_fast(v_list, poly_degree = poly_degree, a = a, b = b)
    lam = GP.GP_eigen_value(poly_degree = poly_degree, a = a, b = b, d = np.array(v_list).shape[1])
    lam=list(lam)
    sqrt_lambda = list(np.sqrt(lam))
    Psi2 = np.transpose(np.array(Psi))
    Bases = np.zeros((Psi2.shape[0],Psi2.shape[1]))
    for i in range(len(sqrt_lambda)):
        Bases[i,:] = Psi2[i,:]*sqrt_lambda[i]
    
    return Bases
    

def simulate_data(n, r, random_seed=2023):
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    v_list1 = GP.GP_generate_grids(d = 2, num_grids = 224,grids_lim=np.array([-3,3]))
    v_list2 = GP.GP_generate_grids(d = 2, num_grids = 224,grids_lim=np.array([-1,1]))
    true_beta1 = (0.5*v_list1[:,0]**2+v_list1[:,1]**2)<2
    true_beta2 = np.exp(-5*(v_list2[:,0]-1.5*np.sin(math.pi*np.abs(v_list2[:,1]))+1.0)**2)
    p1 = v_list1.shape[0]
    p2 = v_list2.shape[0]
    m = n

    Bases1 = compute_bases(v_list1)
    Bases2 = compute_bases(v_list2)
    theta1 = np.random.normal(size=m*Bases1.shape[0],scale=1/np.sqrt(p1))
    theta1 = theta1.reshape(Bases1.shape[0],m)
    theta2 = np.random.normal(size=m*Bases2.shape[0],scale=1/np.sqrt(p2))
    theta2 = theta2.reshape(Bases2.shape[0],m)
    # simulate an image
    img1 = np.transpose(np.dot(np.transpose(Bases1),theta1))
    img2 = np.transpose(np.dot(np.transpose(Bases2),theta2))
    v_list1 = np.array(v_list1)
    v_list2 = np.array(v_list2)
    
    R2 = r
    # variance of sigma^2
    theta0 =  2
    mean_Y = theta0 + np.dot(img1,true_beta1) + np.dot(img2,true_beta2)
    # mean_Y = theta0 + np.dot(img2,true_beta2)
    print(np.var(np.dot(img1,true_beta1)))
    print(np.var(np.dot(img2,true_beta2)))
    true_sigma2 = np.var(mean_Y)*(1-R2)/R2
    Y = mean_Y + np.random.normal(size=m,scale=np.sqrt(true_sigma2))
    v_list = [v_list1,v_list2]
    img = [img1, img2]
    

    return v_list, img, Y

r = 0.9
n = 1000
coord, img_data, Y = simulate_data(n,r)

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import r2_score

# Assuming img_data[0] and img_data[1] are numpy arrays with shape (1000, 10000)
# and you want to concatenate along the height, then reshape each to 100x100 and concatenate.

img_data_0_reshaped = img_data[0].reshape(n, 224,224)
img_data_1_reshaped = img_data[1].reshape(n, 224,224)

# Concatenate along the height dimension to get a shape of (1000, 200, 100)
combined_stacked_volumes = np.concatenate([img_data_0_reshaped, img_data_1_reshaped], axis=1)

# Ensure the shape is [N, C, H, W] for the model
combined_stacked_volumes = combined_stacked_volumes[:, np.newaxis, :, :]  # Adds the channel dimension

# Convert to PyTorch tensor
combined_stacked_volumes_tensor = torch.tensor(combined_stacked_volumes, dtype=torch.float32)

# Proceed with your model training/testing

# Convert to PyTorch tensor
labels = torch.tensor(Y).view(-1).reshape(-1,1)
labels = labels.float()
labels_tensor = torch.tensor(labels, dtype=torch.float32)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import torch.optim as optim

class FMRI2DCNN(nn.Module):
    def __init__(self, dropout_prob=0.25):
        super(FMRI2DCNN, self).__init__()
        
        # Adjusted convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # New layer
        self.bn4 = nn.BatchNorm2d(128)
        
        # Pooling applied more selectively to preserve spatial information
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.num_flat_features = None  # Dynamically calculated
        
        # Placeholder for the first fully connected layer, will be updated in forward pass
        self.fc1 = nn.Linear(1, 512)  # This will be dynamically updated
        self.fc_bn1 = nn.BatchNorm1d(512)  # Batch normalization for the FC layer
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(512, 256)
        self.fc_bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))  # New convolutional layer
        x = self.pool(x)  # Additional pooling step if necessary
        
        # Dynamically adjust fc1's in_features based on the flattened size
        if self.num_flat_features is None:
            with torch.no_grad():
                self.num_flat_features = x.view(x.size(0), -1).shape[1]
                self.fc1 = nn.Linear(self.num_flat_features, 512).to(x.device)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.fc_bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x



def train_test_model(combined_stacked_volumes, targets, random_seed=10):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    volumes_train, volumes_test, targets_train, targets_test = train_test_split(
        combined_stacked_volumes, targets, test_size=0.2, random_state=random_seed)

    train_dataset = TensorDataset(volumes_train, targets_train)
    test_dataset = TensorDataset(volumes_test, targets_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = FMRI2DCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_predictions, train_actuals = [], []
        for volumes, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(volumes)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_predictions.extend(outputs.detach().numpy().flatten())
            train_actuals.extend(targets.numpy().flatten())
        if (epoch + 1) % 5 == 0:
            train_r2 = np.corrcoef(train_actuals, train_predictions)[0,1]**2
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Train R^2: {train_r2}")

        if (epoch + 1) % 10 == 0:
            model.eval()
            test_predictions, test_actuals = [], []
            with torch.no_grad():
                for volumes, targets in test_loader:
                    outputs = model(volumes)
                    test_predictions.extend(outputs.detach().numpy().flatten())
                    test_actuals.extend(targets.numpy().flatten())

            test_r2 = r2_score(test_actuals, test_predictions)
            print(test_r2)
            test_r2 = np.corrcoef(np.array(test_actuals).reshape(-1), np.array(test_predictions).reshape(-1))[0,1]**2
            print(f"Epoch {epoch+1}, Test R^2: {test_r2}")

    # Clean up variables to free memory
    del volumes_train, volumes_test, targets_train, targets_test
    del train_dataset, test_dataset, train_loader, test_loader
    del model, criterion, optimizer
    
train_test_model(combined_stacked_volumes_tensor, labels_tensor.reshape(-1,1), random_seed=10)

