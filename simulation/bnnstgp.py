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
                     region_info1, region_info2, nii_path, 
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
        :param region_info1 (str): Information about the first region of interest.
        :param region_info2 (str): Information about the second region of interest.
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
        
    
    def beta_post(self):
        """
        Compute coverage ratio based on the number of repetitions, path and data.

        This function uses BetaPost module to compute the coverage ratio.

        Returns:
            None.
        """
        BetaPost(rep=self.rep, path=self.path, imgs=self.imgs,
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
                                    n_hid3=self.n_hid3, n_hid4=self.n_hid4,  
                                    coord=self.coord, imgs=self.imgs, 
                                    W=self.W, hid_u=self.hid_u,
                                    phi=self.phi, nb_layer=self.nb_layer, 
                                    n_img=self.n_img, train_ratio=self.train_ratio).beta_fdr(i,self.fdr_path))
    
    def output_selected(self, fdr_path=None):
        """
        Generate a summary of selected regions based on the stability threshold and save this in a nii image.

        Regions are selected if their corresponding value exceeds a given stability threshold. 
        The summary table of selected regions and saved nii image with selected regions annotated will be saved

        Returns:
            None.
        """
        if fdr_path is not None:
            self.fdr_path = fdr_path    
        for i in range(self.n_img):
            Selection(self.imgs, self.coord, self.rep, self.path, self.coord[i], self.thred, self.bg_image1, self.bg_image2, self.region_info1, self.region_info2, self.nii_path,self.n_img).output_selected_region(i, self.fdr_path)
            
            
# hf = h5py.File("simulation_data/img_coord1.hdf5", 'r')
# img1 = hf.get('image')['img1'][()][:,:]
# img2 = hf.get('image')['img2'][()][:,:]
# coord1 = hf.get('coord')['coord1'][()][:,:]
# coord2 = hf.get('coord')['coord2'][()][:,:]
# Y = hf.get('coord')['Y'][()][:]
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
    

def simulate_data(r, random_seed=2023):
    random.seed(random_seed)
    np.random.seed(int(random_seed))
    
    v_list1 = GP.GP_generate_grids(d = 2, num_grids = 224,grids_lim=np.array([-3,3]))
    v_list2 = GP.GP_generate_grids(d = 2, num_grids = 224,grids_lim=np.array([-1,1]))
    true_beta1 = (0.5*v_list1[:,0]**2+v_list1[:,1]**2)<2
    true_beta2 = np.exp(-5*(v_list2[:,0]-1.5*np.sin(math.pi*np.abs(v_list2[:,1]))+1.0)**2)
    p1 = v_list1.shape[0]
    p2 = v_list2.shape[0]
    m = 2000

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
    
    # plt.imshow(true_beta1.reshape(30,30))
    # plt.show
    # plt.imshow(true_beta2.reshape(20,20))
    # plt.show()
 

    return v_list, img, Y

r = 0.9
coord, img_data, Y = simulate_data(r)

rep_num = 10
a = 0.01
b = 100 
poly_degree = 18
num_weight_samples = 30
lamb = 3
n_hid = 128
n_hid2 = 16
lr = 3e-3
model_dir = "model"
model_save_path = "simulation_dataR9/"+model_dir
os.makedirs(model_save_path, exist_ok=True)
num_layer = 1
num_epochs = 81
thred = 0.4
back_image1 = "../data/neuroimaging/AAL_MNI_2mm.nii"
back_image2 = "../data/neuroimaging/AAL_90_3mm.nii"
regioninfo_file1 = "../data/neuroimaging/AALregion_full.xls"
regioninfo_file2 = "../data/neuroimaging/AAL_region_functional_networks.csv"
batch_size = 128
nii_save_path = 'select_region'

BNN_neuroimg = BNN_model(coord=coord, imgs=img_data, cov=np.zeros((img_data[0].shape[0],1)),
                         Y=Y,rep=rep_num,a=a,b=b, poly_degree=poly_degree, N_saves=num_weight_samples,
                         lamb=lamb,n_hid=n_hid, n_hid2=n_hid2, lr=lr,path=model_save_path,nb_layer=num_layer, n_epochs=num_epochs,
                        thred = thred, bg_image1 = back_image1, bg_image2 = back_image2, region_info1 = regioninfo_file1, 
                         region_info2 = regioninfo_file2, batch_size=batch_size,
                        nii_path = nii_save_path, n_img=len(img_data),
                        device='cuda' if torch.cuda.is_available() else 'cpu')

BNN_neuroimg.load_data()
R2 = BNN_neuroimg.train_model() 
# BNN_neuroimg.beta_post()


# different split
train_ratio = 0.8
rep = 20
fdr_path = "/scratch/jiankang_root/jiankang1/ellahe/"+model_dir
model_save_path = "/scratch/jiankang_root/jiankang1/ellahe/"+model_dir

BNN_neuroimg.beta_fdr_control(rep=rep,path=model_save_path,train_ratio=train_ratio,fdr_path=fdr_path)


# total data
train_ratio = 1
rep = 1
model_dir2 = "multi_test_all2"
model_save_path = "/scratch/jiankang_root/jiankang1/ellahe/"+model_dir2
# os.makedir(model_save_path)

# BNN_neuroimg = BNN_model(coord=coord, imgs=img_data, cov=np.zeros((img_data[0].shape[0],1)),
#                          Y=Y,rep=rep_num,a=a,b=b, poly_degree=poly_degree, N_saves=num_weight_samples,
#                          lamb=lamb,n_hid=n_hid, n_hid2=n_hid2, lr=lr,path=model_save_path,nb_layer=num_layer, n_epochs=num_epochs,
#                         thred = thred, bg_image1 = back_image1, bg_image2 = back_image2,
#                          region_info1 = regioninfo_file1, 
#                          region_info2 = regioninfo_file2, batch_size=batch_size,
#                         nii_path = nii_save_path, n_img=len(img_data), train_ratio=train_ratio,
#                         device='cuda' if torch.cuda.is_available() else 'cpu')

# BNN_neuroimg.load_data()
# BNN_neuroimg.train_model() 

BNN_neuroimg.beta_fdr_control(rep=rep,path=model_save_path,train_ratio=train_ratio)


# BNN_neuroimg.output_selected()