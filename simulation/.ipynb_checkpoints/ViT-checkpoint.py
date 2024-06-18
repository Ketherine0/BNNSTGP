import numpy as np
import os
import matplotlib.pyplot as plt
import copy
import time
import random
import pandas as pd
# import rpy2.robjects as robjects
import numpy as np

import h5py

import os
os.environ['R_HOME'] = '/home/ellahe/.conda/envs/bnnstgp/lib/R'

from pkg_bnnstgp2.pkg_bnnstgp import BNN_model
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
readRDS = robjects.r['readRDS']

from rpy2.robjects.packages import importr

utils = importr('utils')



# options(repos=c('https://repo.miserver.it.umich.edu/cran/'))
# utils.install_packages('BayesGPfit', verbose = 0,repos = 'https://repo.miserver.it.umich.edu/cran/')
GP = importr('BayesGPfit')

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
    np.random.seed(random_seed)
    
    v_list1 = GP.GP_generate_grids(d = 2, num_grids = 224,grids_lim=np.array([-3,3]))
    v_list2 = GP.GP_generate_grids(d = 2, num_grids = 224,grids_lim=np.array([-1,1]))
    true_beta1 = (0.5*v_list1[:,0]**2+v_list1[:,1]**2)<2
    true_beta2 = np.exp(-5*(v_list2[:,0]-1.5*np.sin(math.pi*np.abs(v_list2[:,1]))+1.0)**2)
    p1 = v_list1.shape[0]
    p2 = v_list2.shape[0]
    m = 1000

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
    
r = 0.5
coord, img_data, Y = simulate_data(r)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import Compose, Lambda, Normalize
from torch.utils.data import Dataset, DataLoader
from transformers import ViTConfig, ViTModel
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import Compose, Lambda, Normalize, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip
from torch.utils.data import Dataset, DataLoader
from transformers import ViTConfig, ViTModel
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageDataset(Dataset):
    def __init__(self, img_data, labels, transform=None):
        self.img_data = img_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img = self.img_data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# class RegressionViT(nn.Module):
#     def __init__(self, num_classes=1, dropout_prob=0.5, image_size=448, num_channels=1):
#         super(RegressionViT, self).__init__()

#         # self.config = ViTConfig(
#         #     image_size=image_size,
#         #     num_channels=num_channels,
#         #     patch_size=16,
#         #     hidden_size=512,  # Reduced hidden size
#         #     num_hidden_layers=6,  # Reduced number of hidden layers
#         #     num_attention_heads=8,  # Reduced number of attention heads
#         #     intermediate_size=2048,  # Reduced intermediate size
#         # )
#         self.config = ViTConfig(
#             # Original settings for context
#             hidden_size=512,
#             num_hidden_layers=4,  # Reduced from 6
#             num_attention_heads=8,  # Consider reducing if you still face overfitting
#             intermediate_size=1024,  # Reduced from 2048
#         )

#         self.regressor = nn.Sequential(
#             nn.Linear(self.config.hidden_size, 128),  # Simplified structure
#             nn.ReLU(),
#             nn.Dropout(dropout_prob),
#             nn.Linear(128, num_classes),
#         )

#         self.vit = ViTModel(self.config)

#         self.preprocess = nn.Sequential(
#             nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, self.config.num_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.config.num_channels),
#             nn.ReLU(),
#         )

#         # self.regressor = nn.Sequential(
#         #     nn.Linear(self.config.hidden_size, 256),  # Reduced linear layer size
#         #     nn.BatchNorm1d(256),
#         #     nn.ReLU(),
#         #     nn.Dropout(dropout_prob),
#         #     nn.Linear(256, 128),  # Reduced linear layer size
#         #     nn.BatchNorm1d(128),
#         #     nn.ReLU(),
#         #     nn.Dropout(dropout_prob),
#         #     nn.Linear(128, num_classes),
#         # )

#     def forward(self, x):
#         x = self.preprocess(x)
#         outputs = self.vit(pixel_values=x)
#         x = outputs.last_hidden_state[:, 0]
#         x = self.regressor(x)
#         return x

class RegressionViT(nn.Module):
    def __init__(self, num_classes=1, dropout_prob=0.5, image_size=448, num_channels=3):  # Adjust image_size to 448, assume RGB images
        super(RegressionViT, self).__init__()

        self.config = ViTConfig(
            image_size=image_size,  # Updated image size to 448
            num_channels=num_channels,  # Assuming RGB images
            patch_size=32,  # Keeping the patch size as 16x16
            hidden_size=512,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=2048,
        )

        self.vit = ViTModel(self.config)

        self.preprocess = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.config.num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.config.num_channels),
            nn.ReLU(),
        )

        # Adjust the size of the input layer of the regressor to match the output dimension of the ViT.
        # This requires calculating the number of output features from ViT, which depends on its configuration.
        # Assuming the dimensionality does not change for simplicity, but this may need to be adjusted based on the ViTModel implementation.
        
        # Define the new DNN architecture for the regressor
        self.fc1 = nn.Linear(self.config.hidden_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)  # Output layer adjusted for num_classes
        self.dropout = nn.Dropout(dropout_prob)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.preprocess(x)
        outputs = self.vit(pixel_values=x)
        x = outputs.last_hidden_state[:, 0]  # Use the representation of the [CLS] token

        # Pass through the new DNN
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)  # No activation for the final layer in regression

        return x

    
def train_test_model(train_dataset, test_dataset, random_seed=10, num_epochs=500):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = RegressionViT(num_classes=1, dropout_prob=0.1, image_size=448, num_channels=1)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-4)  # Increased weight_decay for regularization
    # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_test_r2 = -np.inf
    patience = 10
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_predictions, train_actuals = [], []
        
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} - Training")
        for volumes, targets in train_progress_bar:
            volumes = volumes.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(volumes)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_predictions.extend(outputs.detach().cpu().numpy().flatten())
            train_actuals.extend(targets.cpu().numpy().flatten())
            train_progress_bar.set_postfix(loss=running_loss/len(train_progress_bar))
        
        train_r2 = r2_score(train_actuals, train_predictions)
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Train R^2: {train_r2}")

        model.eval()
        test_predictions, test_actuals = [], []
        
        test_progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1} - Testing")
        with torch.no_grad():
            for volumes, targets in test_progress_bar:
                volumes = volumes.to(device, dtype=torch.float)
                targets = targets.to(device, dtype=torch.float)
                outputs = model(volumes)
                test_predictions.extend(outputs.detach().cpu().numpy().flatten())
                test_actuals.extend(targets.cpu().numpy().flatten())

        test_r2 = r2_score(test_actuals, test_predictions)
        print(f"Epoch {epoch+1}, Test R^2: {test_r2}")

        # if test_r2 > best_test_r2:
        #     best_test_r2 = test_r2
        #     counter = 0
        # else:
        #     counter += 1
        #     if counter >= patience:
        #         print(f"Early stopping at epoch {epoch+1}")
        #         break

        # scheduler.step()

# Data preparation code remains the same
img_data_0_tensor = torch.tensor(img_data[0], dtype=torch.float32)
img_data_1_tensor = torch.tensor(img_data[1], dtype=torch.float32)
img_data_0_tensor = img_data_0_tensor.reshape(-1, 224, 224)[:, np.newaxis, :, :] 
img_data_1_tensor = img_data_1_tensor.reshape(-1, 224, 224)[:, np.newaxis, :, :] 
concatenated_data = torch.cat((img_data_0_tensor, img_data_1_tensor), dim=3)

resized_data = F.interpolate(concatenated_data, size=(448, 448))
final_data = resized_data

labels = torch.tensor(Y).view(-1).reshape(-1,1).float()

# Define data augmentation transforms
# transform = Compose([
#     RandomRotation(degrees=10),
#     RandomHorizontalFlip(),
#     RandomVerticalFlip(),
# ])

# dataset = ImageDataset(final_data, labels, transform=transform)
dataset = ImageDataset(final_data, labels)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

config = ViTConfig(
    image_size=448,
    num_channels=1,
    patch_size=32,
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=2048,
)

vit_model = ViTModel(config)

train_test_model(train_dataset, test_dataset, random_seed=10, num_epochs=500)