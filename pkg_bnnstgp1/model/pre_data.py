import pandas as pd
import os
import torch
os.environ['R_HOME'] = '/home/ellahe/.conda/envs/bnnstgp/lib/R'

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import numpy as np
from rpy2.robjects import pandas2ri
readRDS = robjects.r['readRDS']

torch.set_default_dtype(torch.float32)
pandas2ri.activate()

utils = importr('utils')
GP = importr('BayesGPfit')

class LoadData:
    """
    1) Input the scalar data, the image data, and the coordinate data
    2) Function returns x, y, covariate, and the eigen decomposition of coordinate
    """
        
    def __init__(self, coord, cov, img, Y, a, b, poly_d, device):
        self.coord = coord
        self.W = pd.DataFrame(cov)
        self.img = img
        self.Y = Y
        self.a = a
        self.b = b
        self.poly_degree = poly_d
        self.device = device
        
    def preprocess(self):
        # cov = pd.DataFrame(self.W)
        n_img = len(self.img)
        intercept = pd.DataFrame(np.ones((self.W.shape[0])))
        cov = pd.concat([intercept,self.W],axis=1)
        cov2 = cov.dropna()
        # cov2 = cov2.drop(columns = ['site_num'])
        idx = cov.isnull().any(axis=1).to_numpy()
        
        cov_final = pd.get_dummies(cov2, drop_first = True)
        self.W = cov_final.to_numpy()
        
        
        img_data = []
        coord_data = []
        phi = []
        for i in range(n_img):
            img1 = self.img[i]
            img1 = img1[np.invert(idx)]
            img1 -= np.mean(img1)
            img1 /= np.max(np.abs(img1))
            img1 = torch.tensor(img1).float().to(self.device)
            img_data.append(img1)
            
            coord1 = self.coord[i]
            if coord1.shape[0] != img1.shape[1]:
                raise Exception("The image data and the coordinate data are not matched.")
            else:
                for j in range(coord1.shape[1]):
                    coord1[:,j] = coord1[:,j] - np.mean(coord1[:,j])
                    coord1[:,j] = coord1[:,j] / np.max(np.abs(coord1[:,j]))
                coord_data.append(torch.tensor(coord1).float().to(self.device))
                # coord_data.append(coord1)
            
            phi1 = GP.GP_eigen_funcs_fast(coord1, a=self.a, b=self.b, poly_degree = self.poly_degree).astype(np.float32)
            phi1 = torch.tensor(phi1).float().to(self.device)
          
            phi.append(torch.tensor(phi1))
        self.Y = torch.tensor(self.Y).float().to(self.device)
        self.W = torch.tensor(self.W).float().to(self.device)
        # print(self.W.shape)

        return img_data, self.Y, self.W, coord_data, phi
