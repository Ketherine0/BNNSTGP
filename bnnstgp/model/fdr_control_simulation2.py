import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
import random
import pandas as pd
import h5py
import os
import gc

from model.neuroimg_network import NeuroNet

class BetaFdr:
    def __init__(self, rep, path, lr, lamb, Y,
                 n_hid, n_hid2, n_hid3, n_hid4, fdr_thred, train_ratio=0.8, hid_u=None, coord=None, imgs=None, W=None,  
                 phi=None, nb_layer=1, n_img=1):
        self.rep = rep
        self.path = path
        self.lr = lr
        self.lamb = lamb
        self.coord = coord
        self.imgs = imgs
        self.Y = Y
        self.W = W
        self.n_hid = n_hid
        self.n_hid2 = n_hid2
        self.n_hid3 = n_hid3
        self.n_hid4 = n_hid4
        self.phi = phi
        self.nb_layer = nb_layer
        self.n_img = n_img
        self.hid_u = hid_u
        self.train_ratio = train_ratio
        self.fdr_thred = fdr_thred
        
    def calculate_beta_fdr(self, network, weight_samples, network_phi, network_b, coord):
        beta = np.zeros([len(weight_samples), coord.shape[0], self.n_hid])
        for idx, weight in enumerate(weight_samples):
            network.model.load_state_dict(weight)
            output = torch.mm(network_phi, network_b)
            output = F.threshold(output, network.model.lamb, network.model.lamb) - F.threshold(-output, network.model.lamb, network.model.lamb)
            output = network.model.sigma * output
            output = output.cpu().detach().numpy()
            beta[idx,:,:] = output
        beta = np.mean(beta, axis=0)
        return beta

    def beta_fdr(self, idx, fdr_path):
        all_beta = np.zeros((self.rep, self.coord[idx].shape[0], self.n_hid))

        for repetition in range(self.rep):
            print("Reptition: ", repetition)
            np.random.seed(repetition)
            torch.manual_seed(repetition)
            train_idx = np.random.choice(len(self.Y), int(self.train_ratio * len(self.Y)), replace=False)
            reg = LinearRegression().fit(np.ones((len(train_idx), 1)), self.Y[train_idx].detach().numpy())
            net = self._init_nn_model(reg, idx, train_idx, [self.phi[i].shape[1] for i in range(self.n_img)])
            weight_samples = self._load_saved_samples(repetition)
            beta_all_repetition = self.calculate_beta_fdr(net, weight_samples, net.model.phi[idx], net.model.b[idx], self.coord[idx])
            all_beta[repetition, :, :] = beta_all_repetition

        # Assume some operation to compute 'fdr_controlled_beta' from 'all_beta'
        fdr_controlled_beta = np.mean(all_beta, axis=0)  # Simplified example, adjust as needed

        # Save the FDR controlled beta values to the specified path
        dir_path = fdr_path
        os.makedirs(dir_path, exist_ok=True)
        with h5py.File(f"{dir_path}/Beta_FDR_{idx}.hdf5", 'w') as hf:
            hf.create_dataset("FDR_Controlled_Beta", data=fdr_controlled_beta)
        
        # Print feature selection statistics
        num_selected = np.sum(fdr_controlled_beta != 0)
        num_not_selected = np.sum(fdr_controlled_beta == 0)
        total_features = fdr_controlled_beta.size
        proportion_selected = num_selected / total_features
        proportion_not_selected = num_not_selected / total_features

        print(f"Number of Selected Features: {num_selected}")
        print(f"Number of Not Selected Features: {num_not_selected}")
        print(f"Proportion of Selected Features: {proportion_selected:.2f}")
        print(f"Proportion of Not Selected Features: {proportion_not_selected:.2f}")

        return fdr_controlled_beta


    def _init_nn_model(self, reg, idx, train_idx, n_knots):
        return NeuroNet(
            reg_init=np.append(reg.coef_, reg.intercept_).astype(np.float32),
            lr=self.lr, lamb=self.lamb, input_dim=self.imgs[idx].shape[1],
            N_train=len(train_idx), n_hid=self.n_hid, n_hid2=self.n_hid2, n_hid3=self.n_hid3, n_hid4=self.n_hid4,
            phi=self.phi, num_layer=self.nb_layer, w_dim=2,
            n_knots=n_knots, n_img=self.n_img, step_decay_epoch=200, step_gamma=0.2
        )

    def _load_saved_samples(self, repetition):
        checkpoint = torch.load(self.path+f"/neuroimg_prob{repetition}.pth")
        return checkpoint['weight_set_samples']
