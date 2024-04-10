import numpy as np
import torch
import torch.nn.functional as F
import copy
from sklearn.linear_model import LinearRegression
import random
import pandas as pd
import h5py
import os
import gc
import matplotlib.pyplot as plt

# from BNNSTGP3.data_split import TrainTestSplit
from model.neuroimg_network import NeuroNet

class BetaFdr:
    """
    input the model path
    example: "model/neuroimg_prob"
    return the spatially varying function (beta) after FDR control of all reptitions
    """
    def __init__(self, rep, path, lr, lamb, Y,
                 n_hid, n_hid2, n_hid3, n_hid4, fdr_thred, train_ratio=0.8, hid_u=None,coord=None, imgs=None, W=None,  
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
        self.n_img= n_img
        self.hid_u = hid_u
        self.train_ratio = train_ratio
        self.fdr_thred = fdr_thred
        
    def calculate_beta_fdr(self, network, weight_samples, network_phi, network_b, coord):
        """This function calculates hypothesis testing corrections using False Discovery Rate for Beta values.
        
        Args:
        - network: the NeuroNet model
        - weight_samples: tensor of weights samples
        - network_phi: tensor of phi values in network
        - network_b: tensor of b values in network
        - coord: coordinates shape vector

        Returns:
        - Beta False Discovery Rate values"""
        
        # Initialize the beta tensor with dims: weight samples x coordinate shape x hidden units num
        beta = np.zeros([len(weight_samples), coord.shape[0], self.n_hid])
        # Iterate over weight samples
        for idx, weight in enumerate(weight_samples):
            network.model.load_state_dict(weight)  # Load weights into model
            
            # Perform a series of transformations to obtain beta values
            output = torch.mm(network_phi, network_b)
            output = F.threshold(output, network.model.lamb, network.model.lamb) - F.threshold(-output, network.model.lamb, network.model.lamb)
            output = network.model.sigma * output
            output = output.cpu().detach().numpy()
            beta[idx,:,:] = output

        return self._process_beta(beta, weight_samples, coord)

    def _process_beta(self, beta, weight_samples, coord):
        """Helper function to calculate necessary transformations on beta and return Beta False Discovery Rate values.

        Args:
        - beta: beta tensor
        - weight_samples: tensor of weights samples
        - coord: coordinates shape vector

        Returns:
        - Beta False Discovery Rate values"""
        # Pool indices of selected regions across all hidden units
        region_select_all = []
        # Iterate over hidden units
        thred_prob = []
        # beta: weight samples x coordinate shape x hidden units num
        for j in range(beta.shape[2]):
            unit_beta = beta[:,:,j].copy()
            unit_beta[unit_beta != 0] = 1  # Convert beta values to binary

            # filtered_unit_beta = unit_beta[unit_beta < 0.8]
#             print("Statistical Summary of unit_beta:")
#             print(f"Mean: {np.mean(filtered_unit_beta)}")
#             print(f"Median: {np.median(filtered_unit_beta)}")
#             print(f"Standard Deviation: {np.std(filtered_unit_beta)}")
#             print(f"Min: {np.min(filtered_unit_beta)}")
#             print(f"Max: {np.max(filtered_unit_beta)}")

#             # Plotting the distribution of unit_beta
#             plt.hist(filtered_unit_beta, bins=10, alpha=0.7)
#             plt.title("Distribution of unit_beta")
#             plt.xlabel("Value")
#             plt.ylabel("Frequency")
#             plt.show()

            # Rank beta values for each sample
            unit_beta = np.sum(unit_beta, axis=0) / len(weight_samples)
            sorted_indices = np.argsort(-unit_beta)

            # Find index `l` such that the average of sorted beta over l is larger than threshold
            l = next(i for i, x in enumerate(np.cumsum(1-unit_beta[sorted_indices]) / (np.arange(len(unit_beta)) + 1)) if x > self.fdr_thred)
            thred_prob.append(unit_beta[sorted_indices[l]])
            # print("The threshold inclusion probability is: ", unit_beta[sorted_indices[l]])
            region_select_all.extend(np.unique(sorted_indices[:l]))  # Store unique selected regions across all hidden units
        print("The inclusion threshold probability for each hidden unit is: ")
        print(thred_prob)

        # Initialize empty Beta False Discovery Rate tensor
        # hidden units num x coordinates
        beta_fdr_all = np.zeros((beta.shape[2], coord.shape[0]))
        # Averaging beta values after fdr control along different weight samples
        for z in range(beta.shape[2]):
            beta_fdr = np.zeros([coord.shape[0]])
            unit_beta = beta[:,:,z].mean(axis=0)
            beta_fdr[region_select_all] = unit_beta[region_select_all]
            beta_fdr_all[z,:] = beta_fdr
        return beta_fdr_all
        
        

    def beta_fdr(self, idx, fdr_path):
        """This function iterates over each repetition and calculates Beta False Discovery Rate values.
        
        Args:
        - idx: index number for the current repetition

        Returns:
        - Beta False Discovery Rate values for each repetition"""
        
        # Determine the number of hidden units
        self.hid_u = self.n_hid4 or self.n_hid3 or self.n_hid2 or self.n_hid
        beta_all = np.zeros((self.rep, self.n_hid,self.coord[idx].shape[0]))

        for repetition in range(self.rep):
            print("Reptition: ", repetition)
            np.random.seed(repetition)
            torch.manual_seed(repetition)
                    
            # Select training data
            train_idx = np.random.choice(len(self.Y), int(self.train_ratio * len(self.Y)), replace=False)
            
            # Train a linear regression model
            reg = LinearRegression()
            # Fit the model with your data
            num_rows = self.W[train_idx, :].shape[0]
            ones_array = np.ones((num_rows, 1))  # Creating a 2D array with two columns of ones

            # Fitting the linear regression model
            reg.fit(ones_array, self.Y[train_idx].detach().numpy())
            # reg = LinearRegression().fit(self.W[train_idx,:-1], self.Y[train_idx].detach().numpy())
            # reg_init = np.append(reg.coef_, reg.intercept_).astype(np.float32)
            
            # Initialize a NeuroNet model
            n_knots = [self.phi[i].shape[1] for i in range(self.n_img)]
            net = self._init_nn_model(reg, idx, train_idx, n_knots)
            
            # Load network samples
            weight_samples = self._load_saved_samples(repetition)
            dir_path = fdr_path
            os.makedirs(dir_path, exist_ok=True)

            # Calculate beta after fdr control
#             beta_all[repetition,:,:] = self.calculate_beta_fdr(net, weight_samples, net.model.phi[idx], net.model.b[idx], self.coord[idx])
#             beta_all = np.where(np.abs(beta_all)>0, 1, 0)
#             print("beta_all sum: ", beta_all[repetition,:,:].sum())
            
#             # save voxel selection to a hdf5 file
#             b_mean = np.mean(beta_all[repetition,:,:], axis=0)
            
#             # dir_path = "Voxel"
#             # dir_path = "/scratch/jiankang_root/jiankang1/ellahe/Voxel1"
      
#             selection = np.zeros((self.rep,b_mean.shape[0]))
#             for i,b in enumerate(b_mean):
#                 if b >= 1:
#                     selection[repetition, i] = 1
#             if self.train_ratio==1:
#                 with h5py.File(f"{dir_path}/Modality_total{idx}{repetition}.hdf5", 'w') as hf:  
#                     hf.create_group("voxel").create_dataset(f"mod{idx}", data=selection)
#             else:
#                 with h5py.File(f"{dir_path}/Modality{idx}{repetition}.hdf5", 'w') as hf:  
#                     hf.create_group("voxel").create_dataset(f"mod{idx}", data=selection)

            # num of neurns x coord
            beta_all_repetition = self.calculate_beta_fdr(net, weight_samples, net.model.phi[idx], net.model.b[idx], self.coord[idx])

            # save voxel selection to a hdf5 file
            b_mean_repetition = np.mean(beta_all_repetition, axis=0)
            selection = np.zeros((1,b_mean_repetition.shape[0]))
            for i,b in enumerate(b_mean_repetition):
                if b > 0:
                    selection[0, i] = 1
            # Calculate and print the distribution of selection
            num_selected = np.sum(selection)
            num_not_selected = selection.size - num_selected
            proportion_selected = num_selected / selection.size
            proportion_not_selected = num_not_selected / selection.size

            print(f"Number of Selected Features: {num_selected}")
            print(f"Number of Not Selected Features: {num_not_selected}")
            print(f"Proportion of Selected Features: {proportion_selected:.2f}")
            print(f"Proportion of Not Selected Features: {proportion_not_selected:.2f}")
            
            
            
            # Writing the output as you go
            if self.train_ratio==1:
                with h5py.File(f"{dir_path}/Modality_total{idx}{repetition}.hdf5", 'w') as hf:  
                    hf.create_group("voxel").create_dataset(f"mod{idx}", data=selection)
            else:
                with h5py.File(f"{dir_path}/Modality{idx}{repetition}.hdf5", 'w') as hf:  
                    hf.create_group("voxel").create_dataset(f"mod{idx}", data=selection)
            
            # Delete large variables that are no longer in use
            del beta_all_repetition
            del b_mean_repetition
            del selection
            gc.collect()

        return beta_all
    
    def _init_nn_model(self, reg, idx, train_idx, n_knots):
        """Helper function to initialize a NeuroNet model"""
        return NeuroNet(
            reg_init=np.append(reg.coef_, reg.intercept_).astype(np.float32), 
            lr = self.lr, lamb = self.lamb, input_dim = self.imgs[idx].shape[1], 
            N_train=len(train_idx), n_hid=self.n_hid, n_hid2=self.n_hid2, n_hid3=self.n_hid3, n_hid4=self.n_hid4,
            phi = self.phi, num_layer=self.nb_layer, w_dim = 2,
            n_knots=n_knots, n_img=self.n_img, step_decay_epoch = 200, step_gamma = 0.2,
        )
        
    def _load_saved_samples(self, repetition):
        """Helper function to load saved Neural Network weights samples"""
        checkpoint = torch.load(self.path+f"/neuroimg_prob{repetition}.pth")
        return checkpoint['weight_set_samples']
