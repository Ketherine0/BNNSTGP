import numpy as np
import torch
import torch.nn.functional as F
import copy
from sklearn.linear_model import LinearRegression
import random
import pandas as pd

# from BNNSTGP3.data_split import TrainTestSplit
from model.neuroimg_network import NeuroNet

class BetaFdr:
    """
    input the model path
    example: "model/neuroimg_prob"
    return the spatially varying function (beta) after FDR control of all reptitions
    """
    def __init__(self, rep, path, lr, lamb, Y,
                 n_hid, n_hid2, n_hid3, n_hid4, hid_u=None,coord=None, imgs=None, W=None,  
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
        
    def fdr(self, net, weight_samples, net_phi, net_b, coord):
        # self.coord = coord

        beta = np.zeros([len(weight_samples),coord.shape[0],self.n_hid])
        print("beta",beta.shape)

        for i, weight_dict in enumerate(weight_samples):
            net.model.load_state_dict(weight_dict)
            out = 0
            net.model.load_state_dict(weight_dict)
            out = torch.mm(net_phi, net_b)
            out = F.threshold(out, net.model.lamb, net.model.lamb) - F.threshold(-out, net.model.lamb, net.model.lamb)
            out = net.model.sigma * out
            b = out.cpu().detach().numpy()
            beta[i,:,:] = b
        select_len = []
        beta_fdr_all = np.zeros((beta.shape[2], coord.shape[0]))
        
        r = []

        # calculate according to each hidden unit
        delta = np.zeros((beta.shape[2]))
        region_select_all = []
        for j in range(beta.shape[2]):
            b1 = copy.deepcopy(beta[:,:,j])
            b1[b1!=0] = 1
            b1 = b1.sum(0)/len(weight_samples)
            b1_sort = np.sort(b1)[::-1]
            b1_ind = np.argsort(-b1)

            i = 0
            b1_sort_inv = 1-b1_sort
            while i < len(b1):
                i += 1
                if sum(b1_sort_inv[:i+1])/(i+1) > 0.01:
                    l = i
                    ind = b1_ind[i]
                    delta[j] = i
                    break
            threshold = b1[ind]
        print(delta)
        delta_max = np.max(delta)

        for j in range(beta.shape[2]):
            b1 = copy.deepcopy(beta[:,:,j])
            b1[b1!=0] = 1
            b1 = b1.sum(0)/len(weight_samples)
            b1_sort = np.sort(b1)[::-1]
            b1_ind = np.argsort(-b1)

            region_select = b1_ind[:int(delta_max)]
            for i in range(len(region_select)):
                if region_select[i] not in region_select_all:
                    region_select_all.append(region_select[i])

        for z in range(beta.shape[2]):
            beta_fdr = np.zeros([coord.shape[0]])
            b_unit = copy.deepcopy(beta[:,:,z])
            b_unit = b_unit.mean(0)
            beta_fdr[region_select_all] = b_unit[region_select_all]
            beta_fdr_all[z,:] = beta_fdr
        # beta_all[seed,:,:] = beta_fdr_all
        
        return beta_fdr_all
        

    def beta_fdr(self,idx):

        input_dim = self.imgs[idx].shape[1]
        if self.n_hid4!=None:
            self.hid_u = self.n_hid4
        elif self.n_hid3!=None:
            self.hid_u = self.n_hid3
        elif self.n_hid2!=None:
            self.hid_u = self.n_hid2
        else:
            self.hid_u = self.n_hid
        beta_all = np.zeros((self.rep, self.n_hid,self.coord[idx].shape[0]))
        print("beta_all",beta_all.shape)

        for seed in range(self.rep):
        # for seed in range(self.rep):
            print(seed)
            path = self.path+"/neuroimg_prob"+str(seed)+".pth"

            np.random.seed(seed)
            torch.manual_seed(seed)
            input_dim = self.imgs[idx].shape[1]
            train_ratio = 0.8

            train_idx = np.random.choice(np.arange(len(self.Y)), int(train_ratio * len(self.Y)), replace = False)
            n_train = int(train_ratio * len(self.Y))

            # reg = LinearRegression().fit(self.W[train_idx,:self.W.shape[1]-1], self.Y[train_idx])
            reg = LinearRegression().fit(self.W[train_idx,:self.W.shape[1]-1].detach().cpu().numpy(), self.Y[train_idx].detach().cpu().numpy())
            reg_init = np.append(reg.coef_, reg.intercept_).astype(np.float32)

            n_knots = []
            for i in range(self.n_img):
                n_knots.append(self.phi[i].shape[1])

            net = NeuroNet(reg_init=reg_init, 
                           lr = self.lr, lamb = self.lamb, input_dim = input_dim, 
                           N_train=n_train, n_hid=self.n_hid, n_hid2=self.n_hid2,
                           n_hid3=self.n_hid3, n_hid4=self.n_hid4,
                           phi = self.phi, num_layer=self.nb_layer, w_dim = self.W.shape[1],
                           n_knots=n_knots, n_img=self.n_img,
                           step_decay_epoch = 200, step_gamma = 0.2, b_prior_sig = None,
                           langevin=False)    

            checkpoint = torch.load(path)
            weight_samples = checkpoint['weight_set_samples']
            net.load_state_dict(checkpoint['model'])

            b = self.fdr(net, weight_samples, net.model.phi[idx], net.model.b[idx], self.coord[idx])
            beta_all[seed,:,:] = b
            b[abs(b)>0] = 1
            # b = np.mean(b,axis=0)
            pd.DataFrame(b).to_csv("Simulation"+str(idx)+"/Itr"+str(seed)+".csv")
            # pd.DataFrame(b).to_csv("Simulation4_total"+str(idx)+"/all.csv")
            
        # beta_all = np.mean(beta_all,axis=1)
        # beta_f = beta_all
        # beta_f[beta_f!=0] = 1
        # pd.DataFrame(beta_f).to_csv("Simulation_total"+str(idx))+"/Itr"+".csv")
            
        return beta_all
      