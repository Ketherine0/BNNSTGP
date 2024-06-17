import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from scipy.stats import invgamma
import os
import copy
import time
from sklearn.linear_model import LinearRegression
import random

from model.data_split import TrainTestSplit
from model.neuroimg_network import NeuroNet

class PostPred:
    """
    input the number of reptitions and the file save path
    function will compute the coverage ratio
    """
    
    def __init__(self, rep, path, lr, lamb, batch_size,n_hid, n_hid2, n_hid3, n_hid4, nb_layer,Y, W=None, phi=None, imgs=None, n_img=1):
        self.rep = rep
        self.path = path
        self.imgs = imgs
        self.Y = Y
        self.W = W
        self.phi = phi
        self.lr = lr
        self.lamb = lamb
        self.batch_size = batch_size
        self.nb_layer = nb_layer
        self.n_hid = n_hid
        self.n_hid2 = n_hid2
        self.n_hid3 = n_hid3
        self.n_hid4 = n_hid4
        self.n_img = n_img
    
    
    def compute_loss(self, net, weight_samples, x, w, y):
        """
        input the model, model weight samples, and training x, w, y for update
        function will output the computed MSE loss for each iteration
        """
        loss_all = np.zeros(len(weight_samples))
        for i, weight_dict in enumerate(weight_samples):
            net.model.load_state_dict(weight_dict)
            out = 0
            for j in range(self.n_img):
                out1 = torch.mm(self.phi[j],net.model.b[j])
                out1 = F.threshold(out1, self.lamb, self.lamb) - F.threshold(-out1, self.lamb, self.lamb)
                out1 = net.model.sigma * out1
                out += torch.mm(x[j], out1)
            out += net.model.eta
            out = net.model.act(out)
            if self.nb_layer >= 2:
                out = net.model.fc(out)
                # out = self.bn(out)
                out = net.model.act(out)
                if self.nb_layer >= 3:
                    out = net.model.fc2(out)
                    # out = self.fc_bn2()
                    out = net.model.act(out)
                    if self.nb_layer >= 4:
                        out = net.model.fc3(out)
                        # out = self.fc_bn3()
                        out = net.model.act(out)
            out = torch.mm(out, net.model.zeta) + torch.mm(w, net.model.alpha) + torch.normal(mean=0,std=abs(net.model.noise)**(1/2))
            loss = F.mse_loss(out, y, reduction='sum')
            loss = loss.cpu().detach().numpy()
            out = out.cpu().detach().numpy()
            # n = out.shape[0]
            loss_all[i] = loss

        return torch.tensor(loss_all)



    def coverage_ratio(self, data,error,y,confidence=0.5):
        """
        input output from model and the noise sample
        compute coverage ratio
        """
        cov = np.zeros((data.shape[0]))
        # print('error',error)
        for i in range(data.shape[0]):
            samples = np.zeros((len(error)))
            for j in range(len(error)):
                samples[j] = data[i]+error[j]
            lower = np.quantile(samples,0.01)
            upper = np.quantile(samples,0.99)
            # print("intervals: ", lower,upper)
            # print("real y:", y[i])
            if y[i]>=lower and y[i]<=upper:
                cov[i]=1
        return np.mean(cov)
    
    
    def credible_output(self, net, batch_size, weight_samples, x, w, y, loss):
        """
        input x, w, y and the MSE loss, estimate sigma and noise
        finally output the coverage ratio
        """
        beta_all = []
        for j in range(self.n_img):
            beta_all.append(np.zeros([len(weight_samples),x[j].shape[1],net.n_hid]))
        for i, weight_dict in enumerate(weight_samples):
            net.model.load_state_dict(weight_dict)
            out = 0
            for k in range(self.n_img):
                out1 = torch.mm(self.phi[k],net.model.b[k])
                out1 = F.threshold(out1, self.lamb, self.lamb) - F.threshold(-out1, self.lamb, self.lamb)
                out1 = net.model.sigma * out1
                b = out1.cpu().detach().numpy()
                beta_all[k][i,:] = b
                
        out_all = []
        for i in range(self.n_img):
            out_all.append(np.zeros((len(weight_samples),x[i].shape[0])))
        if loss.shape[0]>1:
            loss = loss[:loss.shape[0]-1,:]
        ns = batch_size*loss.shape[0]
        sigma_all = np.zeros((loss.shape[1]))
        for z in range(loss.shape[1]):
            # sigma_all[z] = invgamma.rvs(size=1,a=100+ns/2,scale=0.01+loss[:,z].sum(0)/2)
            sigma_all[z] = invgamma.rvs(size=1,a=1+ns/2,scale=10+loss[:,z].sum(0)/2)
        sigma_all = torch.tensor(sigma_all)
        cov_all = np.zeros((len(weight_samples)))
        for i in range(len(weight_samples)):
            out = 0
            for k in range(self.n_img):
                b = torch.tensor(copy.deepcopy(beta_all[k][i,:,:]),dtype=torch.float32)
                out += torch.mm(x[k], b)
            out += net.model.eta
            out = net.model.act(out)
            if self.nb_layer >= 2:
                out = net.model.fc(out)
                # out = self.bn(out)
                out = net.model.act(out)
                if self.nb_layer >= 3:
                    out = net.model.fc2(out)
                    # out = self.fc_bn2()
                    out = net.model.act(out)
                    if self.nb_layer >= 4:
                        out = net.model.fc3(out)
                        # out = self.fc_bn3()
                        out = net.model.act(out)
            out = torch.mm(out, net.model.zeta) + torch.mm(w, net.model.alpha)
            out = out.cpu().detach().numpy()
            error = np.random.normal(0,(sigma_all[i])**(1/2),100)
            cov = self.coverage_ratio(out,error,y,confidence=0.5)
            cov_all[i]=cov

        return np.mean(cov_all)
    
    def compute_cov(self):
    
        for seed in range(self.rep):
            start_epoch = 0

            np.random.seed(seed)
            torch.manual_seed(seed)
            input_dim = self.imgs[0].shape[1]
            train_ratio = 0.8
            
            train_idx = np.random.choice(np.arange(len(self.Y)), int(train_ratio * len(self.Y)), replace = False)
            test_idx = np.ones(len(self.Y), dtype=bool)
            test_idx[train_idx] = 0
            n_train = int(train_ratio * len(self.Y))
            n_test = len(self.Y)-n_train
            nbatch_train = (n_train+self.batch_size-1) // self.batch_size
            nbatch_test = (n_test+self.batch_size-1) // self.batch_size
            
            reg = LinearRegression().fit(self.W[train_idx,:self.W.shape[1]-1], self.Y[train_idx])
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

            nsamples0 = (n_train+self.batch_size-1) // self.batch_size
            nsamples1 = (n_test+self.batch_size-1) // self.batch_size

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            path = self.path+"/neuroimg_prob"+str(seed)+".pth"

            if os.path.exists(path):
                print('seed',seed)
                checkpoint = torch.load(path)
                net.load_state_dict(checkpoint['model'])
                start_epoch = checkpoint['epoch']
                best_R2 = checkpoint['best_R2']
                weight_samples = checkpoint['weight_set_samples']
                print('For random split '+str(seed)+', loading epoch {} successfully!'.format(start_epoch))

                
                loss_all = np.zeros((nsamples0, len(weight_samples)))
                indices = list(train_idx)
                random.seed(seed)
                random.shuffle(indices)
                
                batch_n = 0
                while batch_n < nbatch_train:
                    batch_indices = np.asarray(indices[0:self.batch_size])  
                    indices = indices[self.batch_size:] + indices[:self.batch_size] 
                    x = []
                    for j in range(self.n_img):
                        x.append(torch.tensor(self.imgs[j][indices]).float().to(device))
                    w = torch.tensor(self.W[indices])
                    w = w.float().to(device)
                    y = torch.tensor(self.Y[indices])
                    y = y.float().to(device).reshape(-1, 1)
                    loss = self.compute_loss(net, weight_samples, x, w, y)
                    loss_all[batch_n,:]=loss
                    batch_n += 1
                cov_total = []
                indices = list(test_idx)
                random.seed(seed)
                random.shuffle(indices)
                
                batch_n = 0
                while batch_n < nbatch_test:
                    batch_indices = np.asarray(indices[0:self.batch_size])  
                    indices = indices[self.batch_size:] + indices[:self.batch_size] 
                    x = []
                    for j in range(self.n_img):
                        x.append(torch.tensor(self.imgs[j][indices]).float().to(device))
                    w = torch.tensor(self.W[indices])
                    w = w.float().to(device)
                    y = torch.tensor(self.Y[indices])
                    y = y.float().to(device).reshape(-1, 1)
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    cov = self.credible_output(net,self.batch_size, weight_samples, x, w, y, loss_all)
                    cov_total.append(cov)
                    batch_n += 1
                print("For random split "+str(seed)+", coverage ratio is "+str(np.mean(cov_total)))

            else:
                raise Exception("Model path does not exist")
            
