import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import scipy.stats
from scipy.stats import invgamma

class BNN(nn.Module):
        def __init__(self, input_dim, reg_init, n_hid, n_hid2, n_hid3, n_hid4, output_dim, w_dim, 
                     n_knots, phi, 
                     lamb=1., b_prior_sig=1, zeta_prior_sig=1, eta_prior_sig=1, alpha_prior_sig=1,
                     noise_prior_sig=1, num_layer = 3, n_img=1,
                     act='relu'):
            super(BNN, self).__init__()

            self.input_dim = input_dim
            self.n_hid = n_hid
            self.n_hid2 = n_hid2
            self.n_hid3 = n_hid3
            self.n_hid4 = n_hid4
            self.output_dim = output_dim
            self.w_dim = w_dim
            self.n_knots = n_knots
            # self.phi = nn.ParameterList([nn.Parameter(phi[0]),nn.Parameter(phi[1])])
            self.phi = []
            for index in range(n_img):
                self.phi.append(nn.Parameter(phi[index]))
                self.phi = nn.ParameterList(self.phi)
            self.lamb = lamb
            self.num_layer = num_layer
            self.n_img  = n_img

            self.sigma = nn.Parameter(torch.tensor(.1))
            self.b = nn.ParameterList([])
            if b_prior_sig is None:
                for i in range(self.n_img):
                    b1 = nn.Parameter(torch.Tensor(n_knots[i], n_hid).normal_(0, 1.))
                    self.b.append(b1)
            else:
                for i in range(self.n_img):
                    tmp_tensor = torch.Tensor(n_knots[i], n_hid)
                    for j in range(n_knots[i]):
                        for k in range(n_hid):
                            tmp_tensor[j,k].normal_(std=b_prior_sig)
                    b1 = nn.Parameter(tmp_tensor)
                    self.b.append(b1)
            
            self.zeta = nn.Parameter(torch.Tensor(n_hid, output_dim).normal_(0, .2))
            self.eta = nn.Parameter(torch.Tensor(n_hid).zero_())
            self.alpha = nn.Parameter(torch.tensor(reg_init).reshape(w_dim, output_dim))
            self.shape =100
            self.scale = 0.01
            self.noise = nn.Parameter(torch.tensor(np.float32(invgamma.rvs(size=output_dim,a=self.shape,scale=self.scale))))  

            self.b_prior_sig = b_prior_sig
            self.zeta_prior_sig = zeta_prior_sig
            self.eta_prior_sig = eta_prior_sig
            self.alpha_prior_sig = alpha_prior_sig
            self.noise_prior_sig = noise_prior_sig
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # self.bn = nn.BatchNorm2d(n_hid)
            if num_layer >= 2:
                self.fc = nn.Linear(self.n_hid, self.n_hid2)
                # self.fc_bn = nn.BatchNorm2d(n_hid2)
                self.zeta = nn.Parameter(torch.Tensor(self.n_hid2, self.output_dim).normal_(0, .2))
                if num_layer >= 3:
                    self.fc2 = nn.Linear(self.n_hid2, self.n_hid3)
                    # self.fc2_bn = nn.BatchNorm2d(n_hid3)
                    self.zeta = nn.Parameter(torch.Tensor(self.n_hid3, self.output_dim).normal_(0, .2))
                    if num_layer >= 4:
                        self.fc3 == nn.Linear(self.n_hid3, self.n_hid4)
                        # self.fc3_bn = nn.BatchNorm2d(n_hid4)
                        self.zeta = nn.Parameter(torch.Tensor(self.n_hid4, self.output_dim).normal_(0, .2))

            if act == 'relu':
                self.act = torch.relu
            elif act == 'tanh':
                self.act = torch.tanh
            elif act == 'sigmoid':
                self.act = torch.sigmoid
            else:
                raise ValueError('Invalid activation function %s' % act)

        def forward(self, x, w):
            out = 0
            for i in range(self.n_img):
                # print(self.phi[i].shape)
                # print(self.b[i].shape)
                out1 = torch.matmul(self.phi[i],self.b[i])
                out1 = F.threshold(out1, self.lamb, self.lamb) - F.threshold(-out1, self.lamb, self.lamb)
                out1 = self.sigma * out1
                out += torch.mm(x[i], out1)
            out += self.eta
            # out1 = torch.mm(self.phi1, self.b1)
            # out2 = torch.mm(self.phi2, self.b2)
            # out1 = F.threshold(out1, self.lamb, self.lamb) - F.threshold(-out1, self.lamb, self.lamb)
            # out2 = F.threshold(out2, self.lamb, self.lamb) - F.threshold(-out2, self.lamb, self.lamb)
            # out1 = self.sigma * out1
            # out2 = self.sigma * out2
            # out = torch.mm(x1, out1) + torch.mm(x2, out2) + self.eta

            out = self.act(out)
            if self.num_layer >= 2:
                out = self.fc(out)
                # out = self.bn(out)
                out = self.act(out)
                if self.num_layer >= 3:
                    out = self.fc2(out)
                    # out = self.fc_bn2()
                    out = self.act(out)
                    if self.num_layer >= 4:
                        out = self.fc3(out)
                        # out = self.fc_bn3()
                        out = self.act(out)
                        
            out = torch.mm(out, self.zeta) + torch.mm(w, self.alpha)+torch.normal(mean=0,std=abs(self.noise)**(1/2))

            return out

        def log_prior(self):
            logprior = 0
            if self.b_prior_sig is None:
                for i in range(self.n_img):
                    logprior += 0.5*(self.b[i]**2).sum()
            else:
                self.b_prior_sig = self.b_prior_sig.to(self.b.device)
                for i in range(self.n_img):
                    logprior += 0.5*((self.b[i]**2)/((self.b_prior_sig**2).reshape(-1,1))).sum()           
            logprior += 0.5*(self.zeta**2).sum()/(self.zeta_prior_sig**2)
            logprior += 0.5*(self.eta**2).sum()/(self.eta_prior_sig**2)
            logprior += 0.5*(self.alpha**2).sum()/(self.alpha_prior_sig**2)
            logprior += 0.5*(self.noise**2).sum()/(self.noise_prior_sig**2)
            return logprior


