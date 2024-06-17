import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import copy

from model.BNN_base import BNN
from model.SGLD_optim import SGLD

class NeuroNet(nn.Module):

        def __init__(self, reg_init=None, lr=1e-3, input_dim=784, n_hid = 32, n_hid2 = None, n_hid3 = None, 
                     n_hid4 = None, output_dim = 1, w_dim = 1, n_knots = 66, 
                     N_train=200, phi=None,
                     lamb = 1, langevin = True, step_decay_epoch = 100, step_gamma = 0.1, act =
                     'relu', b_prior_sig=None, num_layer=None, n_img=None):
            super(NeuroNet, self).__init__()

            #print(' Creating Net!! ')
            self.lr = lr

            self.input_dim = input_dim
            self.n_hid = n_hid
            self.n_hid2 = n_hid2
            self.n_hid3 = n_hid3
            self.n_hid4 = n_hid4
            self.reg_init = reg_init
            self.output_dim = output_dim
            self.w_dim = w_dim
            self.n_knots = n_knots
            self.phi = phi
            self.lamb = lamb
            self.act = act
            if b_prior_sig:
                self.b_prior_sig = nn.ParameterList([])
                for i in range(n_img):
                    self.b_prior_sig.append(torch.Tensor(b_prior_sig[i]))
            else:
                self.b_prior_sig = None
            self.num_layer = num_layer
            self.n_img = n_img

            self.N_train = N_train
            self.langevin = langevin
            self.step_decay_epoch = step_decay_epoch
            self.step_gamma = step_gamma

            self.create_net()
            self.create_opt()
            self.epoch = 0

            self.weight_set_samples = []


        def create_net(self):
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = BNN(input_dim=self.input_dim,reg_init=self.reg_init,  
                             n_hid=self.n_hid, n_hid2=self.n_hid2,
                                         n_hid3=self.n_hid3, n_hid4=self.n_hid4, output_dim=self.output_dim,
                                         w_dim=self.w_dim, n_knots = self.n_knots,
                                         phi=self.phi,
                                         lamb = self.lamb, act = self.act, b_prior_sig=self.b_prior_sig,
                                         num_layer=self.num_layer, n_img=self.n_img)
            self.model.to(self.device)
            #print('    Total params: %.2fK' % (self.get_nb_parameters() / 1000.0))


        def create_opt(self):
            self.optimizer = SGLD(params=self.model.parameters(), lr=self.lr, langevin = self.langevin)
            self.scheduler = StepLR(self.optimizer, step_size = self.step_decay_epoch, gamma=self.step_gamma)


        def fit(self, x, w, y):
            for i in range(self.n_img):
                x[i] = x[i].to(self.device)
            w = w.to(self.device)
            y = y.float().to(self.device).reshape(-1, 1)
            y = y.reshape(-1, 1)
            self.optimizer.zero_grad()

            out = self.model(x, w)
            # loss = F.mse_loss(out, y, reduction='mean')
            # loss = loss * self.N_train
            loss = F.mse_loss(out, y, reduction='sum')
            # likelihood
            loss += self.model.log_prior()

            loss.backward()
            self.optimizer.step()

            #R2 = 1 - torch.sum((out-y)**2)/torch.sum((y-y.mean())**2)

            return loss*x[0].shape[0]/self.N_train, out


        def eval(self, x, w, y):
            for i in range(self.n_img):
                x[i] = x[i].to(self.device)
            w = w.to(self.device)
            y = y.float().to(self.device).reshape(-1, 1)
            y = y.reshape(-1, 1)

            out = self.model(x, w)
            loss = F.mse_loss(out, y, reduction='mean')
            loss = loss * self.N_train

            return loss*x[0].shape[0]/self.N_train, out


        def get_nb_parameters(self):
            return sum(p.numel() for p in self.model.parameters())


        def save_net_weights(self, max_samples):

            if len(self.weight_set_samples) >= max_samples:
                self.weight_set_samples.pop(0)
                
            self.weight_set_samples.append(copy.deepcopy(self.model.state_dict()))


        def all_sample_eval(self, x, w, y):
            for i in range(n_img):
                x[i] = x[i].to(self.device)
                
            w = w.to(self.device)

            pred = x[0].new(len(self.weight_set_samples), x[0].shape[0], self.output_dim)

            for i, weight_dict in enumerate(self.weight_set_samples):
                self.model.load_state_dict(weight_dict)
                pred[i] = self.model(x, w)

            return pred.mean(0)


        def save(self, filename):
            print('Writting %s\n' % filename)
            torch.save({
                'epoch': self.epoch,
                'lr': self.lr,
                'model': self.model,
                'optimizer': self.optimizer,
                'scheduler': self.scheduler}, filename)


        def load(self, filename):
            print('Reading %s\n' % filename)
            state_dict = torch.load(filename)
            self.epoch = state_dict['epoch']
            self.lr = state_dict['lr']
            self.model = state_dict['model']
            self.optimizer = state_dict['optimizer']
            self.scheduler = state_dict['scheduler']
            print('  restoring epoch: %d, lr: %f' % (self.epoch, self.lr))
            return self.epoch
