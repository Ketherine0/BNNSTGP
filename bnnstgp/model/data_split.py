import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset, DataLoader

class TrainTestSplit:
    """
    input the random seed and x,y,w data, output 
    1) the train loader and test loader
    2) number of training sets and test sets
    3) linear regression coefficient for parameter initialization
    """
    
    def __init__(self, seed, x, y, w):
        self.seed = seed
        self.x = x
        self.y = y
        self.w = w
        
    def train_test_split(self):
    
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        train_ratio = 0.8
        batch_size = 128

        train_idx = np.random.choice(np.arange(len(self.y)), int(train_ratio * len(self.y)), replace = False)
        train_x = self.x[train_idx]
        train_y = self.y[train_idx]

        mask = np.ones(len(self.y), np.bool)
        mask[train_idx] = 0
        test_x = self.x[mask]
        test_y = self.y[mask]
        self.n_train = len(train_x)
        self.n_test = len(test_x)
        train_w = self.w[train_idx]    
        test_w = self.w[mask]


        class ABCD_train(Dataset):
            def __len__(self):
                return len(train_y)
            def __getitem__(self, i):
                xdata = torch.tensor(train_x[i], dtype=torch.float32)
                wdata = torch.tensor(np.concatenate((train_w[i],np.array([1.]))), dtype=torch.float32)
                ydata = torch.tensor(train_y[i], dtype=torch.float32)
                return (xdata, wdata), ydata

        class ABCD_test(Dataset):
            def __len__(self):
                return len(test_y)
            def __getitem__(self, i):
                xdata = torch.tensor(test_x[i], dtype=torch.float32)
                wdata = torch.tensor(np.concatenate((test_w[i],np.array([1.]))), dtype=torch.float32)
                ydata = torch.tensor(test_y[i], dtype=torch.float32)
                return (xdata, wdata), ydata

        train_ABCD = ABCD_train()
        test_ABCD = ABCD_test()

        self.train_loader = DataLoader(train_ABCD, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_ABCD, batch_size=batch_size, shuffle=True)

        reg = LinearRegression().fit(train_w, train_y)

        self.reg_init = np.append(reg.coef_, reg.intercept_).astype(np.float32)


        return self.train_loader, self.test_loader, self.n_train, self.n_test, self.reg_init
