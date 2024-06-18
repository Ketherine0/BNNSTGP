# from ex_bnnstgp import BNN_model
import pandas as pd
import h5py
import numpy as np
import torch
import os
import sys
# current_dir = os.getcwd()
# sub_dir = os.path.join(current_dir, 'packaging2/pkg_bnnstgp2') 
# sys.path.append(sub_dir)
# from pkg_bnnstgp import BNN_model
import os
os.environ['R_HOME'] = '/home/ellahe/.conda/envs/bnnstgp/lib/R'
from pkg_bnnstgp2.pkg_bnnstgp import BNN_model


# import rpy2.robjects as robjects


from rpy2.robjects.packages import importr

utils = importr('utils')



# options(repos=c('https://repo.miserver.it.umich.edu/cran/'))
# utils.install_packages('BayesGPfit', verbose = 0,repos = 'https://repo.miserver.it.umich.edu/cran/')
GP = importr('BayesGPfit')


Y = pd.read_csv("../data/y1.csv").iloc[:,1].values
idx = np.invert(np.isnan(Y))
Y = Y[idx]

hf = h5py.File('/scratch/jiankang_root/jiankang1/ellahe/image1.hdf5', 'r')
img1 = hf.get('img')['img1'][()][idx,:]
# img2 = hf.get('img')['img2'][()][idx,:]

h2 = h5py.File('/scratch/jiankang_root/jiankang1/ellahe/coord1.hdf5', 'r')
hf = h5py.File('/scratch/jiankang_root/jiankang1/ellahe/image_fMRI2.hdf5', 'r')
img2 = hf.get('img')['img_fMRI'][()][idx,:]

h2 = h5py.File('/scratch/jiankang_root/jiankang1/ellahe/coord_fMRI2.hdf5', 'r')
img_data = [img1, img2]



import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Assuming img_data is a list of your image data arrays [img1, img2] and Y is your target
X_combined = np.hstack(img_data)  # Combine your image features

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Convert to torch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_tensor, Y_tensor, test_size=0.2, random_state=42)

# Create DataLoader for both training and validation sets
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

import torch.nn as nn
import torch.nn.functional as F


class DNNModel(nn.Module):
    def __init__(self, input_size):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(512)  # Batch normalization after first layer
        self.bn2 = nn.BatchNorm1d(256)  # Batch normalization after second layer
        self.bn3 = nn.BatchNorm1d(128)  # Batch normalization after third layer
        self.bn4 = nn.BatchNorm1d(64)   # Batch normalization after fourth layer

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.dropout(x)
        x = self.bn4(F.relu(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x



from torch.optim import Adam
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the model
model = DNNModel(input_size=X_train.shape[1])

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)  # Learning rate

print_every = 5
print_test_every = 20
epochs = 100  # Number of epochs
for epoch in range(epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    if (epoch + 1) % print_every == 0:
        with torch.no_grad():
            model.eval()
            train_predictions = model(X_train)
            train_r2 = r2_score(Y_train, train_predictions.cpu().numpy())
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Train R2: {train_r2}")
            model.train()
            
    if (epoch + 1) % print_test_every == 0:
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            test_predictions = []
            true_targets = []
            for inputs, targets in val_loader:
                outputs = model(inputs)
                test_predictions.append(outputs.cpu().numpy())
                true_targets.append(targets.cpu().numpy())

            test_predictions = np.concatenate(test_predictions)
            true_targets = np.concatenate(true_targets)
            test_r2 = r2_score(true_targets, test_predictions)
            print(f"Epoch {epoch+1}/{epochs}, Test R2: {test_r2}")