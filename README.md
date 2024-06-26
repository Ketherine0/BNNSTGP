# Bayesian Spatially Varying Weight Neural Networks with the Soft-Thresholded Gaussian Process Prior

![alt text](https://github.com/Ketherine0/BNNSTGP/blob/main/picture/BNN_struc.png)

### Clone python package
```
git clone https://github.com/Ketherine0/BNNSTGP.git
```

```
import pandas as pd
import h5py
import numpy as np
import torch
import os
import sys
```

### import package
```
current_dir = os.getcwd()
sub_dir = os.path.join(current_dir, 'bnnstgp') 
sys.path.append(sub_dir)
sub_dir = os.path.join(current_dir, '../') 
sys.path.append(sub_dir)

# Change to your R home
os.environ['R_HOME'] = '/home/ellahe/.conda/envs/bnnstgp/lib/R'
from pkg_bnnstgp import BNN_model
```
```
from rpy2.robjects.packages import importr
utils = importr('utils')
GP = importr('BayesGPfit')
```

### Example of loading neuroimaging data and coordinate data
```
Y = pd.read_csv("data/y1.csv").iloc[:,1].values
idx = np.invert(np.isnan(Y))
Y = Y[idx]

hf = h5py.File('/scratch/jiankang_root/jiankang1/ellahe/image1.hdf5', 'r')
img1 = hf.get('img')['img1'][()][idx,:]
# img2 = hf.get('img')['img2'][()][idx,:]

h2 = h5py.File('/scratch/jiankang_root/jiankang1/ellahe/coord1.hdf5', 'r')
coord1 = h2.get('coord')['coord1'][()]
# coord2 = h2.get('coord')['coord2'][()]

hf = h5py.File('/scratch/jiankang_root/jiankang1/ellahe/image_fMRI2.hdf5', 'r')
img2 = hf.get('img')['img_fMRI'][()][idx,:]

h2 = h5py.File('/scratch/jiankang_root/jiankang1/ellahe/coord_fMRI2.hdf5', 'r')
coord2 = h2.get('coord')['coord_fMRI'][()]

coord = [coord1, coord2]
img_data = [img1, img2]
```

rep_num: the number of repetitive trainings using different seeds (should set to 1 when using all data to train)\
a, b, poly_degree: parameters for eigen decomposition using the BayesGPfit package\
num_weight_samples: number of saved weight samples per epochs\
lamb: the threshold parameter for BNN-STGP\
n_hid: number of hidden units for the first layer\
n_hid2: number of hidden units for the second layer\
n_hid3: number of hidden units for the third layer\
n_hid4 = number of hidden units for the fourth layer\
lr: learning rate\
model_save_path: path where model is saved\
num_layer: number of layers for Bayesian Neural Network\
num_epochs: number of epochs to train per repetition\
thred: threshold of beta FDR control

```
rep_num = 10
a = 0.01
b = 100 
poly_degree = 18
num_weight_samples = 50
lamb = 10
n_hid = 128
n_hid2 = 16
lr = 3e-3
model_dir = "multi_test_resize_all2"

model_save_path = "/scratch/jiankang_root/jiankang1/ellahe/multi_test_resize_all2/"+model_dir
os.makedirs(model_save_path, exist_ok=True)
num_layer = 2
num_epochs = 131
thred = 0.4
back_image2 = "data/neuroimaging/AAL_MNI_2mm.nii"
back_image1 = "data/neuroimaging/AAL_90_3mm.nii"
regioninfo_file1 = "data/neuroimaging/AALregion_full.xls"
regioninfo_file2 = "data/neuroimaging/AAL_region_functional_networks.csv"
mni_file1 = "MNI_coords.csv"
mni_file2 = "MNI_coords_2mm.csv"
fdr_thred = 0.3
batch_size = 128
nii_save_path = model_save_path+'/select_region'
```

### Constructing model
```
BNN_neuroimg = BNN_model(coord=coord, imgs=img_data, cov=np.zeros((img_data[0].shape[0],1)),
                         Y=Y,rep=rep_num,a=a,b=b, poly_degree=poly_degree, 
                         N_saves=num_weight_samples,
                         lamb=lamb,n_hid=n_hid, n_hid2=n_hid2, 
                         lr=lr,path=model_save_path,nb_layer=num_layer, n_epochs=num_epochs,
                        thred = thred, bg_image1 = back_image1, bg_image2 = back_image2, 
                         region_info1 = regioninfo_file1, 
                         region_info2 = regioninfo_file2, mni_file1 = mni_file1,
                         mni_file2 = mni_file2, batch_size=batch_size,
                        nii_path = nii_save_path, fdr_thred = fdr_thred, 
                         n_img=len(img_data),
                        device='cuda' if torch.cuda.is_available() else 'cpu')
```

### Data preprocessing
```
BNN_neuroimg.load_data()
```

### Model training 
Prespesified for traning 20 repetitions and 150 epochs per repetition
```
R2_total = np.zeros(BNN_neuroimg.rep)
for seed in range(BNN_neuroimg.rep):
    model_train = BNN_neuroimg.create_model_train()
    best_R2 = model_train.train(seed)
    R2_total[seed] = best_R2
    print(f'{seed}: Best test R2: = {best_R2}')

print(R2_total)
```

For later region selection part, we have to do one training with all data included. Change the training ratio, repetition number and model saving path
```
train_ratio = 1
rep = 1
model_dir2 = "multi_test_resize_all2_total"
model_save_path = "/scratch/jiankang_root/jiankang1/ellahe/"+model_dir2
os.makedir(model_dir2)

BNN_neuroimg = BNN_model(coord=coord, imgs=img_data, cov=np.zeros((img_data[0].shape[0],1)),
                         Y=Y,rep=rep_num,a=a,b=b, poly_degree=poly_degree, N_saves=num_weight_samples,
                         lamb=lamb,n_hid=n_hid, n_hid2=n_hid2, lr=lr,path=model_save_path,nb_layer=num_layer, n_epochs=num_epochs,
                        thred = thred, bg_image = background_image, batch_size=batch_size,
                        nii_path = nii_save_path, n_img=len(img_data), train_ratio=train_ratio,
                        device='cuda' if torch.cuda.is_available() else 'cpu')
```


### Calculate Coverage Ratio
The three parts calculating coverage ratio, beta fdr control, and region selection, can only be accessed once the model is trained completely and stored in our pre-specified directory\
```
BNN_neuroimg.beta_post()
```

### Beta FDR control(used for next step's region selection)
Used for next step's region selection, we first calculate the beta values of 20 different reprtitions after FDR control
```
train_ratio = 0.8
rep = 20
model_save_path = "/scratch/jiankang_root/jiankang1/ellahe/multi_test_resize_all2/"+model_dir

BNN_neuroimg.beta_fdr_control()
```

Then calculate the beta values of one whole training using all data
```
train_ratio = 1
rep = 1
model_save_path = "/scratch/jiankang_root/jiankang1/ellahe/"+model_dir2

BNN_neuroimg.beta_fdr_control()
```
After those two steps, the beta values of different repetitions and with whole data training will be saved under directory "Voxel"

# Demonstration of Region Selection
<img src="https://github.com/Ketherine0/BNNSTGP/blob/main/picture/nii_region.png" width="500">

<img src="https://github.com/Ketherine0/BNNSTGP/blob/main/picture/region_table.png" width="600">

                        
