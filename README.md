# pkg_bnnstgp

#### from ex_bnnstgp import BNN_model
`import pandas as pd`
import h5py
import numpy as np
import torch

### Install package
pip install -i https://test.pypi.org/simple/ pkg-bnnstgp1

### import package
from pkg_bnnstgp1.pkg_bnnstgp import BNN_model

### Example of loading neuroimaging data and coordinate data
Y = pd.read_csv("/nfs/turbo/jiankanggroup/ellahe/y1.csv").iloc[:,1].values
idx = np.invert(np.isnan(Y))
Y = Y[idx]

hf = h5py.File('/nfs/turbo/jiankanggroup/ellahe/image1.hdf5', 'r')
img1 = hf.get('img')['img1'][()][idx,:]

h2 = h5py.File('/nfs/turbo/jiankanggroup/ellahe/coord1.hdf5', 'r')
coord1 = h2.get('coord')['coord1'][()]

hf = h5py.File('/nfs/turbo/jiankanggroup/ellahe/image_fMRI2.hdf5', 'r')
img2 = hf.get('img')['img_fMRI'][()][idx,:]

h2 = h5py.File('/nfs/turbo/jiankanggroup/ellahe/coord_fMRI2.hdf5', 'r')
coord2 = h2.get('coord')['coord_fMRI'][()]

coord = [coord1, coord2]
img_data = [img1, img2]

### Constructing model
BNN_neuroimg = BNN_model(coord=coord, imgs=img_data, cov=np.zeros((img_data[0].shape[0],1)),
                         Y=Y,rep=20,a=0.01,b=100, poly_degree=18, N_saves=70,
                         lamb=10,n_hid=128, n_hid2=16, lr=3e-3,path="/nfs/turbo/jiankanggroup/ellahe/multi_test_resize",nb_layer=2, n_epochs=151,
                        thred = 0.5, bg_image = "../data/neuroimaging/AAL_MNI_2mm.nii", batch_size=128,
                        nii_path = 'model_sig_nii/select_region_unit', n_img=len(img_data),
                        device='cuda' if torch.cuda.is_available() else 'cpu')

### Data preprocessing
BNN_neuroimg.load_data()

### Model training (Prespesified for traning 150 epochs
R2 = BNN_neuroimg.train_model() 

### Calculate Coverage Ratio
BNN_neuroimg.beta_post()

### Beta FDR control(used for next step's region selection)
#### The three parts calculating coverage ratio, beta fdr control, and region selection, can only be accessed once the model is trained completely and stored in our #### pre-specified directory
BNN_neuroimg.beta_fdr_control()

                        
