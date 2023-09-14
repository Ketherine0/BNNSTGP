# Bayesian Spatially Varying Weight Neural Networks with the Soft-Thresholded Gaussian Process Prior

### Install package
```
pip install -i https://test.pypi.org/simple/ pkg-bnnstgp1
```

### import package
```
from pkg_bnnstgp1.pkg_bnnstgp import BNN_model
```

### Example of loading neuroimaging data and coordinate data
```
Y = pd.read_csv("/nfs/turbo/jiankanggroup/ellahe/y1.csv").iloc[:,1].values\
idx = np.invert(np.isnan(Y))`\
Y = Y[idx]

hf = h5py.File('/nfs/turbo/jiankanggroup/ellahe/image1.hdf5', 'r')\
img1 = hf.get('img')['img1'][()][idx,:]\

h2 = h5py.File('/nfs/turbo/jiankanggroup/ellahe/coord1.hdf5', 'r')\
coord1 = h2.get('coord')['coord1'][()]

hf = h5py.File('/nfs/turbo/jiankanggroup/ellahe/image_fMRI2.hdf5', 'r')\
img2 = hf.get('img')['img_fMRI'][()][idx,:]

h2 = h5py.File('/nfs/turbo/jiankanggroup/ellahe/coord_fMRI2.hdf5', 'r')\
coord2 = h2.get('coord')['coord_fMRI'][()]

coord = [coord1, coord2]`\
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
rep_num = 20
a = 0.01
b = 100 
poly_degree = 18
num_weight_samples = 70
lamb = 10
n_hid = 128
n_hid2 = 16
lr = 3e-3
model_dir = "multi_test_resize_all"
model_save_path = "/scratch/jiankang_root/jiankang1/ellahe/"+model_dir
os.makedir(model_dir)
num_layer = 2
num_epochs = 151
thred = 0.5
background_image = "../data/neuroimaging/AAL_MNI_2mm.nii"
batch_size = 128
nii_save_path = 'model_sig_nii/select_region_unit'
```

### Constructing model
```
BNN_neuroimg = BNN_model(coord=coord, imgs=img_data, cov=np.zeros((img_data[0].shape[0],1)),
                         Y=Y,rep=20,a=0.01,b=100, poly_degree=18, N_saves=70,
                         lamb=10,n_hid=128, n_hid2=16, lr=3e-3,path="/nfs/turbo/jiankanggroup/ellahe/multi_test_resize",nb_layer=2, n_epochs=151,
                        thred = 0.5, bg_image = "../data/neuroimaging/AAL_MNI_2mm.nii", batch_size=128,
                        nii_path = 'model_sig_nii/select_region_unit', n_img=len(img_data),
                        device='cuda' if torch.cuda.is_available() else 'cpu')
```

### Data preprocessing
```
BNN_neuroimg.load_data()
```

### Model training 
Prespesified for traning 20 repetitions and 150 epochs per repetition
```
R2 = BNN_neuroimg.train_model()
```

For later region selection part, we have to do one training with all data included. Change the training ratio, repetition number and model saving path
```
train_ratio = 1
rep = 1
model_dir2 = "multi_test_resize_all"
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
```
BNN_neuroimg.beta_fdr_control()
```

                        
