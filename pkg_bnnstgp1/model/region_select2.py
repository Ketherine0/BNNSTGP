import copy
import numpy as np
import nibabel as nib
## import R packages
import os
os.environ['R_HOME'] = '/home/ellahe/.conda/envs/BNNSTGP/lib/R'

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import pandas as pd


class Selection:
    
    def __init__(self, rep, path, coord, beta, thred, bg_image, nii_path, n_img):
        self.rep = rep
        self.path = path
        self.coord = coord
        self.beta = beta
        self.thred = thred
        self.bg_image = bg_image
        self.nii_path = nii_path
        self.n_img = n_img
    
    
#     def unit_match(self):
#         """
#         input the model path and beta after FDR control using beta_weight()
#         function will match hidden units along different reptitions 
#         """

#         beta_true = copy.deepcopy(self.beta)
#         print(beta_true.shape)
#         for h in range(1,self.rep):
#             for i in range(self.beta.shape[1]):
#                 beta_cor_all = []
#                 for j in range(i,self.beta.shape[1]):
#                     beta1 = copy.deepcopy(beta_true[0,i,:])
#                     beta2 = copy.deepcopy(beta_true[h,j,:])
#                     beta_cor = np.corrcoef(beta1,beta2)
#                     beta_cor_all.append(beta_cor[0,1])
#                 beta_cor_all = np.array(beta_cor_all)
#                 beta_sort = np.sort(beta_cor_all)[::-1]
#                 beta_sort_ind = np.argsort(-beta_cor_all)
#                 ind_max1 = beta_sort_ind[0]
#                 ind_max = ind_max1 + i

#                 if ind_max!=i:
#                     beta_tmp = copy.deepcopy(beta_true[h,i,:])
#                     beta_tmp_max = copy.deepcopy(beta_true[h,ind_max,:])
#                     beta_true[h,i,:]=beta_tmp_max
#                     beta_true[h,ind_max,:]=beta_tmp

#         return beta_true
    
    
    def output_selected_region(self, beta_m, idx):
        """
        input the matched beta, stability threshold, background image path, and nii image saving path
        example: 
        bg_image = "../data/neuroimaging/AAL_MNI_2mm.nii"
        top five hidden units with most regions selected will be demonstrated
        function will output the coordinate and save nii image data with selected region annotated
        """
        # r = 5
        readRDS = robjects.r['readRDS']
        # beta_m = beta_m[:r,:,:]
        stab_all = []
        beta_threshold = np.zeros((beta_m.shape[0],beta_m.shape[1],beta_m.shape[2]))
        for unit in range(beta_m.shape[1]):
            beta_unit = beta_m[:,unit,:]
            stab = []
            # print(np.count_nonzero(beta_unit))
            for i in range(self.coord.shape[0]):
                if np.count_nonzero(beta_unit[:,i])/beta_m.shape[0] >= self.thred:
                    stab.append(i)
            beta_threshold[:,unit,stab] = beta_unit[:,stab]
            stab_all.append(len(stab))
        stab_all = np.array(stab_all)
        print(stab_all)
        len_sort = np.argsort(-stab_all)

        select_unit_ind = len_sort[:5]

        beta_select = np.zeros((len(select_unit_ind), beta_m.shape[2]))
        z = 0
        coordinate = self.coord
        print(coordinate.shape)
        # AAL_img = nib.load(self.bg_image)
        # img_affine = AAL_img.affine
        for i in select_unit_ind:
            beta_unit = beta_m[:,i,:]
            stab = []
            for j in range(coordinate.shape[0]):
                if np.count_nonzero(beta_unit[:,j])/beta_m.shape[0] >= self.thred:
                    stab.append(j)
            beta_select[z,stab] = beta_unit[:,stab].mean(0)
            ### comment
            # beta_select[abs(beta_select)!=0] = 1
            # print(np.count_nonzero(beta_select))
            z+=1

        beta_select = beta_select.sum(0)
        print(np.count_nonzero(beta_select))
        loc_all = np.zeros((np.count_nonzero(beta_select),coordinate.shape[1]))
        # grid = round(coordinate.shape[0]**(1/coordinate.shape[1]))
        # print(grid)
        # loc_list = [grid]*coordinate.shape[1]
        # loc_all = np.zeros((loc_list))
        
        # b=0
        # for h in range(coordinate.shape[0]):
        #     if beta_select[h] != 0:
        #         loc_all[b,:] = coordinate[h,:]
        #         b+=1
        b = 0
        while b<coordinate.shape[0]:
            # for i in range(loc_all.shape[0]):
            #     for j in range(loc_all.shape[1]):
            #         for k in range(loc_all.shape[2]):
            #             if beta_select[b] != 0:
            #                 loc_all[i,j,k] = 1
            #             b +=1
            for idx, x in np.ndenumerate(loc_all):
                if beta_select[b] != 0:
                    loc_all[idx] = beta_select[b]
                b += 1
        # loc_all_df = pd.DataFrame(loc_all)
        # loc_all_df.to_csv("loc_"+idx+".csv")
                
            
        return loc_all