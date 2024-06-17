import copy
import numpy as np
import nibabel as nib
## import R packages
# import os
# os.environ['R_HOME'] = '/home/ellahe/.conda/envs/bnnstgp/lib/R'

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import pandas as pd
import h5py
import matplotlib.pyplot as plt


class Selection:
    
    def __init__(self, imgs, coords, rep, path, coord, thred, bg_image1, bg_image2, region_info1, region_info2, mni_file1, mni_file2, nii_path, n_img):
        """
        The class constructor initializes several class attributes including images, coordinates, repetitions, paths, threshold, background images, region information, NIfTI (Neuroimaging Informatics Technology Initiative) image path, and number of images
        """
        self.imgs = imgs
        self.coords = coords
        self.rep = rep
        self.path = path
        self.coord = coord
        self.thred = thred
        self.bg_image1 = bg_image1
        self.bg_image2 = bg_image2
        self.region_info1 = region_info1
        self.region_info2 = region_info2
        self.mni_file1 = mni_file1
        self.mni_file2 = mni_file2
        self.nii_path = nii_path
        self.n_img = n_img
    

    def find_MNI_coords(self, region_name, AAL_info, AAL_mask, idx):
        """
        Find MNI (Montreal Neurological Institute) coordinates for a given region of interest.
        Args:
        region_name: Name of the region of interest
        AAL_info: DataFrame containing region mapping information based on Automated Anatomical Labeling (AAL)
        AAL_mask: Neuroimaging mask according to the AAL
        idx: Index representing the modality of the region

        Returns:
        MNI_coords: MNI coordinates of the region of interest
        """
        # Determine the region code from corresponding region name
        # region_code = (AAL_info[AAL_info['Short.Name'] == region_name].index[0]
        #                if idx == 0
        #                else AAL_info['region_code'][AAL_info[AAL_info['AAL_Region_Names'] == region_name].index[0]])
        region_code = (AAL_info[AAL_info['Short.Name'] == region_name].index[0]
                       if idx == 0
                       else AAL_info[AAL_info['Short.Name'] == region_name].index[0])

        # Find voxel (3D grid addressing a point in 3D space) coordinates from the mask
        voxel_coords = np.column_stack(np.where(AAL_mask.get_fdata() == region_code))

        # Transform voxel coordinates to real world space coordinates
        affine = AAL_mask.affine
        MNI_coords = np.dot(voxel_coords, affine[:3, :3].T) + affine[:3, 3]

        return MNI_coords


    def affine_transformation(self, idx, offsets, affines, coordinates, mni_coord, stab):
        """
        Perform an affine transformation on 3D coordinates.
        Args:
        offsets: The x, y, and z shifts applied to the coordinates
        affines: The x, y, and z scalars applied to the coordinates after the shift
        coordinates: The original coordinates
        mni_coord: The MNI coordinates used as a reference for the transformation
        stab: The stability values for each coordinate

        Returns:
        transformed_coords: The transformed coordinates
        matched_indices: Indices of the regions the transformed coordinates belong to
        merged_df: DataFrame that includes transformed coordinates and their matched indices
        """
        transformed_coords = [[aff * coord + offset for aff, coord, offset in zip(affines, coord, offsets)] for coord in coordinates]

        transformed_df = pd.DataFrame(transformed_coords, columns=['x', 'y', 'z'])
        if idx==0:
            mni_coords_df = pd.read_csv(self.mni_file1)
        elif idx==1:
            mni_coords_df = pd.read_csv(self.mni_file2)
        mni_coord =mni_coords_df
        merged_df = pd.merge(transformed_df, mni_coord, how='left', on=['x', 'y', 'z'])
        merged_df['value'] = stab
        matched_indices = merged_df['index'].tolist()

        return transformed_coords, matched_indices, merged_df


    def summarize_matched_regions(self, xls_file, matched_indices, stab, coord, idx):
        """
        Generate a summary of given regions indicating their appearance frequency.
        Args:
        xls_file: File path to the spreadsheet containing region names and indices
        matched_indices: A list of region indices
        stab: Stability values for each coordinate
        coord: The original coordinates
        idx: The modality index

        Returns:
        summary_df: Summary DataFrame of regions with their counts
        """
        regions_df = pd.read_excel(xls_file) if idx == 0 else pd.read_csv(xls_file)
        region_counts_df = pd.DataFrame.from_dict(Counter(matched_indices), orient='Index', columns=['count']).reset_index()
        region_counts_df.columns = ['Index', 'count']
        summary_df = pd.merge(region_counts_df, regions_df, on='Index', how='left')
        total_count = summary_df['count'].sum()
        summary_df['percentage'] = summary_df['count'] / total_count * 100
        summary_df = (summary_df[['Short.Name', 'count', 'percentage']]
                      if idx == 0
                      else summary_df[['AAL_Region_Names', 'count', 'percentage']])
        summary_df.sort_values(by='count', ascending=False, inplace=True)
        summary_df.reset_index(drop=True, inplace=True)

        return summary_df


    def calculate_stats_with_region(self, df_values, region_file, idx):
        """
        Calculate descriptive statistics for each region.
        Args:
        df_values: DataFrame containing values of the analyzed region
        region_file: File containing names of the regions
        idx: The modality index

        Returns:
        result_df: DataFrame with descriptive statistics for each region
        """
        df_regions = pd.read_excel(region_file) if idx == 0 else pd.read_csv(region_file)
        stats_df = df_values.groupby('index')['value'].agg(['min', 'max', 'mean', 'std', 'count']).reset_index()
        result_df = pd.merge(stats_df, df_regions, left_on='index', right_on='Index', how='left')
        total_count = result_df['count'].sum()
        result_df['percentage'] = result_df['count'] / total_count * 100
        result_df.drop(columns=['Index', 'index'], inplace=True)
        # if idx == 0:
        result_df = result_df[['Short.Name','count','percentage','min', 'max', 'mean', 'std']]
        result_df.rename(columns={"Short.Name": "AAL_Region_Names"}, inplace=True)
        # else:
        #     result_df.rename(columns={'AAL_region_name': 'Region'}, inplace=True)
        #     result_df = result_df[['AAL_Region_Names','count','percentage','min', 'max', 'mean', 'std']]
        result_df.sort_values(by='count', ascending=False, inplace=True)
        result_df.reset_index(drop=True, inplace=True)

        return result_df


    def output_selected_region(self, idx, fdr_path):
        """
        Generate a summary of selected regions based on the stability threshold and save this in a nii image.
        Inputs: 
        idx: The index for selecting the appropriate modality related parameters (images, coordinates, background images, and info files)

        Outputs: 
        The summary table of selected regions and saved nii image with selected regions annotated
        """
        # voxels_dir = "Voxel"
        # voxels_dir = "/scratch/jiankang_root/jiankang1/ellahe/Voxel"
        voxels_dir = fdr_path
        repetitions = self.rep

        # Select images, coordinates, background image and info file based on idx 
        if idx == 0:
            images = self.imgs[0]
            coordinates = self.coord[0]
            background_img = self.bg_image1
            info_file = self.region_info1
        else:
            images = self.imgs[1]
            coordinates = self.coord[1]
            background_img = self.bg_image2
            info_file = self.region_info2
            
        # Load mask and information using nibabel and pandas
        AAL_mask = nib.load(background_img)
        AAL_info = pd.read_excel(info_file) if idx == 0 else pd.read_csv(info_file)
        affine = AAL_mask.affine

        image_arr = np.zeros((repetitions, images.shape[1]))
        frequency = 1

        # Find voxels with frequency larger than threshold across different repetitions
        for seed in range(repetitions):
            hf = h5py.File(f"{voxels_dir}/Modality{idx}{seed}.hdf5", 'r')
            voxel = hf.get('voxel')["mod"+str(idx)][:,:]

            for j in range(image_arr.shape[1]):
                if voxel[:,j].sum() >= frequency:
                    image_arr[seed, j] = 1
        # use regions selected by whole data training as the threshold
        hf = h5py.File(f"{voxels_dir}/Modality_total{idx}{0}.hdf5", 'r')
        m = hf.get('voxel')["mod"+str(idx)][:,:]

        # Calculate each voxel's stability by comparing with voxel selected by training all data 
        stability = np.zeros((1, images.shape[1]))

        # Iterate through each voxel
        for i in range(images.shape[1]):
            if m[:,i] != 0:
                for seed in range(repetitions):
                    split = image_arr[seed,:]

                    if split[i] != 0:
                        stability[:,i] += 1
        stability /= repetitions
        print('sta',len(stability[stability>=0.4]))
        # Print statistical summary of stability scores
        print("Statistical Summary of Stability Scores:")
        print(f"Mean: {np.mean(stability)}")
        print(f"Median: {np.median(stability)}")
        print(f"Standard Deviation: {np.std(stability)}")
        print(f"Min: {np.min(stability)}")
        print(f"Max: {np.max(stability)}")

        # Plotting the distribution of stability scores
        plt.hist(stability.flatten(), bins=20, alpha=0.7)
        plt.title("Distribution of Stability Scores")
        plt.xlabel("Stability Score")
        plt.ylabel("Frequency")
        plt.show()
        stable_indices = np.where(stability >= self.thred)[1]

        # Get the indexes of stable and non-stable regions
        stable_vocab = []
        nonstable_vocab = []

        for i in range(stability.shape[1]):
            if stability[:,i] >= self.thred:
                stable_vocab.append(i)
            else:
                nonstable_vocab.append(i)

        stable_coord = []
        stable_values = []

        # Initialize a zero array with the same shape as coordinates
        image_mat = np.zeros((AAL_mask.get_fdata().shape[0], AAL_mask.get_fdata().shape[1], AAL_mask.get_fdata().shape[2]))
        image_mat = image_mat.astype(np.float32)

        # save the nii image with regions selected
        for i in range(len(stable_vocab)):
            xyz = coordinates[stable_vocab[i], :]
            if xyz[0] < image_mat.shape[0] and xyz[1] < image_mat.shape[1] and xyz[2] < image_mat.shape[2]:
                stable_coord.append(list(xyz))
                va = stability[0][stable_indices[i]]
                stable_values.append(va)
                image_mat[int(stable_coord[-1][0]),int(stable_coord[-1][1]),int(stable_coord[-1][2])] = float(stable_values[-1])

        if not os.path.exists(self.nii_path):
            os.makedirs(self.nii_path)
        save_path = f"{self.nii_path}/select_region_unit{idx}.nii.gz"
        nib.Nifti1Image(image_mat, affine).to_filename(save_path)

        # Get counts of each region in the AAL mask
        unique_counts = np.unique(AAL_mask.get_fdata(), return_counts=True)
        AAL_counts = pd.DataFrame({'region_name': AAL_info['Short.Name'], 'counts': unique_counts[1][1:]})

        # Apply MNI_coords function to each region
        MNI_coordinates = [self.find_MNI_coords(region, AAL_info, AAL_mask, idx) for region in AAL_info['Short.Name']]
        if idx==1:
            MNI_coordinates = MNI_coordinates[0]
        elif idx==0:
            MNI_coordinates = np.concatenate(MNI_coordinates, axis=0)

        affines = [3,3,3]
        offsets = [-90,-126,-72]

        # Apply affine transformation and calculate statistics
        transformed_coords, matched_indices, df = self.affine_transformation(idx, offsets, affines, stable_coord, pd.DataFrame(MNI_coordinates, columns=['x', 'y', 'z']), stable_values)
        result_df = self.calculate_stats_with_region(df, info_file, idx)
        result_df.to_csv(f"{self.nii_path}/result_df"+str(idx)+".csv", index=False)

        print(result_df)

