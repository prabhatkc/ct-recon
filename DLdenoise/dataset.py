from torch.utils.data import Dataset
import h5py
import torch 
import numpy as np 
import os
from torchvision.transforms import ToTensor
import natsort
import glob 
import cv2
import sys
import util
import io_func

#sys.path.append('/home/prabhat.kc/Implementations/python/')
#import global_files as gf 


class DatasetFromHdf5(Dataset):
    #def __init__(self, file_path):
    def __init__(self, hvd, file_path, mod_num=1, drop_style='remove_last'):
        super(DatasetFromHdf5, self).__init__()
        shuffle_patches=False
        # shuffling patches at the Sampler Distribution is more efficient
        # for h5 files. so let this options be False for this subroutine
        if os.path.isfile(file_path):
            if (os.path.exists(file_path)==True):
                hf = h5py.File(file_path, mode='r')
                self.data = hf.get("input")
                self.target = hf.get("target")
            else:
                if hvd.rank()==0:
                    print('\n-------------------------------------------------------------')
                    print("ERROR! No training/tuning h5 files. Re-check input data paths.")
                    print('--------------------------------------------------------------')
                    sys.exit()         
        # for multiple h5 files in a directory
        elif os.path.isdir(file_path):
            all_h5s = sorted(glob.glob(file_path+'/*'))
            if (len(all_h5s)==0 and hvd.rank()==0): 
                print('\n-------------------------------------------------------------')
                print("ERROR! No training/tuning h5 files. Re-check input data paths.")
                print('--------------------------------------------------------------')
                sys.exit()
            hf  = h5py.File(all_h5s[0], mode='r')
            dt  = hf.get('input')
            tgt = hf.get('target')
            _, _, h, w = dt.shape
            all_dt = np.empty([0, 1, h, w])
            all_tgt = np.empty([0, 1, h, w])
            
            all_dt = np.append(all_dt, dt, axis=0)
            all_tgt = np.append(all_tgt, tgt, axis=0)
            for i in range(1, len(all_h5s)):
                hf  = h5py.File(all_h5s[i], mode='r')
                dt  = hf.get('input')
                tgt = hf.get('target')
                all_dt = np.append(all_dt, dt, axis=0)
                all_tgt = np.append(all_tgt, tgt, axis=0)
            if shuffle_patches:
                Npatches = len(all_dt)
                shuffled_Npatches_arr = np.arange(Npatches)
                np.random.shuffle(shuffled_Npatches_arr)
                self.data = all_dt[shuffled_Npatches_arr, :, :, :]
                self.target = all_tgt[shuffled_Npatches_arr, :, :, :]
            else:
                self.data = all_dt
                self.target = all_tgt
        else:
            if hvd.rank()==0:
                    print('\n----------------------------------------------------------------------------')
                    print("ERROR! Issues related to training/tuning path. Re-check training-fname option.")
                    print('------------------------------------------------------------------------------')
                    sys.exit()
        if np.mod(self.data.shape[0], mod_num)!=0:
            if drop_style == 'remove_last':
                # this option removes last b patches where a (mod n) eq b
                remove_n    = np.mod(self.data.shape[0], mod_num)
                self.data   = self.data[:-remove_n, :, :, :]
                self.target = self.target[:-remove_n, :, :, :]
            else:
                # add_n_remove_last
                # this options first adds patches of len mod_num
                # these addition patches are randomly selected from 
                # the range [0, data.shape[0]. Then from the subsequent
                # generated array last b patches are removed where a (mod n ) eq b
                add_length = np.mod(self.data.shape[0], mod_num)
                add_ind    = np.random.randint(low=0, high=self.data.shape[0], size=add_length)
                add_ind    = np.sort(add_ind)
                print(add_length)
                extra_input  = self.data[add_ind, :, :, :]
                extra_target = self.target[add_ind, :, :, :]
                gf.multi2dplots(3, 3, extra_target[:, 0, :, :], 0)
                gf.multi2dplots(3, 3, extra_input[:, 0, :, :], 0)
                sys.exit()

    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        input = self.data[index,:,:,:]
        trgt = self.target[index,:,:,:]
        return(torch.from_numpy(input).float(), torch.from_numpy(trgt).float())

class DatasetFromNpz(Dataset):
#class DatasetFromHdf5(self, file_path):
    def __init__(self, file_path):
        super(DatasetFromNpz, self).__init__()
        hf = np.load(file_path)
        self.data = hf["data"]
        #self.target = self.data
        self.target = hf["label"]
        #hf.close()
        #self.filename=file_path
        
    def __getitem__(self, index):
        input = self.data[index,:,:,:]
        target = self.target[index,:,:,:]
        #sample = {'input': input, 'target': target}
        return input, target
    
    def __len__(self):
        return self.data.shape[0]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif"])

def interpolation_hr(norm_lr, scale):
    h, w, _ = norm_lr.shape
    #norm_hr = resize(norm_lr, (h *scale, w*scale), anti_aliasing=True)
    norm_hr = cv2.resize(norm_lr, (w*scale, h*scale),interpolation=cv2.INTER_AREA)
    return norm_hr

class DatasetfromFolder4rSRCNN(Dataset):
    def __init__(self, image_dir_lr, image_dir_hr, input_transform=None, target_transform=None):
        super(DatasetfromFolder4rSRCNN, self).__init__()
        self.input_fnames = natsort.natsorted(glob.glob(os.path.join(image_dir_lr, "*.*")))
        self.target_fnames = natsort.natsorted(glob.glob(os.path.join(image_dir_hr, "*.*")))    
        self.input_transform = input_transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.input_fnames)

    def __getitem__(self, index):
        input = io_func.imageio_imread(self.input_fnames[index])
        target = io_func.imageio_imread(self.target_fnames[index])
        
        input = io_func.normalize_data_ab(0, 1, input)
        #for SRCNN model
        input = interpolation_hr(input, 4)
        
        in_h, in_w, _ = input.shape
        img_to_tensor = ToTensor()
        input = img_to_tensor(input).view(-1, in_h, in_w)
        
        if self.input_transform:
            input =self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
            
        return((input).float(), torch.from_numpy(target).float())

class DatasetfromFolder(Dataset):
    def __init__(self, image_dir_lr, image_dir_hr, input_transform=None, target_transform=None):
        super(DatasetfromFolder, self).__init__()
        self.input_fnames = natsort.natsorted(glob.glob(os.path.join(image_dir_lr, "*.*")))
        self.target_fnames = natsort.natsorted(glob.glob(os.path.join(image_dir_hr, "*.*")))    
        self.input_transform = input_transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.input_fnames)

    def __getitem__(self, index):
        input = io_func.imageio_imread(self.input_fnames[index])
        target = io_func.imageio_imread(self.target_fnames[index])
        
        input = util.normalize_data_ab(0, 1, input)
        
        in_h, in_w, _ = input.shape

        img_to_tensor = ToTensor()
        input = img_to_tensor(input).view(-1, in_h, in_w)
        
        if self.input_transform:
            input =self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
            
        return((input).float(), torch.from_numpy(target).float())