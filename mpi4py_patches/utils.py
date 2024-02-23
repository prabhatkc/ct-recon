
import os 
import glob
import numpy as np 
import random 
import cv2
import sys

import glob_funcs as gf
from random import randrange

# cv2 intakes only float32 data types
# neg values dues to dose augmentation is accounted
# rotation or ds augmentation does not yield neg values
def modcrop(image, scale=3):
  """ to ensure that transition between HR to LR 
  and vice-versa is divisible by the scaling 
  factor without any remainder
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def get_sorted_random_ind(foldername, Nimgs):
    data_dir = os.path.join(os.getcwd(), foldername)
    images = sorted(glob.glob(os.path.join(foldername, "*.*")))
    randind = sorted(random.sample(range(len(images)), int(Nimgs)))
    return(randind)

def getimages4rmdir(foldername, randN=None):
  # sorted is true by default to remain consistant 
  data_dir = os.path.join(os.getcwd(), foldername)
  images = sorted(glob.glob(os.path.join(data_dir, "*.*")))

  if (randN !=None):
    images = np.array(images)
    images = list(images[randN]) 
  return images

def downsample_4r_augmentation(initial_image):
    
    #aug_ds_facs = np.asarray([0.9, 0.8, 0.7, 0.6])
    # cv2 only works for float type 32 
    aug_ds_facs = np.asarray([0.8, 0.6])
    aug_ds_input = []
    h, w = initial_image.shape
    #print(initial_image.shape)
    #print(h, w)
    #sys.exit()
    aug_ds_input.append(initial_image)
    for i in range(len(aug_ds_facs)):
        scale = aug_ds_facs[i]
        aug_label = cv2.resize(initial_image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        #sys.exit()
        aug_ds_input.append(aug_label)
    return(aug_ds_input)

def bn_seed(Nlen, bratio, nratio):

  """ returns a three valued (1 or 2 or 3) array with length (Nlen).
  bratio*Nlen will receive first label as 1 and the associated images
  will be blurred. Likewise nratio*Nlen will receive second label as 2
  and the corresponding images will be noised.
  """

  blist           = random.sample(range(Nlen), int(bratio*Nlen))
  rlist           = list(set(range(Nlen))-set(blist))
  nlist           = random.sample(rlist, int(nratio*Nlen))
  rlist           = list(set(rlist) - set(nlist))
  bn_seed         = np.zeros(Nlen)
  bn_seed[blist]  = 1
  bn_seed[nlist]  = 2
  bn_seed[rlist]  = 3
  return(bn_seed)

def interpolation_hr(norm_lr, scale):
  h, w = norm_lr.shape
  #norm_hr = resize(norm_lr, (h *scale, w*scale), anti_aliasing=True)
  norm_hr = cv2.resize(norm_lr, (w*scale, h*scale), interpolation=cv2.INTER_AREA)
  return norm_hr 

def interpolation_lr(hr_image, scale):

  h, w = hr_image.shape
  lr = cv2.resize(hr_image, (int(w*(1/scale)), int(h*(1/scale))), interpolation=cv2.INTER_AREA)
  return(lr)

def add_blurr_n_noise(input_, seed):

  if (seed == 1):
    #adding blurr
    corrupted    = cv2.blur(input_, (3,3))
 
  elif (seed ==2):
    # adding noise
    h, w         = input_.shape
    mu           = 0.0
    sigma        = 8.0
    # noise        = np.random.normal(mu, sigma, (h, w))
    # corrupted    = input_ + noise
    corrupted    = gf.add_rnl_white(0.01, input_, mu, sigma)
    corrupted    = gf.normalize_data_ab(np.min(input_), np.max(input_), corrupted)

  else:
    # doing nothing 
    corrupted    = input_
  return(corrupted)

def overlap_based_sub_images(config, input_, label_):

  image_size, label_size, lr_stride, scale, lr_padding = config.input_size, config.label_size, config.lr_stride, config.scale, config.lr_padding 
  hr_padding = lr_padding*scale

  ih, iw = input_.shape
  lh, lw = label_.shape
  sub_input_of_one_input, sub_label_of_one_label = [], []
  for x in range(0, ih, lr_stride):
      if(x==0):
          L_xs  = x
          L_xe  = x + (image_size )
      
          H_xs  = x
          H_xe  = x + (label_size )
      else:    
          L_xs  = x - lr_padding
          L_xe  = (x - lr_padding) + image_size
      
          H_xs  = x*scale -hr_padding
          H_xe  = (x*scale -hr_padding) + label_size
      
      if (L_xe >=ih):
          L_xe = ih-1
      if (H_xe >=lh):
          H_xe =lh -1
          
      #print("Lr: xs, xe: ", L_xs, L_xe, "\t\thr: xs, xe: ", H_xs,  H_xe)
      for y in range(0, iw, lr_stride):
      
          if(y==0):
              L_ys = y
              L_ye = y + (image_size )
              
              H_ys = y
              H_ye = y + (label_size )
          else:    
              L_ys = y - lr_padding
              L_ye = (y - lr_padding) + image_size 
          
              H_ys = y*scale - hr_padding
              H_ye = (y*scale - hr_padding) + label_size        
          
          if (L_ye >=iw):
              L_ye = iw - 1
          if (H_ye >=lw):
              H_ye = lw -1
              
          #print("\tLr: ys, ye: ", L_ys,  L_ye, "\t\tHr: ys, ye: ",  H_ys,  H_ye)
          sub_input = input_[L_xs : L_xe, L_ys : L_ye]
          sub_label = label_[H_xs : H_xe, H_ys : H_ye]
          
          if (sub_input.shape!=(image_size, image_size)):
              sih, siw = sub_input.shape
              sub_input = np.pad(sub_input,((0, image_size-sih),(0,image_size-siw)) , 'constant')
              #print('shape not eq', sub_input.shape)
          
          if (sub_label.shape!=(label_size, label_size)):
              slh, slw = sub_label.shape
              sub_label = np.pad(sub_label,((0, label_size-slh),(0,label_size-slw)) , 'constant')
              #print('label shape not eq', sub_label.shape)
          
          sub_input = sub_input.reshape([image_size, image_size, 1])
          sub_label = sub_label.reshape([label_size, label_size, 1])
          
          sub_input_of_one_input.append(sub_input)
          sub_label_of_one_label.append(sub_label)

  return(np.asarray(sub_input_of_one_input), np.asarray(sub_label_of_one_label))

def rotation_based_augmentation(args, sub_input, sub_label):
  #-----------------------------------
  # ROTATION BASED DATA AUGMENTATION       
  #----------------------------------
  # in addition to the regular upright positioned
  # images, we also include the same image
  # rotated by 90,180 , 270 degs or flippped 
  # LR or UD in our training
  # instances
  # sub_input is of shape [chunk_size, input_size, input_size, 1]
  rotated_all_inputs = np.empty([0, args.input_size, args.input_size, 1])
  rotated_all_labels =   np.empty([0, args.label_size, args.label_size, 1])
  
  #print("inside rotation subroutine", sub_input.shape, sub_label.shape)
  #sys.exit()
  for exp in range(3):
    if exp==0:
        add_rot_input = sub_input
        add_rot_label = sub_label
    elif exp==1:
        # rotates given patch either by 90 deg or 180 deg or 270 deg
        k = randrange(3)+1
        add_rot_input = np.rot90(sub_input, k, (1,2))
        add_rot_label = np.rot90(sub_label, k, (1,2))
    elif exp==2:
        # flips either the Patch either LR or UP
        k = randrange(2)
        add_rot_input = sub_input[:,::-1] if k==0 else sub_input[:,:,::-1]
        add_rot_label = sub_label[:,::-1] if k==0 else sub_label[:,:,::-1]      

    rotated_all_inputs =  np.append(rotated_all_inputs, add_rot_input, axis=0)
    rotated_all_labels =  np.append(rotated_all_labels, add_rot_label, axis=0)

  return(rotated_all_inputs, rotated_all_labels)

def dose_blending_augmentation(args, sub_input, sub_label, blend_factor):
  # dose blending can cause certain patches to exhibit negative values
  # Therefore minimum of these neg values should be added as (-min(patch))
  # to the corresponding input and label patches
  blended_all_inputs = np.empty([0, args.input_size, args.input_size, 1])
  blended_all_labels = np.empty([0, args.label_size, args.label_size, 1])

  for exp in range(2):
    if exp==0:
        add_blend_input = sub_input
        add_blend_label = sub_label
    elif exp==1:
        add_blend_label = sub_label
        add_blend_input = sub_label + blend_factor*(sub_input-sub_label)
        # blending might include neg values for certain patches
        # so adding the neg of those neg values for each of these patches in input as well as label
        if (np.min(add_blend_input) < 0.0):
          neg_ind = np.where(add_blend_input < 0.0)
          neg_ind_unq_ax0 = np.unique(neg_ind[0])
          # get negative values for each patch w.r.t axis 0
          neg_vals_arr = add_blend_input[neg_ind_unq_ax0, :, :, :]
          min_neg_vals = np.squeeze(np.min(np.min(neg_vals_arr, axis=1), axis=1))
          for i in range(len(neg_ind_unq_ax0)):
            add_blend_input[neg_ind_unq_ax0[i]] = add_blend_input[neg_ind_unq_ax0[i]] + (-min_neg_vals[i])
            add_blend_label[neg_ind_unq_ax0[i]] = add_blend_label[neg_ind_unq_ax0[i]] + (-min_neg_vals[i])
          #print("inside neg rotation subroutine", add_blend_input.shape, add_blend_label.shape, (np.min(add_blend_input)), np.min(add_blend_label))
          #sys.exit()
    
    blended_all_inputs = np.append(blended_all_inputs, add_blend_input, axis=0)
    blended_all_labels = np.append(blended_all_labels, add_blend_label, axis=0)
  return(blended_all_inputs, blended_all_labels)
    
def img_pair_normalization(input_image, target_image, normalization_type=None):
  
  if normalization_type == 'unity_independent':
    # both LD-HD pair are independently normalized to from (min_val, max_val) to (0, 1)
    out_input_image  = gf.normalize_data_ab(0.0, 1.0, input_image)
    out_target_image = gf.normalize_data_ab(0.0, 1.0, target_image)
    
  elif normalization_type == 'max_val_independent':
    # both LD-HD pair are independently normalized to from (min_val, max_val) to (min_val/max_val, min_val/max_val)
    if np.min(input_image)<0: input_image += (-np.min(input_image))
    if np.min(target_image)<0: target_image += (-np.min(target_image))
    out_input_image  =input_image/np.max(input_image)
    out_target_image =target_image/np.max(target_image)

  elif normalization_type == 'unity_wrt_ld':
    # both LD-HD pair are normalized to from (min_val, max_val) to (0, 1) with range (LD_max_val - LD_min_val)
    out_target_image = gf.normalize_data_ab_cd(0.0, 1.0, np.min(input_image), np.max(input_image), target_image)
    out_input_image  = gf.normalize_data_ab(0.0, 1.0, input_image)

  elif normalization_type == 'max_val_wrt_ld':
    # both LD-HD pair is normalized to from (min_val, max_val) to (min_val/LD_max_val, min_val/LD_max_val) 
    if np.min(input_image)<0: input_image += (-np.min(input_image))
    if np.min(target_image)<0: target_image += (-np.min(target_image))
    out_target_image =target_image/np.max(input_image)
    out_input_image =input_image/np.max(input_image)
  
  elif normalization_type == 'std_independent':
    # LD-HD pair is independently stardarized based on their respective values
    out_input_image = (input_image - np.mean(input_image))/(np.max(input_image)-np.min(input_image))
    out_target_image = (target_image - np.mean(target_image))/(np.max(target_image)-np.min(target_image))

  elif normalization_type == 'std_wrt_ld':
    # LD-HD pair is jointly stardarized based on LD values 
    out_target_image = (target_image - np.mean(input_image))/(np.max(input_image)-np.min(input_image))
    out_input_image = (input_image - np.mean(input_image))/(np.max(input_image)-np.min(input_image))
  
  # for dicom unity  as well as dicom_std everything has been scaled between (0, 2^16) hence there
  # is only one option for both and not the independent and wrt_ld types 
  elif normalization_type == 'dicom_unity':
    # all LD-HD pair are normalized between (0,1) while considering they exhibit (0, 2^16) initial value range
    out_input_image = gf.normalize_data_ab_cd(0.0, 1.0, 0.0, 2.0**12, input_image)
    out_target_image = gf.normalize_data_ab_cd(0.0, 1.0, 0.0, 2.0**12, target_image)
  
  elif normalization_type == 'dicom_std':
    # all LD-HD pair are standarized while considering they exhibit (0, 2^16) initial value range
    out_input_image = (input_image - np.mean(input_image))/(2.0**12)
    out_target_image = (target_image - np.mean(target_image))/(2.0**12)

  elif normalization_type =='set_max_val':
    out_input_image = input_image/4094.0
    out_target_image = target_image/4094.0
 
  else:
    out_input_image = input_image
    out_target_image = target_image

  return(out_input_image, out_target_image)

def air_thresholding(args, sub_input, sub_label, sub_label_un):

  pix_thresh = 200
  count_thresh = int(0.1* (args.input_size**2))
  mean_thresh = 100
  threshold_pass_ind = []
  for i in range(len(sub_label_un)):
    patch_orig = sub_label_un[i]
    num_pix_above_thresh = np.sum(patch_orig > pix_thresh)
    
    if (num_pix_above_thresh > count_thresh) & (patch_orig.mean() > mean_thresh):
      threshold_pass_ind.append(i)

  threshold_pass_ind=np.asarray(threshold_pass_ind).astype('int')
  return(sub_input[threshold_pass_ind], sub_label[threshold_pass_ind])