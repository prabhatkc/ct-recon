
import sys
import os
import numpy as np
import glob
import utils

import glob_funcs as gf
# import torch

def img_paths4rm_training_directory(args):
  # return paths of target images from input_folder with 
  # sub-directories. Each with target images for SR
  # if random_N is True: It returns random N images' paths

  all_dir_paths = sorted(glob.glob(args.input_folder+'/*/'))
  all_input_paths, all_target_paths = [], []
  random_ind = None
  
  for dir_paths in all_dir_paths:
    if args.random_N: random_ind = utils.get_sorted_random_ind(os.path.join(dir_paths, args.input_gen_folder), args.N_rand_imgs)

    in_paths = utils.getimages4rmdir(os.path.join(dir_paths, args.input_gen_folder), random_ind)
    all_input_paths.extend(in_paths)
    target_paths = utils.getimages4rmdir(os.path.join(dir_paths, args.target_gen_folder), random_ind)
    all_target_paths.extend(target_paths)
  
  return (np.asarray(all_input_paths), np.asarray(all_target_paths))

def partition_read_normalize_n_augment(args, bcasted_input_data, pid):
	chunck 		     = bcasted_input_data['chunck']
	all_input_paths  = bcasted_input_data['all_input_paths']
	all_target_paths = bcasted_input_data['all_target_paths']
	nproc 			 = bcasted_input_data['nproc']
	blend_fact_arr   = bcasted_input_data['blend_fact_arr']
	# partition trackers to transfer
	pre_norm_tar_min, pre_norm_tar_max = [], []
	post_norm_tar_min, post_norm_tar_max = [], []
	
	# declaring null array to store overall patches for a 
	# given chunk of dataset and a given pid
	each_rank_input_patch = np.empty([0, args.input_size, args.input_size, 1], dtype=args.dtype)
	each_rank_target_patch = np.empty([0, args.label_size, args.label_size, 1], dtype=args.dtype)	
	
	for j in range(chunck):
		if args.img_format == 'dicom':
			input_image  = gf.pydicom_imread(all_input_paths[pid*chunck+j])
			target_image = gf.pydicom_imread(all_target_paths[pid*chunck+j])
			input_image  = input_image[33:455]
			target_image = target_image[33:455]
		else:
			# to account for the fact that realistic dose simulation output were sized [424, 512]
			input_image  = gf.raw_imread(all_input_paths[pid*chunck+j], (424, 512), 'uint16')
			target_image = gf.pydicom_imread(all_target_paths[pid*chunck+j])
			target_image = target_image[31:455] 

		sp = target_image.shape

		# --------------------------------
		# Data channels & precision setup
		#---------------------------------
		if len(sp)==3:
			if(pid==0 and j==0): 
				print('\n==>Here target images have 3 colored channels but for training purposes we are only taking the first channel')
			target_image = (target_image[:, :, 0])
		
		if(pid==0 and j==0): 
			print('==>Here images from training dataset is of type-', target_image.dtype, end='.')
			print(' And is assigned as-', (target_image.astype(args.dtype)).dtype,'for the network training.')
			print('==>Here input images are sampled from', args.input_gen_folder, 'and are sized', input_image.shape, end='.')
	
		target_image = target_image.astype(args.dtype)
		input_image  = input_image.astype(args.dtype)
		# dummy place holder to for air thresholding
		if(args.air_threshold): target_image_un = target_image
		# get blending factor 
		# in case of no dodse augmentation blending factor is simply 0
		blend_factor=blend_fact_arr[pid*chunck+j]
			#print(pid, chunck, blend_factor)
		if (pid==0 and j==0): 
			print("\n==>First target image in the raw stack (i.e. before patching) is of shape", target_image.shape)
			print("   Input image is LD correspondance of its target image.")
			print('   For now SR model implements Normalization/Standardization independently to each pair or W.R.T target_image.')
		
		# -----------------------------------------------------
		# Data normalization & augmentation & air-thresholding
		# -----------------------------------------------------w
		pre_norm_tar_min.append(np.min(target_image)); pre_norm_tar_max.append(np.max(target_image))
		input_image, target_image = utils.img_pair_normalization(input_image, target_image, args.normalization_type)		
		post_norm_tar_min.append(np.min(target_image)); post_norm_tar_max.append(np.max(target_image))
		#sys.exit()
		if(args.air_threshold): t_input_patch, t_target_patch = augment_n_return_patch(args, input_image, target_image, j, pid, blend_factor, target_image_un)
		else:					t_input_patch, t_target_patch = augment_n_return_patch(args, input_image, target_image, j, pid, blend_factor)
		each_rank_input_patch  = np.append(each_rank_input_patch, t_input_patch, axis=0)
		each_rank_target_patch = np.append(each_rank_target_patch, t_target_patch, axis=0)

	#converting the stacked 3d image patch in to a 1d buffer
	#print('inside inner function: ranks is:', pid, 'its input patch size is', each_rank_input_patch.shape, '& target patch size is', each_rank_target_patch.shape )
	each_rank_input_patch_inbuff  = arrtobuff(each_rank_input_patch.astype(args.dtype))
	each_rank_target_patch_inbuff = arrtobuff(each_rank_target_patch.astype(args.dtype))

	#returning buffers
	partitioned_data ={'chunck_pre_norm_min':np.full(1, min(pre_norm_tar_min)),\
					   'chunck_pre_norm_max':np.full(1, max(pre_norm_tar_max)),\
					   'chunck_post_norm_min':np.full(1, min(post_norm_tar_min)),\
					   'chunck_post_norm_max':np.full(1, max(post_norm_tar_max)),\
					   'chunck_input_patch_inbuff':each_rank_input_patch_inbuff,\
					   'chunck_target_patch_inbuff':each_rank_target_patch_inbuff}	
	return(partitioned_data)
	
def augment_n_return_patch(args, input_image, target_image, i, pid, blend_factor, target_image_un=None):
	
	if args.ds_augment:
		#sys.exit()
		input_aug_images = utils.downsample_4r_augmentation(input_image)
		target_aug_images = utils.downsample_4r_augmentation(target_image)
		if (args.air_threshold): target_un_aug_images = utils.downsample_4r_augmentation(target_image_un)
		if (i==0 and pid==0): 
			print("\n==>Downscale based data augmentation is PERFORMED ...")
			print("   Also, each input-target image pair is downscaled by", len(input_aug_images)-1,"different scaling factors ...")
			print("   due to downscale based augmentation")
	else:
		h, w = input_image.shape
		input_aug_images = np.reshape(input_image, (1, h, w))
		target_aug_images = np.reshape(target_image, (1, h, w))
		if (args.air_threshold): target_un_aug_images = np.reshape(target_image_un, (1, h, w))
		if(i==0 and pid==0): print("\n==>Downscale based data augmentation is NoT PERFORMED")

	# declaring null array to append patches from augmented input & label later
	each_img_input_patch = np.empty([0, args.input_size, args.input_size, 1], dtype=args.dtype)
	each_img_target_patch = np.empty([0, args.label_size, args.label_size, 1], dtype=args.dtype)
	
	# Now working on each augmented images
	for p in range(len(input_aug_images)):
		label_ = (target_aug_images[p])
		input_ = (input_aug_images[p])
		if (args.air_threshold):un_label_ = target_un_aug_images[p]
		''' 
		#adding noise and downscaling the input images as instructed
		label_ = utils.modcrop(target_aug_images[p], args.scale)
		input_ = utils.modcrop(input_aug_images[p], args.scale)
		Add additional blurr ... bicubic init here 
		if args.scale ==1:
			cinput_ = input_
		else:
			cinput_ = utils.interpolation_lr(input_, args.scale)

		# if bicubic initilization is applied to input images
		# as in the case of SRCNN model
		if args.bicubic_init:
			cinput_ = utils.interpolation_hr(cinput_, args.scale)
		
	
		cinput_ = utils.add_blurr_n_noise(input_, seed[i])
		print('seed=', seed[i])
		if (pid==0 and i==0):
			gf.plot2dlayers(cinput_, title='input')
			gf.plot2dlayers(label_, title='target')
		'''
		cinput_ = input_
		sub_input, sub_label = utils.overlap_based_sub_images(args, cinput_, label_)
		
		if(args.air_threshold):
			_, sub_label_un = utils.overlap_based_sub_images(args, cinput_, un_label_) #cinput_ doesnot matter here
			sub_input, sub_label = utils.air_thresholding(args, sub_input, sub_label, sub_label_un)
		
		augmented_input, augmented_label = sub_input, sub_label

		if(args.rot_augment): augmented_input, augmented_label = utils.rotation_based_augmentation(args, augmented_input, augmented_label)
		if(args.dose_blend): augmented_input, augmented_label = utils.dose_blending_augmentation(args, augmented_input, augmented_label, blend_factor)
		each_img_input_patch = np.append(each_img_input_patch, augmented_input, axis=0)
		each_img_target_patch = np.append(each_img_target_patch, augmented_label, axis=0)

		#if args.rot_augment:
		#	add_rot_input, add_rot_label= utils.rotation_based_augmentation(args, sub_input, sub_label)
		#else:
		#	add_rot_input, add_rot_label = sub_input, sub_label
		#each_img_input_patch = np.append(each_img_input_patch, add_rot_input, axis=0)
		#each_img_target_patch = np.append(each_img_target_patch, add_rot_label, axis=0)
	
	'''
	if(i==0 and pid ==0):
		print('shape of first img from processor', pid, ':', each_img_input_patch.shape, each_img_target_patch.shape, each_img_input_patch.dtype, each_img_target_patch.dtype)
		gf.multi2dplots(4, 8, each_img_input_patch[0:66, :, :, 0], 0, passed_fig_att = {"colorbar": False, "figsize":[4*args.scale, 4*args.scale]})
		gf.multi2dplots(4, 8, each_img_target_patch[0:66, :, :, 0], 0, passed_fig_att = {"colorbar": False, "figsize":[4*args.scale, 4*args.scale]})
		sys.exit()
	'''
	return(each_img_input_patch, each_img_target_patch)
	
def arrtobuff(arr):
	buff = arr.ravel()
	return(buff)

def bufftoarr(buff, tot_element_count, ph, pw, pc):
 	pz = int(tot_element_count/(ph*pw))
 	arr = buff.reshape(pz, ph, pw, pc)
 	return(arr)
'''
def torch_shuffle(np_input, np_target):
	torch_input     = torch.from_numpy(np_input)
	torch_target    = torch.from_numpy(np_target)

	torch_input     = torch_input.to(device='cuda')
	torch_target    = torch_target.to(device='cuda')

	shuff_idx          = torch.randperm(torch_input.shape[0])
	shuff_torch_input  = torch_input[shuff_idx].view(torch_input.size())
	shuff_torch_target = torch_target[shuff_idx].view(torch_target.size())
	shuff_np_in        = shuff_torch_input.cpu().numpy()
	shuff_np_out       = shuff_torch_target.cpu().numpy()

	return (shuff_np_in, shuff_np_out)
'''
