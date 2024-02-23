import os
import glob
import utils
import numpy as np
import random
import h5py
import sys

import glob_funcs as gf

def makeh5patches(args):

	print('\n----------------------------------------')
	print('Command line arguements')
	print('----------------------------------------')
	for i in args.__dict__: print((i),':',args.__dict__[i])
	print('----------------------------------------')
	if args.img_format != 'dicom':
			print('WARNING. Ensure the img format for input-target pair and their sizes are acccurate in \n line 61 in file serial_h5patches.py.')
	# reading all the image paths for given patients
	all_dir_paths = sorted(glob.glob(args.input_folder+'/*/'))#/*/-> to enter sub-folders
	all_input_paths, all_target_paths = [], []
	
	# allocating arrays for input/target min/max
	pre_norm_in_min, pre_norm_in_max = [], []
	pre_norm_tar_min, pre_norm_tar_max = [], []
	post_norm_in_min, post_norm_in_max = [], []
	post_norm_tar_min, post_norm_tar_max = [], []
	
	random_ind = None
	for dir_paths in all_dir_paths:
		if args.random_N: random_ind = utils.get_sorted_random_ind(os.path.join(dir_paths, args.input_gen_folder), args.N_rand_imgs)		
		
		in_paths = utils.getimages4rmdir(os.path.join(dir_paths, args.input_gen_folder), random_ind)
		target_paths = utils.getimages4rmdir(os.path.join(dir_paths, args.target_gen_folder), random_ind)
		
		all_input_paths.extend(in_paths)
		all_target_paths.extend(target_paths)

	print('\nTraining input image paths:')
	print(np.asarray(all_input_paths))
	print('\n\nTraining target image paths:')
	print(np.asarray(all_target_paths))	
	#declaring null array for input & label to append later
	sub_input_of_all_inputs = np.empty([0, args.input_size, args.input_size, 1])
	sub_label_of_all_labels = np.empty([0, args.label_size, args.label_size, 1])

	#declaring path to save sanity check results
	sanity_chk_path = 'sanity_check/'+((args.input_folder).split('/'))[-1] + '/norm_' + str(args.normalization_type) + '_patch_size_' + str(args.patch_size)
	if not os.path.isdir(sanity_chk_path): os.makedirs(sanity_chk_path)

	# if the input is to be blurred and noise is to be added
	# get the label of the indices that is to be blurred and noised
	if args.blurr_n_noise: seed = utils.bn_seed(len(all_input_paths), 0.4, 0.4)
	else:				   sN = len(all_input_paths); seed = [None]*sN

	if (args.dose_blend): blend_fact_arr = np.random.uniform(0.5,1.2,size=len(all_target_paths))

	for i in range(len(all_input_paths)):
		
		input_image = gf.pydicom_imread(all_input_paths[i])
		target_image = gf.pydicom_imread(all_target_paths[i])
		#input_image = input_image[33:455]
		#target_image = target_image[33:455]
		
		if (input_image.shape != target_image. shape):
			print("MISMATCH in image size for \
				input: ", all_input_paths[i].split('/'), "& output: ", all_target_paths[i].split('/')[-1])
			print("Exiting the program")
			sys.exit()		
		if(i==0): 
			print('\nHere target images from training dataset is of type-', target_image.dtype,\
				  '. And is assigned as-', (target_image.astype('float32')).dtype,\
							'before network training')
			print("\nFirst image pair (target : input) in the raw stack (i.e. before patching)"\
				  " are of shapes {} : {}".format(target_image.shape, input_image.shape))
		
		target_image    = target_image.astype('float32')
		input_image     = input_image.astype('float32')
		if (args.air_threshold): target_image_un = target_image #used to for air thresholding

		pre_norm_in_min.append(np.min(input_image)); pre_norm_in_max.append(np.max(input_image))
		pre_norm_tar_min.append(np.min(target_image)); pre_norm_tar_max.append(np.max(target_image))

		# sp 	  = input_image.shape
		# if len(sp) == 3:
		# image = image[:, :, 0]

		if (args.dose_blend): blend_factor= blend_fact_arr[i]
		# ------------------
		# Data normalization
		# ------------------
		input_image, target_image = utils.img_pair_normalization(input_image, target_image, args.normalization_type)
		post_norm_in_min.append(np.min(input_image)); post_norm_in_max.append(np.max(input_image))
		post_norm_tar_min.append(np.min(target_image)); post_norm_tar_max.append(np.max(target_image))
		
		# -----------------
		# Data Augmentation
		# -----------------
		if args.ds_augment:
			# need to change image into uint type before augmentation
			# if Pil augmentation is used
			# image = (gf.normalize_data_ab(0, 255, image)).astype(np.uint8)
			# else no need 
			input_aug_images = utils.downsample_4r_augmentation(input_image)
			target_aug_images = utils.downsample_4r_augmentation(target_image)
			if (args.air_threshold): target_un_aug_images = utils.downsample_4r_augmentation(target_image_un)
			if (i==0): 
				print("\nDownscale based data augmentation is PERFORMED")
				print("Also, each input-target image pair is downscaled by", \
					 len(input_aug_images)-1,"different scaling factors due to downscale based augmentation")
		else:
			h, w = input_image.shape
			input_aug_images = np.reshape(input_image, (1, h, w))
			target_aug_images = np.reshape(target_image, (1, h, w))
			if (args.air_threshold): target_un_aug_images = np.reshape(target_image_un, (1, h, w))
			if(i==0): print("\nDownscale based data augmentation is NoT PERFORMED")
		
		# print(len(aug_images))
		# Now working on each augmented images
		for p in range(len(input_aug_images)): 

			#adding noise and downscaling the input images as instructed
			label_ = utils.modcrop(target_aug_images[p], args.scale)
			input_ = utils.modcrop(input_aug_images[p], args.scale)
			if (args.air_threshold):un_label_ = target_un_aug_images[p]

			if args.scale ==1: input_ = input_
			else:			   input_ = utils.interpolation_lr(input_, args.scale)

			if args.blurr_n_noise: cinput_ = utils.add_blurr_n_noise(input_, seed[i])
			else:				   cinput_ = input_
			# print('seed=', seed[i])
			# gf.plot2dlayers(cinput_, title='input')
			# gf.plot2dlayers(label_, title='target')

			sub_input, sub_label = utils.overlap_based_sub_images(args, cinput_, label_)
			
			if(args.air_threshold):
				_, sub_label_un = utils.overlap_based_sub_images(args, cinput_, un_label_)
				sub_input, sub_label = utils.air_thresholding(args, sub_input, sub_label, sub_label_un)

			augmented_input, augmented_label = sub_input, sub_label
			if (args.rot_augment): augmented_input, augmented_label = utils.rotation_based_augmentation(args, augmented_input, augmented_label)
			if (args.dose_blend):  augmented_input, augmented_label = utils.dose_blending_augmentation(args, augmented_input, augmented_label, blend_factor)
			#else:
			#	add_rot_input, add_rot_label = sub_input, sub_label
			sub_input_of_all_inputs = np.append(sub_input_of_all_inputs, augmented_input, axis=0)
			sub_label_of_all_labels = np.append(sub_label_of_all_labels, augmented_label, axis=0)	

		#gf.multi2dplots(4, 8, sub_input_of_all_inputs[0:66, :, :, 0], 0, passed_fig_att = {"colorbar": False, "figsize":[4*2, 4*2]})
		#gf.multi2dplots(4, 8, sub_label_of_all_labels[0:66, :, :, 0], 0, passed_fig_att = {"colorbar": False, "figsize":[4*2, 4*2]})
		#sys.exit()
	# --------------------------
	# Shuffling the patches     
	# --------------------------
	if args.shuffle_patches:
		Npatches = len(sub_input_of_all_inputs)
		shuffled_Npatches_arr = np.arange(Npatches)
		np.random.shuffle(shuffled_Npatches_arr)
		sub_input_of_all_inputs = sub_input_of_all_inputs[shuffled_Npatches_arr, :, :, :]
		sub_label_of_all_labels = sub_label_of_all_labels[shuffled_Npatches_arr, :, :, :]
	
	# -----------------------------------------------------
	# Sanity check
	# making patch plot of random patches for sanity check
	#------------------------------------------------------
	if args.sanity_plot_check:
		window = 12
		lr_N = len(sub_input_of_all_inputs)
		rand_num=random.sample(range(lr_N-window), 5)
		#print(sub_input_of_all_inputs.shape)
		#print(rand_num)
		#sys.exit()
		for k in range(len(rand_num)):
			s_ind  = rand_num[k]
			e_ind  =  s_ind+window
			lr_out_path = os.path.join(sanity_chk_path+'/lr_input_sub_img_rand_'+str(rand_num[k])+'.png')
			hr_out_path = os.path.join(sanity_chk_path+'/hr_input_sub_img_rand_'+str(rand_num[k])+'.png')
			gf.multi2dplots(3, 4, sub_input_of_all_inputs[s_ind:e_ind, :, :, 0], 0, passed_fig_att = {"colorbar": False, "figsize":[4,4], "out_path": lr_out_path})
			gf.multi2dplots(3, 4, sub_label_of_all_labels[s_ind:e_ind, :, :, 0], 0, passed_fig_att = {"colorbar": False, "figsize":[4*args.scale, 4*args.scale], "out_path": hr_out_path})
		
	# data format based on API used for network training
	# torch reads tensor as [batch_size, channels, height, width]
	# tensorflow reads tensor as [batch_size, height, width, channels]
	if args.tensor_format == 'torch':
		sub_input_of_all_inputs = np.transpose(sub_input_of_all_inputs, (0, 3, 1, 2))
		sub_label_of_all_labels = np.transpose(sub_label_of_all_labels, (0, 3, 1, 2))
	elif args.tensor_format == 'tf':
		sub_input_of_all_inputs = sub_input_of_all_inputs
		sub_label_of_all_labels = sub_label_of_all_labels
	
	# --------------------
	# creating h5 file
	#---------------------
	print("\n==>patch work completed. Now exporting the data to h5 files")
	print('   arr data type while incorperating various augmentation', sub_input_of_all_inputs.dtype, sub_label_of_all_labels.dtype)
	print('   patched data are saved as data-type:', args.out_dtype, 'in h5 files')
	

	if args.out_dtype=='int16' or args.out_dtype=='int':
		sdtype=np.int16
	elif args.out_dtype=='float' or args.out_dtype=='float64':
		sdtype=np.float64
	elif args.out_dtype=='float32':
		sdtype=np.float32
	elif args.out_dtype=='uint16':
		sdtype=np.uint16
	else:
		sdtype=np.float16
		
	output_folder = os.path.split(args.output_fname)[0]
	if not os.path.isdir(output_folder): os.makedirs(output_folder)
	nsplit=int(args.nsplit)
	if nsplit == 1:
		hf = h5py.File(args.output_fname, mode='w')
		hf.create_dataset('input', data=sub_input_of_all_inputs.astype(sdtype), dtype=sdtype)
		hf.create_dataset('target', data=sub_label_of_all_labels.astype(sdtype), dtype=sdtype)
		hf.close()
	else:
		split_dt=np.array_split(sub_input_of_all_inputs, nsplit, axis=0)
		split_tgt=np.array_split(sub_label_of_all_labels, nsplit, axis=0)
		for i in range(nsplit):
			hf = h5py.File(args.output_fname[:-3] + '_' + str(i) + '.h5', mode='w')
			hf.create_dataset('input', data=(split_dt[i]).astype(sdtype), dtype=sdtype)
			hf.create_dataset('target', data=(split_tgt[i]).astype(sdtype), dtype=sdtype)
			hf.close()

	print("\nshape of the overall input  subimages: {}".format(sub_input_of_all_inputs.shape))
	print("shape of the overall target subimages: {}".format(sub_label_of_all_labels.shape))
	print("\nFinally, due to data normalization based on:", args.normalization_type)
	print("input image range changes from (%.4f, %.4f) to (%.4f, %.4f)" % (min(pre_norm_in_min), max(pre_norm_in_max), min(post_norm_in_min), max(post_norm_in_max)))
	print("target image range changes from (%.4f, %.4f) to (%.4f, %.4f)" % (min(pre_norm_tar_min), max(pre_norm_tar_max), min(post_norm_tar_min), max(post_norm_tar_max)))
	#print('final sum of input, target is:', np.sum(sub_input_of_all_inputs.astype(args.out_dtype)), np.sum(sub_label_of_all_labels.astype(args.out_dtype)))
	
	