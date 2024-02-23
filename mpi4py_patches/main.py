import argparse
import sys

# Command line arguments
parser = argparse.ArgumentParser(description='Storing input-target images as patches in h5 format from all patient data / category sets')
parser.add_argument('--input-folder',  type=str, required=True, help='directory name containing images')
parser.add_argument('--output-fname',  type=str, default='results/patched_out.h5', help='output filename to save patched h5 file')
parser.add_argument('--patch-size',    type=str, default='p55', help="p96 or p75 or p55 or p42 or p24 or p12. p96 yields 96x96 patched window")
parser.add_argument('--normalization-type', type=str, required=False, default=None,
					help='None or unity_independent or unity_wrt_ld or std_independent or std_wrt_ld \
					 or None, for more info look at function img_pair_normalization in utils.py')
parser.add_argument('--tensor-format', type=str, required=False, default='torch',
					help='other option is tf. Depending upon the DL API tool,  h5 input and target patches \
					are saved accordingly. Eg torch tensor [batch_size, c, h, w]')
parser.add_argument('--random_N', 	     action="store_true", help="extracts random N complimentary images from \
					input - target folders. For more info refer to in-built options")
parser.add_argument('--rot_augment', 	 action='store_true', help='employs rotation-based augmentation')
parser.add_argument('--ds_augment',      action="store_true", help="incorperate downscale based data augmentation")
parser.add_argument('--air_threshold',   action='store_true', help='removes patches devoid of contrast')
parser.add_argument('--blurr_n_noise',   action='store_true', help="whether or not you want to add noise and blurr input data. \
																	Non-funtional in for the mpi-run. Only works in serial run (for now).")
parser.add_argument('--mpi_run', 		 action='store_true', help='if you want to employ mpi-based parallel computation')
parser.add_argument('--dose_blend', 	 action='store_true', help='if you want to employ dose blend-base data augmendation')
parser.add_argument('--sanity_plot_check', 	 action='store_true', help='if you want to view some of the patched plots')
parser.add_argument('--nsplit', type=int, default=1, help='no. of h5 files containing n chunks of patches')
parser.add_argument('--out-dtype', type=str, default='float16', help="array type of output h5 file. Options include \
				 	float32, float64, int16, uint16.")
parser.add_argument('--input-gen-folder', type=str, default='quarter_3mm_sharp_sorted', help="folder name containing noisy (input) measurements")
parser.add_argument('--target-gen-folder', type=str, default='full_3mm_sharp_sorted', help="folder name containing clean (target) measurements")
parser.add_argument('--img-format', type=str, default='dicom', help='image format for input and target images. Dicom/raw/tif?')
parser.add_argument('--shuffle-patches', type=str, default=None, help='options include np_shuffle or none')

if __name__ == '__main__':

	args = parser.parse_args()
	#-------------------------------------------------------------------------------------------------------------------------
	# in-built additional options
	#-------------------------------------------------------------------------------------------------------------------------
	args.scale                      = 1 # for now all low dose and high dose complimentary pairs are of same scale and no upscaling/downscaling is required
	padding_options                 = {'p96':12, 'p75':11, 'p64':4, 'p55':8, 'p42':6, 'p32':4, 'p24':4, 'p12':4}
	args.lr_padding                 = padding_options[args.patch_size]
	patch_option                    = {'p96':[84 + args.lr_padding, 84 + args.lr_padding], 'p75':[64 + args.lr_padding, 64 + args.lr_padding], \
									   'p64':[60 + args.lr_padding, 60 + args.lr_padding], \
									   'p55':[47 + args.lr_padding, 47 + args.lr_padding],  \
									   'p42':[36 + args.lr_padding, 36 + args.lr_padding], 'p32':[28 + args.lr_padding, 28 + args.lr_padding],\
									   'p24':[20+args.lr_padding, 20+args.lr_padding], 'p12':[8+args.lr_padding, 8+args.lr_padding]} 
	args.input_size, args.label_size= patch_option[args.patch_size]
	args.lr_stride                  = int(args.input_size - args.lr_padding)
	args.channel                    = 1
	
	# the array type below is the precision type that MPI performs its operations after the read of data
	# and up until saving the h5 patches. There is not much flexibity in changing the data type for 
	# the MPI operations. Have a look at https://pages.tacc.utexas.edu/~eijkhout/pcse/html/mpi-data.html#Python
	args.dtype                      = 'float32' 

	# number of random images from each sub-folders 
	# used to formulate the h5 training set instead of all the images from
	# all input-target subfolders
	if args.random_N: args.N_rand_imgs = 7 
	if (args.nsplit <=0):
		print('nsplit or no. of h5 files must be a positive integer')
		sys.exit()
	if(args.mpi_run==True): from mpi_h5patches import makeh5patches_in_mpi; makeh5patches_in_mpi(args)
	else:                   from serial_h5patches import makeh5patches; makeh5patches(args)
	
	