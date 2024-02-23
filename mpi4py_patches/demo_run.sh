<< COMMENT
#------------------------------------------------------------------------------------------------------------------#
#  									OPTIONS
#------------------------------------------------------------------------------------------------------------------#
usage: main.py [-h] --input-folder INPUT_FOLDER [--output-fname OUTPUT_FNAME] [--patch-size PATCH_SIZE]
               [--normalization-type NORMALIZATION_TYPE] [--tensor-format TENSOR_FORMAT] [--random_N] [--rot_augment]
               [--ds_augment] [--air_threshold] [--blurr_n_noise] [--mpi_run] [--dose_blend] [--sanity_plot_check] 
               [--nsplit NSPLIT] [--out-dtype OUT_DTYPE] [--input_gen_folder INPUT_GEN_FOLDER]
               [--target_gen_folder TARGET_GEN_FOLDER] [--img-format IMG_FORMAT]

Storing input-target images as patches in h5 format from all patient data /
category sets

optional arguments:
  -h, --help            show this help message and exit
  --input-folder        directory name containing images
  --output-fname        output filename to save patched h5 file
  --patch-size          p96 or p75 or p55 or p42 or p24 or p12. p96 yields 96x96 patched window
  --normalization-type  None or unity_independent or unity_wrt_ld or std_independent or std_wrt_ld. 
                        For more info look at function img_pair_normalization in utils.py
  --tensor-format       other option is tf. Depending upon the DL API tool, h5 input and target patches are saved accordingly. 
                        Eg. torch tensor [batch_size, c, h, w]
  --random_N            extracts random N complimentary images from input - target folders. For more info refer to 
                        in-built options.
  --rot_augment         employs rotation-based augmentation
  --ds_augment          incorperate downscale based data augmentation
  --air_threshold       removes patches devoid of contrast
  --blurr_n_noise       whether or not you want to add noise and blurr input data. Non-funtional in for the mpi-run. 
                        Only works in serial run (for now).
  --mpi_run             if you want to employ mpi-based parallel computation
  --dose_blend          if you want to employ dose blend-base data
                        augmendation
  --sanity_plot_check   if you want to view some of the patched plots
  --nsplit              no. of h5 files containing n chunks of patches
  --out-dtype           array type of output h5 file. Options include float32
  --input-gen-folder    folder name containing noisy (input) measurements
  --target-gen-folder   folder name containing clean (target) measurements
  --img-format          image format for input and target images.
  --shuffle-patches     options include np_shuffle or none
#------------------------------------------------------------------------------------------------------------------#
#                   WARNINGS
#------------------------------------------------------------------------------------------------------------------#
(1) If your training data size is too big i.e. in the order of 10s or 100s of GB, 
    you may incur segmentation faults as the data array size might become larger than what your machine/processor 
    can process at a given time. So it is advised that you split your training raw data as Train_data1, Train_data2, 
    Train_data3 or so on such that each Train_data3 is less than 5 GB or your machine's each processor's threshold/memory 
    space.
    
(2) If you opt for normalization other than "None", do not use out-dtype such as int16 or uint16. 


COMMENT
# seriel run
# OUTPUT_FNAME='./serial_patches/val_patches.h5'
# python main.py --input-folder 'raw_data' --output-fname $OUTPUT_FNAME --patch-size 'p55' --air_threshold --out-dtype 'float32' \
# --sanity_plot_check --air_threshold --ds_augment --rot_augment --dose_blend --nsplit 2

# mpi run
# here mpiexec -n 4 means that the code is going to use 4 processors  

# ----------------------------------------------------------
# direct patching without any shuffling
# ----------------------------------------------------------
OUTPUT_FNAME='./mpi_patches/p55_val_patches.h5'
mpiexec -n 4 python main.py --input-folder 'raw_data' --output-fname $OUTPUT_FNAME --patch-size 'p55' --out-dtype 'float16' \
--air_threshold --ds_augment --rot_augment --mpi_run --sanity_plot_check --dose_blend --nsplit 2 \
--input-gen-folder 'quarter_3mm_sharp_sorted' --target-gen-folder 'full_3mm_sharp_sorted' --shuffle-patches 'torch_shuffle'

OUTPUT_FNAME='./mpi_patches/p96_val_patches.h5'
mpiexec -n 4 python main.py --input-folder 'raw_data' --output-fname $OUTPUT_FNAME --patch-size 'p96' --out-dtype 'float16' \
--air_threshold --ds_augment --rot_augment --mpi_run --sanity_plot_check --dose_blend --nsplit 2 \
--input-gen-folder 'quarter_3mm_sharp_sorted' --target-gen-folder 'full_3mm_sharp_sorted'

# ----------------------------------------------------------
# for mixed training 
# ----------------------------------------------------------
# following executes shuffling scans at patient level
# ----------------------------------------------------------
# remove dummy place holder files from previous executions (if any)
rm -r raw_data_mixed_cp
rm -r pre_randomize
rm -r placeholder
python pre_patch_scan_shuffle.py --input-folder 'raw_data_mixed' --output-folder 'pre_randomize' --nsplit 2

# ----------------------------------------------------------
# following executes shuffling scans at patient level
# ----------------------------------------------------------
OUTPUT_FNAME='./mpi_patches/part_0_shuffled.h5'
mpiexec -n 1 python main.py --input-folder 'pre_randomize/part_0' --output-fname $OUTPUT_FNAME --patch-size 'p55' --out-dtype 'float16' \
--air_threshold --ds_augment --rot_augment --mpi_run --sanity_plot_check --dose_blend --nsplit 2 \
--input-gen-folder 'quarter_3mm' --target-gen-folder 'full_3mm' --shuffle-patches 'np_shuffle'

# time check 
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=6
# OUTPUT_FNAME='./mpi_patches/torch_shuffled/trainA.h5'
# INPUT_FOLDER='/gpfs_projects/prabhat.kc/lowdosect/data/clin_LDCT_post_40train_partA'
# time mpiexec -n 4 python main.py --input-folder $INPUT_FOLDER --output-fname $OUTPUT_FNAME \
# --patch-size 'p55' --rot_augment --ds_augment --air_threshold --mpi_run --sanity_plot_check --dose_blend --nsplit 9 \
# --input-gen-folder 'quarter_3mm_sharp_sorted' --target-gen-folder 'full_3mm_sharp_sorted' --shuffle-patches 'np_shuffle'
# for trainB -> np takes 8m48.754s for train a it takes 