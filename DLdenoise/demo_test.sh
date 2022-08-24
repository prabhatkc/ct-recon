
#------------------------------------------------------------------------------------------------------------------#
#  									OPTIONS
#------------------------------------------------------------------------------------------------------------------#
<< COMMENT
usage: resolve.py [-h] [--model-name MODEL_NAME] --input-folder INPUT_FOLDER [--gt-folder GT_FOLDER] --model-folder MODEL_FOLDER
                  [--output-folder OUTPUT_FOLDER] --normalization-type NORMALIZATION_TYPE [--cuda] [--input-img-type INPUT_IMG_TYPE] 
                  [--specific-epoch] [--chckpt-no CHCKPT_NO] [--se-plot] [--in-dtype IN_DTYPE] [--out-dtype OUT_DTYPE] [--resolve-nps]

PyTorch application of trained weight on CT images

optional arguments:
  -h, --help            show this help message and exit
  --model-name , --m    choose the network architecture name that you are
                        going to use. Other options include redcnn, dncnn,
                        unet, gan.
  --input-folder        directory name containing noisy input test images.
  --gt-folder           directory name containing test Ground Truth images.
  --model-folder        directory name containing saved checkpoints.
  --output-folder       path to save the output results.
  --normalization-type  None or unity_independent. Look into
                        img_pair_normalization in utils.
  --cuda                use cuda
  --input-img-type      dicom or raw or tif?
  --specific-epoch      If true only one specific epoch based on the chckpt-no
                        will be applied to test images. Else all checkpoints
                        (or every saved checkpoints corresponding to each
                        epoch) will be applied to test images.
  --chckpt-no           epoch no. of the checkpoint to be loaded and then
                        applied to noisy images from the test set. Default is
                        the last epoch.
  --se-plot             If true denoised images from test set is saved inside
                        the output-folder. Else only test stats are saved in
                        .txt format inside the output-folder.
  --in-dtype            data type to save de-noised output.
  --out-dtype           data type to save de-noised output.
  --resolve-nps         is CNN applied to water phantom images?

COMMENT
# ----------------------------------------------------#
# trained CNN3 weights applied on test set
# ----------------------------------------------------#

MODEL_FOLDER='./checkpoints/p55_no_norm/augTrTaTdT/cnn3/hvd_cpt_for_mse__wd_0.0_lr_1e-05_bs_64/'
MODEL_FOLDER='/gpfs_projects/prabhat.kc/lowdosect/transfers/transfers_4_spie/exps/exps/w8_exps_4_spie_dose/checkpoints/p55/augTrTaTdT/three_layers/hvd_cpt_for_mse_l1_wd_0.0_lr_0.001_bs_128/'
NORM_TYPE=None

# on patient images
set -f echo *
INPUT_FOLDER="./test_data/patient_data/*/quarter_3mm_sharp_sorted"
GT_FOLDER="./test_data/patient_data/*/full_3mm_sharp_sorted"
OUTPUT_FOLDER='./results/patient_test/cnn3'
python resolve.py --m 'cnn3' --input-folder $INPUT_FOLDER --model-folder $MODEL_FOLDER --gt-folder $GT_FOLDER \
--output-folder $OUTPUT_FOLDER --cuda --normalization-type $NORM_TYPE --input-img-type 'dicom' --specific-epoch --se-plot
set +f echo *

#on CATPHAN
INPUT_FOLDER='./test_data/ctp404/'
OUTPUT_FOLDER='./results/ctp404/cnn3'
python resolve.py --m 'cnn3' --input-folder $INPUT_FOLDER --model-folder $MODEL_FOLDER --output-folder $OUTPUT_FOLDER --cuda  \
--normalization-type $NORM_TYPE --input-img-type 'raw' --specific-epoch --se-plot --gt-folder '' --in-dtype 'int16' --out-dtype 'int16'

#on cylindrical phantom
INPUT_FOLDER='./test_data/uniform_water_phan/fbp_sharp/'
OUTPUT_FOLDER='./results/water/cnn3'
python resolve.py --m 'cnn3' --input-folder $INPUT_FOLDER --model-folder $MODEL_FOLDER --output-folder $OUTPUT_FOLDER --cuda \
--normalization-type $NORM_TYPE --input-img-type 'raw' --specific-epoch --se-plot --in-dtype 'int16' --out-dtype 'int16' --resolve-nps