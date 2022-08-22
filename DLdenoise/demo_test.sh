
#------------------------------------------------------------------------------------------------------------------#
#  									OPTIONS
#------------------------------------------------------------------------------------------------------------------#
<< COMMENT
usage: resolve_fld.py [-h] [--model-name MODEL_NAME] --input-folder
                      INPUT_FOLDER --gt-folder GT_FOLDER --model-folder
                      MODEL_FOLDER [--output-folder OUTPUT_FOLDER]
                      --normalization-type NORMALIZATION_TYPE [--cuda]
                      [--dicom-input] [--specific-epoch]
                      [--chckpt-no CHCKPT_NO] [--se-plot]

PyTorch application of trained weight on patient dicom images

command line arguments:
  -h, --help            show this help message and exit
  --model-name MODEL_NAME, --m MODEL_NAME
                        choose the network architecture name that you are
                        going to use. Options include redcnn, dncnn,
                        unet, three_layers.
  --input-folder INPUT_FOLDER
                        directory name containing noisy input test images.
  --gt-folder GT_FOLDER
                        directory name containing test Ground Truth images.
  --model-folder MODEL_FOLDER
                        directory name containing saved checkpoints.
  --output-folder OUTPUT_FOLDER
                        path to save the output results.
  --normalization-type NORMALIZATION_TYPE
                        normalization stipulated while training weights.
  --cuda                use cuda.
  --dicom-input         If input images are in dicom format.
  --specific-epoch      If true only one specific epoch based on the chckpt-no
                        will be applied to test images. Else all checkpoints
                        (or every saved checkpoints corresponding to each
                        epoch) will be applied to test images.
  --chckpt-no CHCKPT_NO
                        epoch no. of the checkpoint to be loaded to be applied
                        to noisy images from the test set. Default is the last
                        epoch
  --se-plot             If true denoised images from test set is saved inside
                        the output-folder. Else only test stats are saved in
                        .txt format inside the output-folder.
COMMENT


# ----------------------------------------------------#
# trained CNN3 weights applied on test set
# ----------------------------------------------------#

# on patient images
INPUT_FOLDER="../test_data/patient_data"
GT_FOLDER="../test_data/patient_data"
MODEL_FOLDER='./checkpoints/p55_no_norm/augFrFaTdF/three_layers/hvd_cpt_for_mse_l1_wd_0.0_lr_0.001_bs_128/'
OUTPUT_FOLDER='./results/patient_test/no_norm/p55_augFrFaTdF/three_layers/hvd_cpt_for_mse_l1_wd_0.0_lr_0.001_bs_128'
NORM_TYPE=None
python resolve_fld.py --m 'three_layers' --input-folder $INPUT_FOLDER --model-folder $MODEL_FOLDER \
 --gt-folder $GT_FOLDER --output-folder $OUTPUT_FOLDER --cuda  \
  --normalization-type $NORM_TYPE --dicom-input --specific-epoch --se-plot

#on CATPHAN
INPUT_FOLDER='../test_data/ctp404/'
MODEL_FOLDER='./checkpoints/p55_no_norm/augFrFaTdF/three_layers/hvd_cpt_for_mse_l1_wd_0.0_lr_0.001_bs_128/'
OUTPUT_FOLDER='./results/ctp404/no_norm/p55_augFrFaTdF/three_layers/hvd_cpt_for_mse_l1_wd_0.0_lr_0.001_bs_128'
NORM_TYPE='None'
python resolve_mtf_scaled.py --m 'three_layers' --input-folder $INPUT_FOLDER --model-folder $MODEL_FOLDER \
 --output-folder $OUTPUT_FOLDER --cuda  \
  --normalization-type $NORM_TYPE --input-img-type 'raw' --specific-epoch --se-plot

# on cylindrical water phantom
INPUT_FOLDER='../test_data/uniform_water_phan/fbp_sharp/'
MODEL_FOLDER='./checkpoints/p55_no_norm/augFrFaTdF/three_layers/hvd_cpt_for_mse_l1_wd_0.0_lr_0.001_bs_128/'
OUTPUT_FOLDER='./results/water/fbp_sharp/p55/three_layers_no_norm_augFrFaTdF/hvd_cpt_for_mse_l1_wd_0.0_lr_0.001_bs_128'
NORM_TYPE='None'
python resolve_nps.py --m 'three_layers' --input-folder $INPUT_FOLDER --model-folder $MODEL_FOLDER \
 --output-folder $OUTPUT_FOLDER --cuda  \
  --normalization-type $NORM_TYPE --input-img-type 'raw' --specific-epoch --se-plot 