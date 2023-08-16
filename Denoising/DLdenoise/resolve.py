
import argparse, os
import glob
import numpy as np 
from skimage.transform import rescale, resize
from skimage.metrics import structural_similarity as compare_ssim
import natsort
import cv2

import torch
from torchvision.transforms import ToTensor
import util
import sys

import quant_util
import io_func

#Testing settings
parser = argparse.ArgumentParser(description='PyTorch application of trained weight on CT images')
parser.add_argument('--model-name','--m', type=str, default='cnn3', 
                    help='choose the network architecture name that you are going to use. Other options include redcnn, dncnn, unet, gan.')
parser.add_argument('--input-folder', type=str, required=True, help='directory name containing noisy input test images.')
parser.add_argument('--gt-folder', type=str, required=False, default="", help='directory name containing test Ground Truth images.')
parser.add_argument('--model-folder', type=str, required=True, help='directory name containing saved checkpoints.')
parser.add_argument('--output-folder', type=str, help='path to save the output results.')
parser.add_argument('--normalization-type', type=str, required=True, help='None or unity_independent. Look into img_pair_normalization in utils.')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--input-img-type', type=str, default='dicom', help='dicom or raw or tif?')
parser.add_argument('--specific-epoch', action='store_true', help='If true only one specific epoch based on the chckpt-no will be applied to \
                                                             test images. Else all checkpoints (or every saved checkpoints corresponding to each epoch)\
                                                             will be applied to test images.')
parser.add_argument('--chckpt-no', type=int, required=False, default=-1, help='epoch no. of the checkpoint to be loaded\
                                                                         and then applied to noisy images from the test set. Default is the last epoch.')
parser.add_argument('--se-plot', action='store_true', help='If true denoised images from test set is saved inside the output-folder.\
                                                      Else only test stats are saved in .txt format inside the output-folder.')
parser.add_argument('--in-dtype',  type=str, default="uint16", help="data type of input images. Only needed for .raw format imgs.")
# parser.add_argument('--out-dtype', type=str, default="uint16", help="data type to save de-noised output.")
# out-dtype option is not accurate right now. You need to have out-dtype set same as the in-dtype for now
parser.add_argument('--resolve-patient', action='store_true', help="is CNN applied to images from different patients? \
                                                                    If yes then images will be saved with patient tag.")
parser.add_argument('--resolve-nps', action='store_true', help="is CNN applied to water phantom images?")
parser.add_argument('--rNx', required=False, type=int,    default=None, help="image size for raw image as input.")

args = parser.parse_args()

print('\n----------------------------------------')
print('Command line arguments')
print('----------------------------------------')
for i in args.__dict__: print((i),':',args.__dict__[i])
print('\n----------------------------------------\n')

input_folder       = args.input_folder
gt_folder          = args.gt_folder
output_folder      = args.output_folder
model_folder       = args.model_folder
cuda               = args.cuda 
normalization_type = args.normalization_type
specific_epoch     = args.specific_epoch
chckpt_no          = args.chckpt_no
num_channels       = 1
gt_available       = bool((args.gt_folder).strip())
out_dtype          = args.in_dtype

if (specific_epoch == True and chckpt_no != -1): chckpt_no = chckpt_no-1

# =================================
# Importing model architecture:
# =================================
if args.model_name =='cnn3':
  from models.cnn3 import CNN3
  main_model = CNN3(num_channels=num_channels)
elif args.model_name =='redcnn':
  from models.redcnn import REDcnn10
  main_model = REDcnn10(idmaps=3)
elif args.model_name == 'unet':
  from models.unet import UDnCNN
  main_model = UDnCNN(D=10) # D is no of layers
elif args.model_name == 'dncnn':
  from models.dncnn import DnCNN
  main_model = DnCNN(channels=num_channels) # default: layers used is 17, bn=True
elif args.model_name == 'gan':
  from models.gan import Generator
  main_model = Generator(n_residual_blocks=16, upsample_factor=1, base_filter=64, num_channel=num_channels)
else:
  print("ERROR! Re-check DNN model (architecture) string!")
  sys.exit() 

def main():
    # importing model all the checkpoint NAMES saved in the training phase
    if args.model_name == 'gan':
        model_names = natsort.natsorted(glob.glob(os.path.join(model_folder, "checkpoint-gene*.*")))
    else:
        model_names = natsort.natsorted(glob.glob(os.path.join(model_folder, "*.*")))

    if (len(model_names)==0): sys.exit("ERROR ! No checkpoints or incorrect model path.\n")
    # =============================================================
    # Importing checkpoint paths & creating folders to save results
    # -------------------------------------------------------------
    if specific_epoch is True:
        fm_name = model_names[chckpt_no]
        model_names = []
        model_names.append(fm_name)

        #declaring and creating folders to store results if specific check point no is fed in 
        sp_str = model_names[0].split('/')
        sp_str = sp_str[-1]
        sp_str = sp_str.split('.')
        sp_str = sp_str[0]
        cnn_hd_test_out   = os.path.join(output_folder, sp_str)	
        if not os.path.isdir(cnn_hd_test_out): os.makedirs(cnn_hd_test_out)
        if gt_available: quant_fname = os.path.join(output_folder, sp_str+'_quant_vals.txt') 
    else:
        # when all checkpoints are used to give their respective quant results
        # we save only quant values and not the individual CNN based Hd image results
        if not os.path.isdir(output_folder): os.makedirs(output_folder, exist_ok=True)
        if gt_available: quant_fname = os.path.join(output_folder, 'all_checkpoint_quant_vals.txt')

    # gt data is available save global metrics to a txt file
    if gt_available:
        quantfile = open(quant_fname, '+w')	
        quantfile.write('chckpt-no, CNN rMSE, (+,-std), CNN PSNR [dB], (+,-std), CNN SSIM, (+,-std), LD rMSE, (+,-std), LD PSNR [dB], (+,-std), LD SSIM, (+,-std)\n')

    dir_min, dir_max = util.min_max_4rmdir(input_folder, args.input_img_type, args.in_dtype, rN=args.rNx)

    # ===================================
    # Accessing all (or one) checkpoints
    # ===================================
    for ith_model in range(len(model_names)):
        model = main_model
        model = model.eval()
        if cuda: model = model.cuda()
        checkpoint = torch.load(model_names[ith_model])
        model.load_state_dict(checkpoint['model'])

        #read images from input-folder
        lr_img_names = sorted(glob.glob(os.path.join(input_folder, "*.*")))
        if gt_available:
            gt_img_names = sorted(glob.glob(os.path.join(gt_folder, "*.*")))
            lr_rMSE_arr, lr_psnr_arr, lr_ssim_arr    = [], [], []
            cnn_rMSE_arr, cnn_psnr_arr, cnn_ssim_arr = [], [], []

        # ====================================
        # Denoising all LD images from Test Set
        # =====================================
        for i in range(len(lr_img_names)):
            if args.input_img_type=='dicom':
                lr_img = io_func.pydicom_imread(lr_img_names[i])
                if gt_available: gt_img = io_func.pydicom_imread(gt_img_names[i])
            elif args.input_img_type=='raw':
                lr_img = io_func.raw_imread(lr_img_names[i], (args.rNx, args.rNx), dtype=args.in_dtype)
                if gt_available: gt_img = io_func.raw_imread(gt_img_names[i], (args.rNx, args.rNx), dtype=args.in_dtype)
            else:
                lr_img = io_func.imageio_imread(lr_img_names[i])
                if gt_available: gt_img = io_func.imageio_imread(gt_img_names[i])

            if gt_available: gt_min, gt_max = np.min(gt_img), np.max(gt_img)
            lr_h, lr_w = lr_img.shape
            cnn_output = util.norm_n_apply_model_n_renorm(model, lr_img, dir_min, normalization_type, cuda, args.resolve_nps)
            lr_img     = lr_img.astype(out_dtype)
            cnn_output = cnn_output.astype(out_dtype)
            img_str = lr_img_names[i]
            if args.resolve_patient: 
                patient_str = img_str.split('/')[-3]
            img_str = img_str.split('/')[-1]
            img_no  = img_str.split('.')[-2]
            
            if (i==0): 
                    print('Per image stats:')
                    print('----------------------------------------\n')
            if gt_available:
                gt_img = gt_img.astype(out_dtype)
                cnn_max, cnn_min = max(np.max(gt_img), np.max(cnn_output)), min(np.min(gt_img), np.min(cnn_output))
                cnn_rMSE = quant_util.relative_mse(gt_img, cnn_output)
                cnn_psnr = quant_util.psnr(gt_img, cnn_output, cnn_max)
                cnn_ssim = compare_ssim(cnn_output.reshape(lr_h, lr_w, 1), gt_img.reshape(lr_h, lr_w, 1), multichannel=True, data_range=(cnn_max-cnn_min))
                cnn_rMSE_arr.append(cnn_rMSE)
                cnn_psnr_arr.append(cnn_psnr)
                cnn_ssim_arr.append(cnn_ssim)

                lr_max, lr_min = max(np.max(gt_img), np.max(lr_img)), min(np.min(gt_img), np.min(lr_img))
                lr_rMSE = quant_util.relative_mse(gt_img, lr_img)
                lr_psnr = quant_util.psnr(gt_img, lr_img, lr_max)
                lr_ssim = compare_ssim(lr_img.reshape(lr_h, lr_w, 1), gt_img.reshape(lr_h, lr_w, 1), multichannel=True, data_range=(lr_max-lr_min))
                lr_rMSE_arr.append(lr_rMSE)
                lr_psnr_arr.append(lr_psnr)
                lr_ssim_arr.append(lr_ssim)
                print("IMG: %s || avg CNN [rMSE: %.4f, PSNR: %.4f, SSIM: %.4f] || avg LD [rMSE: %.4f, PSNR: %.4f, SSIM: %.4f]"\
                %(img_str, cnn_rMSE, cnn_psnr, cnn_ssim, lr_rMSE, lr_psnr, lr_ssim))
            else:
                print("IMG: %s || OUT [min: %.4f, max: %.4f, img_type: %s] ||  IN [min: %.4f, max: %.4f, img_type: %s]"\
                %(img_str, np.min(cnn_output), np.max(cnn_output), cnn_output.dtype, np.min(lr_img), np.max(lr_img), lr_img.dtype))

            # ==========================================================		    
            # saving feed forward results from specific epoch (if true)
            # ==========================================================
            if (specific_epoch == True and args.se_plot ==True):
                if args.resolve_patient:
                    patient_dir = cnn_hd_test_out + '/' + patient_str
                    if not os.path.isdir(patient_dir): os.makedirs(patient_dir, exist_ok=True)
                    io_func.imsave_raw((cnn_output), patient_dir     + '/' + img_no + '.raw')
                else:
                    io_func.imsave_raw((cnn_output), cnn_hd_test_out + '/' + img_no + '.raw')
        # command line print + quantfile print
        if gt_available:
            # extract checkpoint no to print
            chckpt_name = model_names[ith_model]
            chckpt_name = chckpt_name.split('/')[-1]
            chckpt_name = chckpt_name.split('.')[0]
            prnt_chckpt_no = int(chckpt_name.split('-')[-1])
            print('\n------------------------------------------')
            print("%s (applied on test data)" % chckpt_name)
            print('------------------------------------------')

            print("avg CNN (std) [rMSE: %.4f (%.4f), PSNR: %.4f (%.4f), SSIM: %.4f (%.4f)] \navg LD  (std) [rMSE: %.4f (%.4f), PSNR: %.4f (%.4f), SSIM: %.4f (%.4f)]" % \
            (np.mean(cnn_rMSE_arr), np.std(cnn_rMSE_arr), np.mean(cnn_psnr_arr), np.std(cnn_psnr_arr), np.mean(cnn_ssim_arr), np.std(cnn_ssim_arr),\
            np.mean(lr_rMSE_arr), np.std(lr_rMSE_arr), np.mean(lr_psnr_arr), np.std(lr_psnr_arr), np.mean(lr_ssim_arr), np.std(lr_ssim_arr)))

            quantfile.write("%9d,%9.4f,%9.4f,%14.4f,%9.4f,%9.4f,%9.4f,%8.4f,%9.4f,%13.4f,%9.4f,%8.4f,%9.4f\n" \
            % (prnt_chckpt_no, np.mean(cnn_rMSE_arr), np.std(cnn_rMSE_arr), np.mean(cnn_psnr_arr), np.std(cnn_psnr_arr), np.mean(cnn_ssim_arr), np.std(cnn_ssim_arr),\
            np.mean(lr_rMSE_arr), np.std(lr_rMSE_arr), np.mean(lr_psnr_arr), np.std(lr_psnr_arr), np.mean(lr_ssim_arr), np.std(lr_ssim_arr)))
        del model

    if gt_available: quantfile.close()

if __name__ == "__main__":
    main()
