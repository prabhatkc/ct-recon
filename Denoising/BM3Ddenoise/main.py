import argparse
import sys

# ====================================================================
# Command line arguments
# ====================================================================
parser = argparse.ArgumentParser(description='Application of bm3d-based denoisers on CT images')
parser.add_argument('--input-folder',type=str, required=True, help='directory name containing low dose images at different dose levels')
parser.add_argument('--input-gen-folder', type=str, default ='quarter_3mm_sharp_sorted', 
                                                       help ="directory name containing specific dose-level images for low dose")
parser.add_argument('--target-gen-folder', type=str, default='', help="directory name containing Full dose imgs(if available).\
                                                                       This option is inactivate for test cases where FD is unavailable.")
parser.add_argument('--output-folder',type=str, default='./results', help='main output foldername to store results')
parser.add_argument('--input-img-type', type=str, default='dicom', help='dicom or raw or tif?')
parser.add_argument('--save-imgs', action="store_true", help="save denoised images?")
parser.add_argument('--in-dtype', type=str, default="uint16", help="data type of input images. out-dtype is eq to in-dtype.")

parser.add_argument('--sigma', type=float, default=5, help="std for noisy image. Range: [0, 1]\
                                                            Increasing std increases smoothing.")
parser.add_argument('--rNx', required=False, type=int, default=None, help="image size for raw image as input.")
parser.add_argument('--crop-xcat', action='store_true', help="crop the xcat data when evaluating global metrics?")
parser.add_argument('--crop-acr', action='store_true', help="crop the acr data when evaluating global metrics?")
parser.add_argument('--by-patient-name', action='store_true', help="is CNN applied to images from different patients? \
                                                                    If yes then images will be saved with patient tag.")
args = parser.parse_args()

from bm3d_denoiser import bm3d_solver; bm3d_solver(args)
