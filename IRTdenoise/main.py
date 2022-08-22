import argparse
import sys

# ====================================================================
# Command line arguments
# ====================================================================
parser = argparse.ArgumentParser(description='Application of non-DNN based denoisers on CT images')
parser.add_argument('--input-folder',type=str, required=True, help='directory name containing low dose images at different dose levels')
parser.add_argument('--input-gen-folder', type=str, default='quarter_3mm_sharp_sorted', 
													help="directory name containing specific dose-level images for low dose")
parser.add_argument('--target-gen-folder', type=str, default='', help="directory name containing Full dose imgs(if available).\
													 This option is inactivate for test cases where FD is unavailable.")
parser.add_argument('--output-folder',type=str, default='./results', help='main output foldername to store results')
parser.add_argument('--loss-func',type=str, default='mse', help='loss function to be used such as mse, l1, ce')
parser.add_argument('--prior-type',type=str, default='tv-fbd', help="prior terms to be combined with the data fedility term.\
                                                                  Options include l1, nl, sobel, tv-fd, tv-fbd")
parser.add_argument('--input-img-type', type=str, default='dicom', help='dicom or raw or tif?')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for a single GPU')
parser.add_argument('--nite', type=int, default=100, help='Number of iteration for each image ')
parser.add_argument('--reg-lambda', type=float, default=0.0, help="pre-factor for the prior term (if used).")
parser.add_argument('--cuda', action="store_true", help="Use cuda?")
parser.add_argument('--print-opt-errs', action="store_true", help="print losses and error updates for each iteration?")
parser.add_argument('--save-imgs', action="store_true", help="save denoised images?")
parser.add_argument('--out-dtype', type=str, default="uint16", help="data type to save/process desnoised output.")
args = parser.parse_args()

from optdenoiser import opt_solver; opt_solver(args)
