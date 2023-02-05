#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# for patient images
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# increasing reg lambda term increases somethering
# tuned parameters: sc 0.02 ws 7 ss 5

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# LCD test (sc 0.02 ws 7 ss 5)
# signal types are disk & bkg for (nd qd tfd hfd)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
SIG_TYPE='/bkg'
INPUT_FOLDER='/home/prabhat.kc/Implementations/matlab/irt/digiNoise/results/mita'
OUTPUT_FOLDER='./results/bilateral/mita_lcd'
python main.py --input-folder $INPUT_FOLDER$SIG_TYPE --input-gen-folder "qd" --input-img-type 'raw' \
--save-imgs  --out-dtype 'uint16' --sigma-color 0.02 \
--sigma-spatial 5 --win-size 7 --rNx 512 --output-folder $OUTPUT_FOLDER

<<COMMENT
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Parameter estimation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

for ws in 3 5 #7 14 21 28 35 
do 
INPUT_FOLDER="../data/phandata/poisson"
OUTPUT_FOLDER="./results/bilateral/para_est"
python main.py --input-folder $INPUT_FOLDER --input-gen-folder "noisy" --input-img-type 'raw' \
--save-imgs  --target-gen-folder "gt" --out-dtype 'uint16' --sigma-color 0.02 \
--sigma-spatial 5 --win-size $ws --rNx 512 --output-folder $OUTPUT_FOLDER
done 

# sigma color test
# for sc in 0.01 0.05 0.07 0.1 0.15
for sc in 0.02 0.03 0.04 
do 
INPUT_FOLDER="../data/phandata/poisson"
OUTPUT_FOLDER="./results/bilateral/para_est"
python main.py --input-folder $INPUT_FOLDER --input-gen-folder "noisy" --input-img-type 'raw' \
--save-imgs  --target-gen-folder "gt" --out-dtype 'uint16' --sigma-color $sc \
--sigma-spatial 5 --win-size 7 --rNx 512 --output-folder $OUTPUT_FOLDER
done 
# 
# increasing sc along (0.01 0.05 0.07 0.1 0.15)
#	smoothers  output. It increases ssim (good) and RMSE (bad) but decreases psnr (bad)
# 	also 0.01 does yield denoised solution as compared to LD even for 10% acquisition
# 	best value obtained at 0.02

for ss in 5 10 15 20 25 
do 
INPUT_FOLDER="../data/phandata/poisson"
#OUTPUT_FOLDER="./results/bilateral/para_est/ss_"$ss"_ws_7_sc_0.01"
OUTPUT_FOLDER="./results/bilateral/para_est"
python main.py --input-folder $INPUT_FOLDER --input-gen-folder "noisy" --input-img-type 'raw' \
--save-imgs  --target-gen-folder "gt" --out-dtype 'uint16' --sigma-color 0.01 \
--sigma-spatial $ss --win-size 7 --rNx 512 --output-folder $OUTPUT_FOLDER
done
#no difference in output for changes in ss for sc of 0.01 and ws 7
# so ss is set to 5 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# denoising uniform phantom
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
INPUT_FOLDER="../../data/uniform"
python main.py --input-folder $INPUT_FOLDER --input-gen-folder "fbp_sharp" --cuda \
--input-img-type 'raw' --lr 0.001 --nite 100 --reg-lambda 0.01 --save-imgs  --out-dtype 'int16' \
--print-opt-errs

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#  ctp404 phantom test
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
INPUT_FOLDER='../../data/ctp404'
python main.py --input-folder $INPUT_FOLDER --cuda --input-img-type 'raw' \
--lr 0.001 --nite 100 --reg-lambda 0.02 --save-imgs --out-dtype 'int16' --input-gen-folder "fbp_sharp" --print-opt-errs
#python resolve_mtf.py --input-folder $INPUT_FOLDER --gt-folder $GT_FOLDER \
#--output-folder $OUTPUT_FOLDER --cuda --se-plot --input-img-type 'raw'
COMMENT