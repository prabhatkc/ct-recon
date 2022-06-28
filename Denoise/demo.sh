#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# for patient images
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# increasing reg lambda term increases somethering
# 0.02 smothers more than 0.01
# 0.06 yeilds blocky results
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=0
INPUT_FOLDER="./data/L096"
python main.py --input-folder $INPUT_FOLDER --input-gen-folder "quarter_3mm_sharp_sorted" \
--cuda --input-img-type 'dicom' \
--lr 0.001 --nite 100 --reg-lambda 0.01 --save-imgs  --target-gen-folder "full_3mm_sharp_sorted" \
--out-dtype 'uint16' --print-opt-errs

<<COMMENT
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