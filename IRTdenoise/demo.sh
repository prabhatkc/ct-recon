
# best value: 0.01 reg_lambda and lr: 1e-3
# increasing reg lambda term increases somethering
# 0.02 smothers more than 0.01
# 0.06 yeilds blocky results

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# for patient images
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
INPUT_FOLDER="./data/L096"
python main.py --input-folder $INPUT_FOLDER --input-gen-folder "quarter_3mm_sharp_sorted" \
--cuda --input-img-type 'dicom' \
--lr 0.001 --nite 100 --reg-lambda 0.01 --save-imgs  --target-gen-folder "full_3mm_sharp_sorted" \
--in-dtype 'uint16' --print-opt-errs

<< COMMENT
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# denoising uniform phantom
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
INPUT_FOLDER="./data/uniform_water_phan"
python main.py --input-folder $INPUT_FOLDER --input-gen-folder "fbp_sharp" --cuda \
--input-img-type 'raw' --lr 0.001 --nite 100 --reg-lambda 0.01 --save-imgs  --in-dtype 'int16' \
--print-opt-errs --rNx 256

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#  ctp404 phantom test
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
INPUT_FOLDER='./data/ctp404'
python main.py --input-folder $INPUT_FOLDER --cuda --input-img-type 'raw' \
--lr 0.001 --nite 100 --reg-lambda 0.01 --save-imgs --in-dtype 'int16' --input-gen-folder "fbp_sharp" \
--print-opt-errs --rNx 256
#python resolve_mtf.py --input-folder $INPUT_FOLDER --gt-folder $GT_FOLDER \
#--output-folder $OUTPUT_FOLDER --cuda --se-plot --input-img-type 'raw'
COMMENT

