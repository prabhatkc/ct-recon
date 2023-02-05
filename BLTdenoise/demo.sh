# increasing reg lambda term increases somethering
# tuned parameters: sc 0.02 ws 7 ss 5

INPUT_FOLDER="../data/phandata/poisson"
OUTPUT_FOLDER="./results/phandata"
python main.py --input-folder $INPUT_FOLDER --input-gen-folder "noisy" --input-img-type 'raw' \
--save-imgs  --target-gen-folder "gt" --in-dtype 'uint16' --sigma-color 0.02 \
--sigma-spatial 5 --win-size 7 --rNx 512 --output-folder $OUTPUT_FOLDER