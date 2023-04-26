INPUT_FOLDER="../data/phandata/poisson"
OUTPUT_FOLDER="./results/bm3d/para_est"
python main.py --input-folder $INPUT_FOLDER --input-gen-folder "noisy" --input-img-type 'raw' \
--save-imgs  --target-gen-folder "gt" --sigma 0.034 --rNx 512 --output-folder $OUTPUT_FOLDER