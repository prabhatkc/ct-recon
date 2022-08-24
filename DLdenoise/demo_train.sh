#------------------------------------------------------------------------------------------------------------------#
#  									OPTIONS
#------------------------------------------------------------------------------------------------------------------#
<< COMMENT
usage: main_hvd.py [-h] [--model-name MODEL_NAME] [--nepochs NEPOCHS] [--cuda] [--batches-per-allreduce BATCHES_PER_ALLREDUCE]
                   [--log-dir LOG_DIR] [--fp16-allreduce] [--batch-size BATCH_SIZE] [--start-epoch START_EPOCH]
                   [--pretrained PRETRAINED] [--checkpoint-format CHECKPOINT_FORMAT] [--base-lr BASE_LR] [--wd WD] [--seed SEED]
                   [--opt-type OPT_TYPE] [--momentum MOMENTUM] [--loss-func LOSS_FUNC] [--prior-type PRIOR_TYPE]
                   [--training-fname TRAINING_FNAME] [--val-chk-prsc VAL_CHK_PRSC] [--scale SCALE] [--num-channels NUM_CHANNELS]
                   [--val-batch-size VAL_BATCH_SIZE] [--validating-fname VALIDATING_FNAME] --descriptor-type DESCRIPTOR_TYPE
                   [--shuffle_patches] [--save_log_ckpts] [--reg-lambda REG_LAMBDA]


usage: main_gan_hvd.py [-h] [--gan-name GAN_NAME] [--nepochs NEPOCHS] [--cuda] [--batches-per-allreduce BATCHES_PER_ALLREDUCE] 
                       [--log-dir LOG_DIR] [--fp16-allreduce] [--batch-size BATCH_SIZE] [--resume RESUME] [--start-epoch START_EPOCH] 
                       [--pretrained PRETRAINED] [--checkpoint-format CHECKPOINT_FORMAT] [--base-lr BASE_LR] [--seed SEED]
                       [--warmup-epochs WARMUP_EPOCHS] [--training-fname TRAINING_FNAME] [--val-chk-prsc VAL_CHK_PRSC]
                       [--scale SCALE] [--num-channels NUM_CHANNELS] [--val-batch-size VAL_BATCH_SIZE]
                       [--validating-fname VALIDATING_FNAME] --descriptor-type DESCRIPTOR_TYPE [--shuffle_patches]
                       [--save_log_ckpts]
PyTorch DLCT
COMMENT
# assign slots from your GPU allocations
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

model_option=$1
model_opt1="CNN3"
model_opt2="REDCNN"
model_opt3="DnCNN"
model_opt4="UNet"
model_opt5="GAN" 

if [ "$model_option" == "$model_opt2" ]; then
    # ----------------------------------------------------# 
    #     REDCNN training for unnormalized data
    # ----------------------------------------------------# 
    NEPOCH=5
    TRAIN_FNAME='./train_data/p96_no_norm/train'
    VAL_FNAME='./train_data/p96_no_norm/tune'
    DES_TYPE='p55_no_norm/augTrTaTdT'
    time horovodrun -np 2 -H localhost:2 python main_hvd.py --batch-size 64 --batches-per-allreduce 1 --cuda \
    --nepochs $NEPOCH --base-lr 1e-5 --training-fname $TRAIN_FNAME --validating-fname $VAL_FNAME \
    --descriptor-type $DES_TYPE --val-chk-prsc 'positive-float' --val-batch-size 64 --loss-func 'mse' \
    --model-name 'redcnn' --prior-type 'tv-fbd' --reg-lambda 1e-4 --shuffle_patches --save_log_ckpts

elif [[ "$model_option" == "$model_opt3"  ]]; then
    # ----------------------------------------------------# 
    #     DnCNN training for unnormalized data
    # ----------------------------------------------------# 
    NEPOCH=5
    TRAIN_FNAME='./train_data/p55_uni_norm/train'
    VAL_FNAME='./train_data/p55_uni_norm/tune'
    DES_TYPE='p55_uni_norm/augTrTaTdT'
    time horovodrun -np 2 -H localhost:2 python main_hvd.py --batch-size 32 --batches-per-allreduce 1 --cuda \
    --nepochs $NEPOCH --base-lr 1e-4 --training-fname $TRAIN_FNAME --validating-fname $VAL_FNAME \
    --descriptor-type $DES_TYPE --val-chk-prsc 'positive-float' --val-batch-size 32 --loss-func 'mse' \
    --model-name 'dncnn' --prior-type 'l1' --reg-lambda 1e-8 --wd 1e-5 --shuffle_patches --save_log_ckpts

elif [[ "$model_option" == "$model_opt4"  ]]; then
    # ----------------------------------------------------# 
    #     UNet training for normalized data
    # ----------------------------------------------------# 
    NEPOCH=10
    TRAIN_FNAME='./train_data/p55_uni_norm/train'
    VAL_FNAME='./train_data/p55_uni_norm/tune'
    DES_TYPE='p55_uni_norm/augTrTaTdT'
    time horovodrun -np 2 -H localhost:2 python main_hvd.py --batch-size 64 --batches-per-allreduce 1 --cuda \
    --nepochs $NEPOCH --base-lr 1e-3 --training-fname $TRAIN_FNAME --validating-fname $VAL_FNAME \
    --descriptor-type $DES_TYPE --val-chk-prsc 'positive-float' --val-batch-size 64 --loss-func 'mse' \
    --model-name 'unet' --prior-type 'l1' --reg-lambda 1e-7 --save_log_ckpts --shuffle_patches

elif [[ "$model_option" == "$model_opt5" ]]; then
    # ----------------------------------------------------# 
    #     GAN training for unnormalized data
    # ----------------------------------------------------# 
    NEPOCH=5
    TRAIN_FNAME='./train_data/p55_uni_norm/train'
    VAL_FNAME='./train_data/p55_uni_norm/tune'
    DES_TYPE='p55_uni_norm/augTrTaTdT'
    time horovodrun -np 2 -H localhost:2 python main_gan_hvd.py --batch-size 32 --batches-per-allreduce 1 --cuda \
    --nepochs $NEPOCH --base-lr 1e-4 --training-fname $TRAIN_FNAME --validating-fname $VAL_FNAME \
    --descriptor-type $DES_TYPE --val-chk-prsc 'positive-float' --val-batch-size 32 --shuffle_patches --save_log_ckpts

else 
    # ----------------------------------------------------# 
    #     CNN3 training for unnormalized data
    # ----------------------------------------------------# 
    NEPOCH=10
    TRAIN_FNAME='./train_data/p55_no_norm/train'
    VAL_FNAME='./train_data/p55_no_norm/tune'
    DES_TYPE='p55_no_norm/augTrTaTdT'
    time horovodrun -np 2 -H localhost:2 python main_hvd.py --batch-size 64 --batches-per-allreduce 1 --cuda \
    --nepochs $NEPOCH --base-lr 1e-5 --training-fname $TRAIN_FNAME --validating-fname $VAL_FNAME \
    --descriptor-type $DES_TYPE --val-chk-prsc 'positive-float' --val-batch-size 64 \
    --loss-func 'mse' --model-name 'cnn3' --shuffle_patches --save_log_ckpts
fi 

    
