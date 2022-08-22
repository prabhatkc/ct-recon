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

optional arguments:
  -h, --help            show this help message and exit
  --model-name, --m     choose the architecture that you are going to use to train your data. (default: cnn3)
  --nepochs             number of epochs to train (default: 5)
  --cuda                use cuda? (default: False)
  --batches-per-allreduce 
                        number of batches processed locally before executing allreduce across workers. It multiplies the total
                        batch size. (RHR: 1 loss function eqs 1 batches-per-allreduce (default: 1)
  --log-dir             tensorboard parent log directory (default: ./logs)
  --fp16-allreduce      use fp16 compression during allreduce (default: False)
  --batch-size          training batch size (default: 16)
  --start-epoch         manual epoch number (useful on restarts) (default: 1)
  --pretrained          path to pretrained model (default: none). Not used so far. (default: )
  --checkpoint-format   checkpoint file format (default: checkpoint-{epoch}.pth.tar)
  --base-lr             learning rate for a single GPU (default: 0.0125)
  --wd                  weight decay a.k.a regularization on weights (default: 0.0)
  --seed                random seed (default: 42)
  --opt-type            adam or sgd to perform loss minimization (default: adam)
  --momentum            SGD momentum (default: 0.9)
  --loss-func           loss function to be used such as mse, l1, ce (default: mse)
  --prior-type          prior terms to be combined with the data fedility term. 
                        Options include l1, nl, sobel, tv-fd, tv-fbd (default: None)
  --training-fname      path to training LR-HR training patches in h5 format (default: None)
  --val-chk-prsc 
                        precision type while calculating the SSIM/PSNR for validation test. (default: natural-float)
  --scale SCALE         up-scaling factor. It is hard-coded to be 1 for denoising (default: 1)
  --num-channels        3 for rgb images and 1 for gray scale images (default: 1)
  --val-batch-size      input batch size for validation data. (default: 16)
  --validating-fname    Path to H5 file(s) containing validation set (default: None)
  --descriptor-type     string to provide addition info to designate path to save logs and checkpoints (default: None)
  --shuffle_patches     shuffles the train/validation patch pairs(input-target) at utils.data.DataLoader & not at the
                        HDF5dataloader (default: False)
  --save_log_ckpts      saves log writer and checkpoints (default: False)
  --reg-lambda          pre-factor for the prior term (default: 0.0)

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
    NEPOCH=5
    TRAIN_FNAME='./train_data/p55_no_norm/train'
    VAL_FNAME='./train_data/p55_no_norm/tune'
    DES_TYPE='p55_no_norm/augTrTaTdT'
    time horovodrun -np 2 -H localhost:2 python main_hvd.py --batch-size 64 --batches-per-allreduce 1 --cuda \
    --nepochs $NEPOCH --base-lr 1e-5 --training-fname $TRAIN_FNAME --validating-fname $VAL_FNAME \
    --descriptor-type $DES_TYPE --val-chk-prsc 'positive-float' --val-batch-size 64 \
    --loss-func 'mse' --model-name 'cnn3' --shuffle_patches --save_log_ckpts
fi 

    
