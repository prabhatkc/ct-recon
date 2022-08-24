# Horovod with PyTorch based Deep Learning (DL) for CT denoising applications #

## Highlights ##
    
1. Multi-GPU implementation for generative (GAN), as well as single loss function based DL, models.
2. Options to integrate different prior terms (like the NLM, TV) to the loss function.
3. Options to read training pairs in different formats like npz, h5, images.
4. Validation loss is accompanied with SSIM, RMSE from the tuning set at the time of training.
5. Use any network architecture in a plug-n-play manner.

```
usage: main_hvd.py [-h] [--model-name MODEL_NAME] [--nepochs NEPOCHS] [--cuda] [--batches-per-allreduce BATCHES_PER_ALLREDUCE]
                   [--log-dir LOG_DIR] [--fp16-allreduce] [--batch-size BATCH_SIZE] [--start-epoch START_EPOCH]
                   [--pretrained PRETRAINED] [--checkpoint-format CHECKPOINT_FORMAT] [--base-lr BASE_LR] [--wd WD] [--seed SEED]
                   [--opt-type OPT_TYPE] [--momentum MOMENTUM] [--loss-func LOSS_FUNC] [--prior-type PRIOR_TYPE]
                   [--training-fname TRAINING_FNAME] [--val-chk-prsc VAL_CHK_PRSC] [--scale SCALE] [--num-channels NUM_CHANNELS]
                   [--val-batch-size VAL_BATCH_SIZE] [--validating-fname VALIDATING_FNAME] --descriptor-type DESCRIPTOR_TYPE
                   [--shuffle_patches] [--save_log_ckpts] [--reg-lambda REG_LAMBDA]

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
  --scale               up-scaling factor. It is hard-coded to be 1 for denoising (default: 1)
  --num-channels        3 for rgb images and 1 for gray scale images (default: 1)
  --val-batch-size      input batch size for validation data. (default: 16)
  --validating-fname    Path to H5 file(s) containing validation set (default: None)
  --descriptor-type     string to provide addition info to designate path to save logs and checkpoints (default: None)
  --shuffle_patches     shuffles the train/validation patch pairs(input-target) at utils.data.DataLoader & not at the
                        HDF5dataloader (default: False)
  --save_log_ckpts      saves log writer and checkpoints (default: False)

``` 
## Example Usage ##
1. Create Deep Learning ready h5 input-target patches. See an example [here](https://github.com/prabhatkc/mpi4py_patches) or use demo h5 files in the train_data folder.
2. <b>TRAIN</b> your network via declaring paths for training/tuning/(to be saved) checkpoints & numerical parameters as:
``` 
    $ NEPOCH=5
    $ TRAIN_FNAME='./train_data/p55_no_norm/train'
    $ VAL_FNAME='./train_data/p55_no_norm/tune'
    $ DES_TYPE='p55_no_norm/augTrTaTdT'
    $ time horovodrun -np 2 -H localhost:2 python main_hvd.py --batch-size 64 --batches-per-allreduce 1 --cuda \
    --nepochs $NEPOCH --base-lr 1e-5 --training-fname $TRAIN_FNAME --validating-fname $VAL_FNAME \
    --descriptor-type $DES_TYPE --val-chk-prsc 'positive-float' --val-batch-size 64 \
    --loss-func 'mse' --model-name 'cnn3' --shuffle_patches --save_log_ckpts
```
> > Instead you may choose to (modify &) execute demo_train.sh file as
```
    $ chmod +x demo_train.sh
    $ ./demo_train.sh CNN3
```
3. Apply trained weights to <b>TEST</b> set as: 
```
    $ set -f echo *
    $ INPUT_FOLDER="./test_data/patient_data/*/quarter_3mm_sharp_sorted"
    $ GT_FOLDER="./test_data/patient_data/*/full_3mm_sharp_sorted"
    $ OUTPUT_FOLDER='./results/patient_test/cnn3'
    $ python resolve.py --m 'cnn3' --input-folder $INPUT_FOLDER --model-folder $MODEL_FOLDER --gt-folder $GT_FOLDER \
    --output-folder $OUTPUT_FOLDER --cuda --normalization-type $NORM_TYPE --input-img-type 'dicom' --specific-epoch --se-plot
    $ set +f echo *
```
> > Instead you may choose to (modify &) execute demo_test.sh file as
```
    $ chmod +x demo_test.sh
    $ ./demo_test.sh 
```

## Package Requirements ##
- Follow the [Hovorod Installation Guide](https://horovod.readthedocs.io/en/stable/install_include.html) to configure Horovod with PyTorch in your environment.
- Use [pip]( https://pypi.org/project/pip/) or [conda]( https://docs.anaconda.com/anaconda/user-guide/tasks/install-packages/) to install additional packages needed:
sklearn, matplotlib, natsort, jupyterlab, opencv-python, tqdm, pydicom, pytorch-mssim, tensorboardX, glob, h5py, numpy, imageio, scipy.
- This implementation was developed using horovod v.0.19.5.

## References ##
- This implementation is built upon [horovod’s](https://github.com/horovod/horovod#install) publicly available source code.
  - Sergeev, A. and Del Balso, M., 2018. Horovod: fast and easy distributed deep learning in TensorFlow. arXiv preprint arXiv:1802.05799.
- Some of the patient image used to formulate training/tuning h5 files is imported from [the Low Dose CT Grand Challenge](https://www.aapm.org/grandchallenge/lowdosect/)
  -  McCollough, C., Chen, B., Holmes III, D. R., Duan, X., Yu, Z., Yu, L., Leng, S., & Fletcher, J. (2020). Low Dose CT Image and Projection Data (LDCT-and-Projection-data) (Version 4) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/9NPB-2637

## License and Copyright ##
This software and documentation (the "Software") were developed at the Food and Drug Administration (FDA) by employees of the Federal Government in the course of their official duties. Pursuant to Title 17, Section 105 of the United States Code, this work is not subject to copyright protection and is in the public domain. Permission is hereby granted, free of charge, to any person obtaining a copy of the Software, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software or derivatives, and to permit persons to whom the Software is furnished to do so. FDA assumes no responsibility whatsoever for use by other parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. Further, use of this code in no way implies endorsement by the FDA or confers any advantage in regulatory decisions. Although this software can be redistributed and/or modified freely, we ask that any derivative works bear some notice that they are derived from it, and any modified versions bear some notice that they have been modified.

## Contact ##
Prabhat KC
prabhat.kc077@gmail.com
