### Highlight



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
  --scale SCALE         up-scaling factor. It is hard-coded to be 1 for denoising (default: 1)
  --num-channels        3 for rgb images and 1 for gray scale images (default: 1)
  --val-batch-size      input batch size for validation data. (default: 16)
  --validating-fname    Path to H5 file(s) containing validation set (default: None)
  --descriptor-type     string to provide addition info to designate path to save logs and checkpoints (default: None)
  --shuffle_patches     shuffles the train/validation patch pairs(input-target) at utils.data.DataLoader & not at the
                        HDF5dataloader (default: False)
  --save_log_ckpts      saves log writer and checkpoints (default: False)


``` 



### License and Copyright
This software and documentation (the "Software") were developed at the Food and Drug Administration (FDA) by employees of the Federal Government in the course of their official duties. Pursuant to Title 17, Section 105 of the United States Code, this work is not subject to copyright protection and is in the public domain. Permission is hereby granted, free of charge, to any person obtaining a copy of the Software, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software or derivatives, and to permit persons to whom the Software is furnished to do so. FDA assumes no responsibility whatsoever for use by other parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. Further, use of this code in no way implies endorsement by the FDA or confers any advantage in regulatory decisions. Although this software can be redistributed and/or modified freely, we ask that any derivative works bear some notice that they are derived from it, and any modified versions bear some notice that they have been modified.

### Contact
Prabhat KC
prabhat.kc077@gmail.com
