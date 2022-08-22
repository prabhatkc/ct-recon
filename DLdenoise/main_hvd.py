
from __future__ import print_function

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn 
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms
import horovod.torch as hvd 
import torch.nn.functional as F

import tensorboardX
import os
import math
from tqdm import tqdm
import sys

from loss import combinedLoss
from dataset import DatasetFromHdf5, DatasetfromFolder
from torchsummary import summary 
import util
import time 

start_time=time.time()
# ====================================================================
# Training settings
# ====================================================================
parser = argparse.ArgumentParser(description="PyTorch DLCT",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model-name','--m', type=str, default='cnn3', 
                    help='Choose the architecture that you are going to use to train your data.')
parser.add_argument('--nepochs', type=int, default=5, 
                    help='number of epochs to train')
parser.add_argument("--cuda", action="store_true", 
                    help="Use cuda?")
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before executing allreduce across workers;'
                    'It multiplies the total batch size. (RHR: 1 loss function eqs 1 batches-per-allreduce')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard parent log directory')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument("--batch-size", type=int, default=16, help="training batch size")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none). Not used so far.")
parser.add_argument('--checkpoint-format', default='checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--wd', type=float, default=0.0,
                    help='weight decay a.k.a regularization on weights')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--opt-type', type=str, default='adam',
                    help='adam or sgd to perform loss minimization')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--loss-func', type=str, default='mse', help='loss function to be used such as mse, l1, ce')
parser.add_argument('--prior-type', type=str, default=None, help="prior terms to be combined with the data fedility term.\
                                                                  Options include l1, nl, sobel, tv-fd, tv-fbd")
parser.add_argument('--training-fname', type=str, help='Path to training LR-HR training patches in h5 format')
parser.add_argument('--val-chk-prsc', type=str, default='natural-float', help='precision type while calculating the SSIM/PSNR for validation test.')
parser.add_argument('--scale', type=int, default=1, help='up-scaling factor. It is hard-coded to be 1 for denoising')
parser.add_argument('--num-channels', type=int, default=1, help='3 for rgb images and 1 for gray scale images')
parser.add_argument('--val-batch-size', type=int, default=16, help='input batch size for validation data.')
parser.add_argument('--validating-fname', type=str, help=' Path to H5 file(s) containing validation set (default: None)')
parser.add_argument('--descriptor-type', type=str, required=True, 
                    help='string to provide addition info to designate path to save logs and checkpoints (default: None)')
parser.add_argument('--shuffle_patches', action="store_true", help="shuffles the train/validation patch pairs(input-target) at \
                                                                    utils.data.DataLoader & not at the HDF5dataloader")
parser.add_argument('--save_log_ckpts', action="store_true", help="saves log writer and checkpoints")
parser.add_argument('--reg-lambda', type=float, default=0.0, help="pre-factor for the prior term")

#parser.add_argument('--val-hr', type=str, help='Path to HR/target image directory in validation set')
#parser.add_argument('--val-lr', type=str, help='Path to LR/input image directory in validation set')


# ====================================================================
# Initilize program with command line arguments and CUDA dependencies
# ====================================================================
args = parser.parse_args()
cuda = args.cuda and torch.cuda.is_available()
allreduce_batch_size = args.batch_size * args.batches_per_allreduce

# -----------------------------------------------------------
# importing architectures:
# -----------------------------------------------------------
if args.model_name =='cnn3':
  from models.cnn3 import CNN3
  model = CNN3(num_channels=args.num_channels)
if args.model_name =='redcnn':
  from models.redcnn import REDcnn10
  model = REDcnn10(idmaps=3) # idmaps: no of skip connections (only 1 or 3 available)
if args.model_name == 'unet':
  from models.unet import UDnCNN
  model = UDnCNN(D=10) # D is no of layers
if args.model_name == 'dncnn':
  from models.dncnn import DnCNN
  model = DnCNN(channels=args.num_channels) # default: layers used is 17, bn=True, dropout=F

pt_str = args.prior_type if args.prior_type is not None else ''
if args.save_log_ckpts:
  # declaring the checkpoint fname
  checkpoint_folder = os.path.join('checkpoints', args.descriptor_type + '/' + args.model_name + '/hvd_cpt_for_' + args.loss_func + '_' + pt_str + '_wd_' + str(args.wd)+'_lr_'+str(args.base_lr)+'_bs_'+str(args.batch_size))
  if (args.reg_lambda != 0.0):
    checkpoint_folder = os.path.join('checkpoints', args.descriptor_type + '/' + args.model_name + '/hvd_cpt_for_' + args.loss_func + '_' + pt_str + '_reg_' + str(args.reg_lambda) +'_wd_' + str(args.wd)+'_lr_'+str(args.base_lr)+'_bs_'+str(args.batch_size))
  if not os.path.isdir(checkpoint_folder): os.makedirs(checkpoint_folder, exist_ok=True)
  args.checkpoint_format = os.path.join(checkpoint_folder, args.checkpoint_format)
  # declaring the log fname
  current_log_folder = os.path.join(args.log_dir, args.descriptor_type + '/' + args.model_name + '/hvd_log_for_' + args.loss_func + '_' + pt_str + '_wd_' + str(args.wd)+'_lr_'+str(args.base_lr)+'_bs_'+str(args.batch_size))
  if (args.reg_lambda !=0.0):
    current_log_folder = os.path.join(args.log_dir, args.descriptor_type + '/' + args.model_name + '/hvd_log_for_' + args.loss_func + '_' + pt_str + '_reg_' + str(args.reg_lambda) + '_wd_' + str(args.wd)+'_lr_'+str(args.base_lr)+'_bs_'+str(args.batch_size))
  args.log_dir = current_log_folder.format(int(time.time()))

hvd.init()
torch.manual_seed(args.seed)

if cuda:
	  # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)
cudnn.benchmark = True

# If checkpoints upto ith iterations have been saved from previous computations
# then iteration and terminal broadcast start from (i+1)th iteration
# old checkpoints are only read if save_log_ckpts is true
resume_from_epoch = 0
for try_epoch in range(args.nepochs, 0, -1):
    if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break

# Horovod: To ensure that all GPUs are initialized with same weights as that 
# of the root_rank whether random or that loaded from already trained checkpoint
resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                  name='resume_from_epoch').item()
# Display command line arguments
if hvd.rank() == 0:
  print('\n----------------------------------------')
  print('Command line arguements')
  print('----------------------------------------')
  print("\nNo. of gpus used:", hvd.size())
  for i in args.__dict__: print((i),':',args.__dict__[i])
  print('\n----------------------------------------\n')

# Horovod: print logs on the first worker
verbose = 1 if hvd.rank() == 0 else 0

# Horovod: write TensorBoard logs on first worker.
comment='loss_func={args.loss_func}_prior_type={args.prior_type}_wd={args.wd}'
log_writer = tensorboardX.SummaryWriter(args.log_dir) if (hvd.rank() == 0 and args.save_log_ckpts) else None
# ==================================================================
# load training data
# ==================================================================
# (1) DatasetFromHdf5 returns input, target. 
# Also if mod(len(input), no_of_GPUs*batchsize) = k, the last k
# patches are chucked off. 
#
# (2) likewise distributed sampler and dataloader will make batches
# of the input and target and distribute over the given no of gpus
#
# (3) shuffle=False, drop_last=False as the default value in the torch's dataloader 
#
kwargs = {'num_workers': 8, 'pin_memory': True} if cuda else {}
train_dataset = DatasetFromHdf5(hvd, args.training_fname, hvd.size()*args.batch_size) 

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, \
	num_replicas=hvd.size(), rank=hvd.rank(), shuffle=args.shuffle_patches)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=allreduce_batch_size, \
	sampler=train_sampler, **kwargs)
if hvd.rank()==0: print(" Dimension of (input <-> target) batches is: {} <-> {}".format(train_dataset.data.shape, \
  train_dataset.target.shape))
_, _, in_h, in_w = train_dataset.data.shape
_, _, tg_h, tg_w = train_dataset.target.shape 
#sys.exit()
# ==================================================================
# load validation data
# ==================================================================
# (1) validation input and target can come from
#     (a) DatasetfromFolder: with full input & target images
#     (b) DatasetFromHdf5: with input, target patches 
# Also if mod(len(input), no_of_GPUs*Val_batchsize) = k, the last k
# patches are chucked off. 
#
# (3) likewise distributed sampler and dataloader will make batches
# of the input and target and distributed over the given no of gpus
#
# val_dataset = DatasetfromFolder(image_dir_hr=args.val_hr, image_dir_lr=args.val_lr)
val_dataset = DatasetFromHdf5(hvd, args.validating_fname, hvd.size()*args.val_batch_size)
val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=args.shuffle_patches)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                         sampler=val_sampler, **kwargs)

# ==================================================================
# Initializing Model and objective loss function
# ==================================================================
reg_lambda = args.reg_lambda
objloss = combinedLoss(args, reg_lambda)
if (hvd.rank()==0): print(objloss)

# transfer models to cuda
if cuda:
  model = model.cuda()

# MODEL SUMMARY
if hvd.rank()==0: 
  summary(model, (args.num_channels, in_h, in_w))
  print(" Data Loading Time is: %.4f min" % ((time.time()-start_time)/60.0))
  print('----------------------------------------------------------------')
#sys.exit()

# ==================================================================
# Initializing the optimizer type
# ==================================================================
if (args.opt_type == 'adam'):
  # adam uses its inbuilt subroutines to determine momemtum parameter
  optimizer = optim.Adam(model.parameters(),
                  lr=(args.base_lr *
                      args.batches_per_allreduce * hvd.size()), weight_decay=args.wd)
else:
  optimizer = optim.SGD(model.parameters(),
                  lr=(args.base_lr *
                      args.batches_per_allreduce * hvd.size()),
                  momentum=args.momentum, weight_decay=args.wd)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), \
	compression=compression, \
	backward_passes_per_step=args.batches_per_allreduce)

# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast weights to other workers.
if resume_from_epoch > 0 and hvd.rank() == 0:
    filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# ==================================================================
# train the model 
# optimer*.zero_grad(): sets the gradient of the optimizer to zero
# _.backward() : computes gradient w.r.t. current leaves (tensor)
# optimizer*.step() : updates the hyper parameters
# ==================================================================
def train(epoch):
  model.train()
  train_loss = util.Metric('train_loss')
  with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
      for batch_idx, (data, target) in enumerate(train_loader):

          #adjust_learning_rate(epoch, batch_idx)
          #if (args.wd) != 0.0:
          util.adjust_learning_rate_3_zones(epoch, 5, 15, args, optimizer, hvd.size())
          #else:
          #util.adjust_learning_rate_3_zones(epoch, 50, 80, args, optimizer, hvd.size())
          
          if cuda: data, target = data.cuda(), target.cuda()
          optimizer.zero_grad()
          # ------------------------------------------------------------
          # this loop, further, sub-divides your batches if your cmd 
          # line input for batches_per_allreduce is more than one
          # say your batch_size is 128 and batches_per_allreduce is 2
          # then dataloader will process at [256, 1, h, w]
          # subsequently, len(i) =2, and i=0 will process 0:128 batches
          # and i=1 will process 128:256 batches
          # ------------------------------------------------------------
          for i in range(0, len(data), args.batch_size):
            data_batch   = data[i:i + args.batch_size]
            target_batch = target[i:i + args.batch_size]
            output       = model(data_batch)
            loss         = objloss(output, target_batch)
            train_loss.update(loss, hvd)

            # Average gradients among sub-batches
            loss.div_(math.ceil(float(len(data)) / args.batch_size))
            loss.backward()
            
          optimizer.step()
          # below shows average loss per batch per GPU for each given iteration.
          # eg. if each of 2 GPUS are processing [4*128, 1, 5, 5] 
          # .s.t train dataset is [1024, 1, 5, 5] and the total loss for 
          # for epoch 'e' is lss, then the stored loss value is lss/(4*2)
          t.set_postfix({'loss': train_loss.avg.item()})
          t.update(1)

  if log_writer:
    log_writer.add_scalar('train/loss', train_loss.avg, epoch)


# ==================================================================
# validate the network using metrics such as PSRN and SSIM
# ==================================================================
def validate(epoch):
  model.eval()
  val_loss = util.Metric('val_loss')
  val_psnr = util.Metric('val_psnr')
  val_ssim = util.Metric('val_ssim')

  with tqdm(total=len(val_loader),
            desc='Validate Epoch  #{}'.format(epoch + 1),
            disable=not verbose) as t:
      with torch.no_grad():
          for data, target in val_loader:
              if args.cuda:
                  data, target = data.cuda(), target.cuda()
              output = model(data)
              #print('data dev type:', data.device, 'target dev type:', target.device, 'output dev type', output.device)
              #print('data, target, output shape', data.size(), target.size(), output.size())
              vloss  = objloss(output, target)
              _psnr, _ssim = util.quant_ana(output, target, args.val_chk_prsc) #quant ana converts gpu types to cpu inside it
              val_loss.update(vloss, hvd)
              val_psnr.update(_psnr, hvd)
              val_ssim.update(_ssim, hvd)
              t.set_postfix({'PSNR ': val_psnr.avg.item(),
                             'SSIM': val_ssim.avg.item(), 
                             'vloss': val_loss.avg.item()})
              t.update(1)
            
  if log_writer:
        # below saves average loss per batch per GPU for each given 
        # iteration in the log files.
        # eg. if each of 2 GPUS are processing [4*128, 1, 5, 5] 
        # s.t train dataset is [1024, 1, 5, 5] and the total loss for 
        # for epoch 'e' is lss, then the stored loss value is lss/(4*2)
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/PSNR', val_psnr.avg, epoch)
        log_writer.add_scalar('val/SSIM', val_ssim.avg, epoch)


# ====================================================================
# Main function with train, validate and save weights
# ======================================================================
for epoch in range(resume_from_epoch, args.nepochs):
    train(epoch)
    validate(epoch)
    if (args.save_log_ckpts): util.save_checkpoint(epoch, args, hvd, model, optimizer)
