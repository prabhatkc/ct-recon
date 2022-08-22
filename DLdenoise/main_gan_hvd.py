import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn 
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms
import horovod.torch as hvd 
import torch.nn.functional as F
import torch.multiprocessing as mp
import tensorboardX
import os
import math
from tqdm import tqdm
import sys

from dataset import DatasetFromHdf5
from torchsummary import summary 
import util
import time 

# ====================================================================
# Training settings
# ====================================================================
parser = argparse.ArgumentParser(description="PyTorch GAN DLCT",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gan-name','--m', type=str, default='simpleGAN',
                    help='simpleGAN or WGAN or so on')
parser.add_argument('--nepochs', type=int, default=5,  help='number of epochs to train')
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before executing allreduce across workers;'
                    'it multiplies total batch size. (RHR: 1 loss function eqs 1 batches-per-allreduce')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard parent log directory')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument("--batch-size", type=int, default=16, help="training batch size")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument('--checkpoint-format', default='checkpoint-{}-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--base-lr', type=float, default=0.0125, help='learning rate for a single GPU')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--warmup-epochs', type=float, default=5, help='number of warmup epochs')
parser.add_argument('--training-fname', type=str, help='Path to training LR-HR patches in h5 format')
parser.add_argument('--val-chk-prsc', type=str, default='natural-float', help='precision type while calculating SSIM/PSNR during validation')
parser.add_argument('--scale', type=int, default=1, help='up-scaling factor')
parser.add_argument('--num-channels', type=int, default=1, help='3 for rgb images and 1 for gray scale images')

parser.add_argument('--val-batch-size', type=int, default=16,
                    help='input batch size for validation and unless same ')
parser.add_argument('--validating-fname', type=str, help='Path to HR/target image directory in validation set')

parser.add_argument('--descriptor-type', type=str, required=True, 
                    help='descriptor-type from train/test.h5. used only to designate log/checkpoint paths')
parser.add_argument('--shuffle_patches', action="store_true", help="shuffles patches at the DistributedSampler sub-routine")
parser.add_argument('--save_log_ckpts', action="store_true", help="saves log writer and checkpoints")
#parser.add_argument('--val-hr', type=str, help='Path to HR/target image directory in validation set')
#parser.add_argument('--val-lr', type=str, help='Path to LR/input image directory in validation set')


# ====================================================================
# Initilize program with command line arguments and CUDA dependencies
# ====================================================================
args = parser.parse_args()  
cuda = args.cuda and torch.cuda.is_available()
allreduce_batch_size = args.batch_size * args.batches_per_allreduce
args.num_residual    = 16 
# -----------------------------------------------------------
# importing architectures:
# -----------------------------------------------------------
if args.gan_name=="simpleGAN":
  from models.gan import Generator, Discriminator
  modelG = Generator(n_residual_blocks=args.num_residual, upsample_factor=args.scale, base_filter=64, num_channel=args.num_channels)
  modelD = Discriminator(base_filter=64, num_channel=args.num_channels)

if args.save_log_ckpts:
  # declaring the checkpoint fname
  checkpoint_folder = os.path.join('checkpoints', args.descriptor_type + '/' + args.gan_name + '/hvd_cpt_for_' +'_lr_'+str(args.base_lr)+'_bs_'+str(args.batch_size))
  if not os.path.isdir(checkpoint_folder): os.makedirs(checkpoint_folder, exist_ok=True)
  args.checkpoint_format = os.path.join(checkpoint_folder, args.checkpoint_format)
  # declaring the log fname
  current_log_folder = os.path.join(args.log_dir, args.descriptor_type + '/' + args.gan_name + '/hvd_log_for_' +'_lr_'+str(args.base_lr)+'_bs_'+str(args.batch_size))
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
    if os.path.exists(args.checkpoint_format.format('discriminator', epoch=try_epoch)):
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
comment='gan_type={args.gan_name}'#'loss_func={args.loss_func}_prior_type={args.prior_type}_wd={args.wd}'
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
#torch.set_num_threads(1)
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
# issues with Infiniband implementations that are not fork-safe
#if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
#      mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
#  kwargs['multiprocessing_context'] = 'forkserver'
train_dataset = DatasetFromHdf5(hvd, args.training_fname, hvd.size()*args.batch_size) 

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, \
	num_replicas=hvd.size(), rank=hvd.rank(), shuffle=args.shuffle_patches)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=allreduce_batch_size, \
	sampler=train_sampler, **kwargs)
if hvd.rank()==0: print("Dimension of (input <-> target) batches is: {} <-> {}".format(train_dataset.data.shape, \
  train_dataset.target.shape))
_, _, in_h, in_w = train_dataset.data.shape
_, _, tg_h, tg_w = train_dataset.target.shape 

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
# of the input and target and distributed over the given no ofpr gpus
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
criterionG = nn.MSELoss()
criterionD = nn.BCELoss()

#if (hvd.rank()==0): print(objloss)

# transfer models to cuda
if cuda:
  modelG, modelD         = modelG.cuda(), modelD.cuda()
  criterionG, criterionD = criterionG.cuda(), criterionD.cuda()
# MODEL SUMMARY
if hvd.rank()==0: 
  print('Architecture summary of the Generator Net')
  summary(modelG, (args.num_channels, in_h, in_w))
  print('Architecture summary of the Discriminator Net')
  summary(modelD, (args.num_channels, in_h, in_w))


# ==================================================================
# Initializing the optimizer type
# ==================================================================
  # adam uses its inbuilt subroutines to determine momemtum parameter
optimizerG = optim.Adam(modelG.parameters(), betas=(0.9, 0.999),
                  lr=(args.base_lr * args.batches_per_allreduce * hvd.size()))

optimizerD = optim.SGD(modelD.parameters(), momentum=0.9, nesterov=True, 
                  lr=((args.base_lr/100) * args.batches_per_allreduce * hvd.size()))


# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizerG = hvd.DistributedOptimizer(optimizerG, named_parameters=modelG.named_parameters(prefix='generator'), \
	compression=compression, backward_passes_per_step=args.batches_per_allreduce)
optimizerD = hvd.DistributedOptimizer(optimizerD, named_parameters=modelD.named_parameters(prefix='discriminator'), \
  compression=compression, backward_passes_per_step=args.batches_per_allreduce)

# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast weights to other workers.
if resume_from_epoch > 0 and hvd.rank() == 0:
    filepathG = args.checkpoint_format.format('generator', epoch=resume_from_epoch)
    filepathD = args.checkpoint_format.format('discriminator', epoch=resume_from_epoch)

    checkpointG = torch.load(filepathG)
    modelG.load_state_dict(checkpointG['model'])
    optimizerG.load_state_dict(checkpointG['optimizer'])
    checkpointD = torch.load(filepathD)
    modelD.load_state_dict(checkpointD['model'])
    optimizerD.load_state_dict(checkpointD['optimizer'])

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(modelG.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizerG, root_rank=0)

hvd.broadcast_parameters(modelD.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizerD, root_rank=0)

# ==================================================================
# train the model 
# optimer*.zero_grad(): sets the gradient of the optimizer to zero
# _.backward() : computes gradient w.r.t. current leaves (tensor)
# optimizer*.step() : updates the hyperparameter
# ==================================================================
def train(epoch):
  modelG.train()
  modelD.train()
  trainD_loss = util.Metric('trainD_loss')
  trainG_loss = util.Metric('trainG_loss')
  with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
      for batch_idx, (data, target) in enumerate(train_loader):
          
          #if hvd.rank()==0: print(batch_idx, data.shape, target.shape)
          #adjust_learning_rate as per epoch
          util.adjust_learning_rate_3_zones(epoch, 5, 15, args, optimizerG, hvd.size())
          util.adjust_learning_rate_3_zones(epoch, 5, 15, args, optimizerD, hvd.size())
          # of dim (batch_size, channel no)
          real_label = torch.ones(data.size(0), data.size(1))
          fake_label = torch.zeros(data.size(0), data.size(1))
          
          if cuda:
            data, target           = data.cuda(), target.cuda()
            real_label, fake_label = real_label.cuda(), fake_label.cuda()

          #discriminator training
          d_real = modelD(target)
          d_real_loss = criterionD(d_real, real_label)
          d_fake = modelD(modelG(data))
          d_fake_loss = criterionD(d_fake, fake_label) 
          d_total = d_real_loss + d_fake_loss
          trainD_loss.update(d_total, hvd)
          optimizerD.zero_grad()
          d_total.backward()
          optimizerD.step()
          optimizerG.synchronize()
          #generator training
          g_real = modelG(data)
          g_fake = modelD(g_real)
          gan_loss = criterionD(g_fake, real_label)
          mse_loss = criterionG(g_real, target)
          g_total = mse_loss + 1e-3* gan_loss
          trainG_loss.update(g_total, hvd)
          optimizerG.zero_grad()
          g_total.backward()
          #with optimizerG.skip_synchronize()
          optimizerG.step()

          optimizerD.synchronize()
          
          # loss options for cmd line view and in the log directory
          # tsum : total loss of all batches per GPU
          # avg  : avergage loss per GPU per batch
          t.set_postfix({'d_loss': trainD_loss.avg.item(), 
                         'g_loss': trainG_loss.avg.item()})
          t.update(1)
      #if hvd.rank()==0: 
      #  print('total Dloss of the epoch is', trainD_loss.tsum.item())
      #  print('total Gloss of the epoch is', trainG_loss.tsum.item())
  if log_writer:
    log_writer.add_scalar('train/d_loss', trainD_loss.avg, epoch)
    log_writer.add_scalar('train/g_loss', trainG_loss.avg, epoch)


# ==================================================================
# validate the network using metrics such as PSRN and SSIM
# ==================================================================
def validate(epoch):
  modelG.eval()
  val_psnr = util.Metric('val_psnr')
  val_ssim = util.Metric('val_ssim')

  with tqdm(total=len(val_loader),
            desc='Validate Epoch  #{}'.format(epoch + 1),
            disable=not verbose) as t:
      with torch.no_grad():
          for batch_num, (data, target) in enumerate(val_loader):
              if args.cuda: data, target = data.cuda(), target.cuda()
              output = modelG(data)
              #print('data dev type:', data.device, 'target dev type:', target.device, 'output dev type', output.device)
              #print('data, target, output shape', data.size(), target.size(), output.size())
              #if hvd.rank()==0: print(batch_num, data.shape, target.shape)
              _psnr, _ssim = util.quant_ana(output, target, args.val_chk_prsc) #quant ana converts gpu types to cpu inside its subroutine
              #val_loss.update(vloss, hvd)
              val_psnr.update(_psnr, hvd)
              val_ssim.update(_ssim, hvd)
              t.set_postfix({'PSNR ': val_psnr.avg.item(),
                             'SSIM': val_ssim.avg.item()})
              t.update(1)
            
  if log_writer:
        #log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/PSNR', val_psnr.avg, epoch)
        log_writer.add_scalar('val/SSIM', val_ssim.avg, epoch)


# ====================================================================
# Main function with train, validate and save weights
# ======================================================================
for epoch in range(resume_from_epoch, args.nepochs):
    train(epoch)
    validate(epoch)
    if (args.save_log_ckpts): 
      util.save_gan_checkpoint("discriminator", epoch, args, hvd, modelD, optimizerD)
      util.save_gan_checkpoint("generator", epoch, args, hvd, modelG, optimizerG)
