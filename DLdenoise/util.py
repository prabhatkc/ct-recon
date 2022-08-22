from skimage.metrics import structural_similarity as compare_ssim
import numpy as np 
import torch 
import matplotlib.pyplot as plt
import sys
import quant_util
#sys.path.append('/home/prabhat.kc/Implementations/python/')
#import global_files as gf 

# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate_via_batches(epoch, batch_idx, len_dataset, args, optimizer, nGPUS):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len_dataset
        lr_adj = 1. / nGPUS * (epoch * (nGPUS - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * nGPUS* args.batches_per_allreduce * lr_adj

# wd: 20, 40
def adjust_learning_rate_3_zones(epoch, ep1, ep2, args, optimizer, nGPUS):
    if epoch < ep1:
        lr_adj = 1
    elif epoch < ep2:
        lr_adj = 1e-1
    else:
        lr_adj = 1e-2
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * nGPUS * args.batches_per_allreduce * lr_adj

# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val, hvd):
        # here values such as mertic loss from outside training/validation 
        # loop for each batch is extracted. 
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

# save checkpoint from rank 0 so that weights from other ranks are
# not repeated
def save_checkpoint(epoch, args, hvd, model, optimizer):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)

def save_gan_checkpoint(gen_dis_str, epoch, args, hvd, model, optimizer):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(gen_dis_str, epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)

def normalize_data_ab(a, b, data):
    # input (min_data, max_data) with range (max_data - min_data) is normalized to (a, b)
    min_x = min(data.ravel())
    max_x = max(data.ravel())  
    range_x = max_x - min_x 
    return((b-a)*((data-min_x)/range_x)+a)

def normalize_data_ab_cd(a, b, c, d, data):
    # input data (min_data, max_data) with range (d-c) is normalized to (a, b)
    min_x = c
    max_x = d  
    range_x = max_x - min_x 
    return((b-a)*((data-min_x)/range_x)+a)

def multi2dplots(nrows, ncols, fig_arr, axis, passed_fig_att=None):
    """
      multi2dplots(1, 2, lena_stack, axis=0, passed_fig_att={'colorbar': False, 'split_title': np.asanyarray(['a','b']),'out_path': 'last_lr.tif'})
      where lena_stack is of size (2, 512, 512)
    """
    default_att= {"suptitle": '',
            "split_title": np.asanyarray(['']*(nrows*ncols)),
            "supfontsize": 12,
            "xaxis_vis"  : False,
            "yaxis_vis"  : False,
            "out_path"   : '',
            "figsize"    : [8, 8],
            "cmap"       : 'Greys_r',
            "plt_tight"  : True,
            "colorbar"   : True
                 }
    if passed_fig_att is None:
        fig_att = default_att
    else:
        fig_att = default_att
        for key, val in passed_fig_att.items():
            fig_att[key]=val
    
    f, axarr = plt.subplots(nrows, ncols, figsize = fig_att["figsize"])
    img_ind  = 0
    f.suptitle(fig_att["suptitle"], fontsize = fig_att["supfontsize"])
    for i in range(nrows):
        for j in range(ncols):                
            if (axis==0):
                each_img = fig_arr[img_ind, :, :]
            if (axis==1):
                each_img = fig_arr[:, img_ind, :]
            if (axis==2):
                each_img = fig_arr[:, :, img_ind]
                
            if(nrows==1):
                ax = axarr[j]
            elif(ncols ==1):
                ax =axarr[i]
            else:
                ax = axarr[i,j]
            im = ax.imshow(each_img, cmap = fig_att["cmap"])
            if fig_att["colorbar"] is True:  f.colorbar(im, ax=ax)
            ax.set_title(fig_att["split_title"][img_ind])
            ax.get_xaxis().set_visible(fig_att["xaxis_vis"])
            ax.get_yaxis().set_visible(fig_att["yaxis_vis"])
            img_ind = img_ind + 1
            if fig_att["plt_tight"] is True: plt.tight_layout()
            
    if (len(fig_att["out_path"])==0):
        plt.show()
    else:
        plt.savefig(fig_att["out_path"])

def quant_ana(output, target, img_type):
    cnn_output = output.cpu()
    target     = target.cpu()
    
    cnn_output = cnn_output[:, 0, :, :].detach().numpy()
    cnn_output = cnn_output.transpose(1, 2, 0)

    target = target[:, 0, :, :].detach().numpy()
    target = target.transpose(1, 2, 0)
    #print(cnn_output.shape, target.shape)
    #print(cnn_output.shape, target.shape)
    #multi2dplots(4, 4, cnn_output, axis=0)
    #multi2dplots(4, 4, target, axis=0)

    # if true each channel will be processed independently while determining
    # the ssim values and finally out will be an average of all these 
    # channels 
    multichannel=True
    if img_type == 'natural':
        model_out = np.uint8(cnn_output*255)
        target = np.uint8(target*255)
        _psnr = quant_util.psnr((model_out), np.uint8(target), max_val=255)
        _ssim = compare_ssim(model_out, np.uint8(target),  multichannel=multichannel, data_range=255)
    
    elif img_type == 'natural-float':
        model_out = np.float64(normalize_data_ab(0, 255, cnn_output))
        target = np.float64(target*255)
        _psnr = quant_util.psnr((model_out), np.float64(target), max_val=255.0)
        _ssim = compare_ssim(model_out, np.float64(target),  multichannel=multichannel, data_range=255)
    
    elif img_type == 'positive-float':
        
        model_out = np.float32(normalize_data_ab(0, 1, cnn_output))
        target = np.float32(normalize_data_ab(0, 1, target))
        _psnr = quant_util.psnr(model_out, np.float32(target), max_val=1.0)
        _ssim = compare_ssim(model_out, np.float32(target),  multichannel=multichannel, data_range=1)
    
    else: 	
        #just float (may have -ve values)
        model_out=(cnn_output*255)
        target = target*255
        _psnr = quant_util.psnr(np.float64(model_out), np.float64(target), max_val=255.0)
        _ssim = compare_ssim(model_out, np.float64(target),  multichannel=multichannel, data_range=255)
    _psnr = torch.tensor(_psnr)
    _ssim = torch.tensor(_ssim)
    return (_psnr.float(), _ssim.float())

def img_pair_normalization(input_image, target_image, normalization_type=None):
  
  if normalization_type == 'unity_independent':
    # both LD-HD pair are independently normalized to from (min_val, max_val) to (0, 1)
    out_input_image  = normalize_data_ab(0.0, 1.0, input_image)
    out_target_image = normalize_data_ab(0.0, 1.0, target_image)
    
  elif normalization_type == 'max_val_independent':
    # both LD-HD pair are independently normalized to from (min_val, max_val) to (min_val/max_val, min_val/max_val)
    if np.min(input_image)<0: input_image += (-np.min(input_image))
    if np.min(target_image)<0: target_image += (-np.min(target_image))
    out_input_image  =input_image/np.max(input_image)
    out_target_image =target_image/np.max(target_image)

  elif normalization_type == 'unity_wrt_ld':
    # both LD-HD pair are normalized to from (min_val, max_val) to (0, 1) with range (LD_max_val - LD_min_val)
    out_target_image = normalize_data_ab_cd(0.0, 1.0, np.min(input_image), np.max(input_image), target_image)
    out_input_image  = normalize_data_ab(0.0, 1.0, input_image)

  elif normalization_type == 'max_val_wrt_ld':
    # both LD-HD pair is normalized to from (min_val, max_val) to (min_val/LD_max_val, min_val/LD_max_val) 
    if np.min(input_image)<0: input_image += (-np.min(input_image))
    if np.min(target_image)<0: target_image += (-np.min(target_image))
    out_target_image =target_image/np.max(input_image)
    out_input_image =input_image/np.max(input_image)
  
  elif normalization_type == 'std_independent':
    # LD-HD pair is independently stardarized based on their respective values
    out_input_image  = (input_image - np.mean(input_image))/(np.max(input_image)-np.min(input_image))
    out_target_image = (target_image - np.mean(target_image))/(np.max(target_image)-np.min(target_image))

  elif normalization_type == 'std_wrt_ld':
    # LD-HD pair is jointly stardarized based on LD values 
    out_target_image = (target_image - np.mean(input_image))/(np.max(input_image)-np.min(input_image))
    out_input_image  = (input_image - np.mean(input_image))/(np.max(input_image)-np.min(input_image))
  
  # for dicom unity  as well as dicom_std everything has been scaled between (0, 2^16) hence there
  # is only one option for both and not the independent and wrt_ld types 
  elif normalization_type == 'dicom_unity':
    # all LD-HD pair are normalized between (0,1) while considering they exhibit (0, 2^16) initial value range
    out_input_image  = normalize_data_ab_cd(0.0, 1.0, 0.0, 2.0**12, input_image)
    out_target_image = normalize_data_ab_cd(0.0, 1.0, 0.0, 2.0**12, target_image)
  
  elif normalization_type == 'dicom_std':
    # all LD-HD pair are standarized while considering they exhibit (0, 2^16) initial value range
    out_input_image  = (input_image - np.mean(input_image))/(2.0**12)
    out_target_image = (target_image - np.mean(target_image))/(2.0**12)
  else: #none
    out_input_image  = input_image 
    out_target_image = target_image
  
  '''print("normalization_type is:", normalization_type)
  h, w = input_image.shape
  stacked = np.append(input_image.reshape(1, h, w), target_image.reshape(1, h, w), axis=0)
  gf.multi2dplots(1, 2, stacked, axis=0)
  sys.exit()
  '''
  return(out_input_image, out_target_image)

def plot2dlayers(arr, xlabel=None, ylabel=None, title=None, cmap=None, colorbar=True):
    """
    'brg' is the best colormap for reb-green-blue image
    'brg_r': in 'brg' colormap green color area will have
        high values whereas in 'brg_r' blue area will have
        the highest values
    """
    if xlabel is None:
        xlabel=''
    if ylabel is None:
        ylabel=''
    if title is None:
        title=''
    if cmap is None:
        cmap='Greys_r'
    plt.imshow(arr, cmap=cmap)
    cb = plt.colorbar()
    if colorbar is False:
      cb.remove()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    return


