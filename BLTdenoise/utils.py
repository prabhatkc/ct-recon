
import glob
import os
import numpy as np
import imageio
import pydicom
import matplotlib.pyplot as plt
import scipy
import sys

def getimages4rmdir(foldername):
  # sorted is true by default to remain consistant 
  data_dir = os.path.join(os.getcwd(), foldername)
  images = sorted(glob.glob(os.path.join(data_dir, "*.*")))
  images = np.array(images)
  if len(images)==0:
    sys.exit("Error: No images found in the input path.")
  return images

def pydicom_imread(path):
  """ reads dicom image with filename path 
  and dtype be its original form
  """
  input_image = pydicom.dcmread(path)
  return(input_image.pixel_array.astype('float32'))


def imageio_imread(path):
  """
   imageio based imread reads image in its orginal form even if its in
   - ve floats
  """
  return(imageio.imread(path))

def normalize_data_ab(a, b, data):
    """
    input (min_data, max_data) with range 
    (max_data - min_data) is normalized to (a, b)
    """
    min_x = min(data.ravel())
    max_x = max(data.ravel())  
    range_x = max_x - min_x 
    return((b-a)*((data-min_x)/range_x)+a)

def tensor_2_img(tensor):
  img      = tensor.cpu()
  img      = img[0].detach().numpy()
  img      = np.squeeze(img.transpose(1, 2, 0))
  return img

def relative_mse(f_true, f_est):
    f_true = f_true.ravel()
    f_est = f_est.ravel()
    imdiff = f_true - f_est
    nume = np.sum(imdiff**2)
    deno = np.mean(f_true) - f_true
    deno = np.sum(deno**2)
    return(nume/deno)

def relative_error(f_old, f_new):
	# --------------------------------
	# ||X_old - X_new||_2/||X_new||_2
	# --------------------------------
	f_old  = f_old.ravel()
	f_new  = f_new.ravel()
	nume   = np.linalg.norm(f_old-f_new, 2)
	deno   = np.linalg.norm(f_new, 2)
	return (nume/deno)

def psnr(f_true, f_est, max_val=1.0):    
    imdff = f_true - f_est
    rmse = np.sqrt(np.mean(imdff **2))
    psnr = 20.0*np.log10(max_val/rmse)
    return(psnr)

def multi2dplots(nrows, ncols, fig_arr, axis, passed_fig_att=None):
    """
      gf.multi2dplots(1, 2, lena_stack, axis=0, passed_fig_att={'colorbar': False, 'split_title': np.asanyarray(['a','b']),'out_path': 'last_lr.tif'})
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
    return;

def imsave(image, path, type=None):
  
  """
    imageio will save values in its orginal form even if its float
    if type='orginal' is specified
    else scipy save will save the image in (0 - 255 values)
    scipy new update has removed imsave from scipy.misc due
    to reported errors ... so just use imwrite from imageio 
    by declaring orginal and changing the data types accordingly
  """
  if type is "original":
    return(imageio.imwrite(path, image))
  else:
    return scipy.misc.imsave(path, image)

def raw_imread(path, shape=(256, 256), dtype='int16'):
  input_image = np.fromfile(path, dtype=dtype).astype('float32')
  input_image = input_image.reshape(shape)
  return(input_image)

def imsave_raw(image, path):
  fileID = open(path, 'wb')
  image.tofile(fileID)
  fileID.close()