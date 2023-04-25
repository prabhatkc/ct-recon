import scipy.misc
import numpy as np
import imageio
import pydicom
import os
import glob
import sys

def imread(path, mode='L', type=np.uint8, is_grayscale=True):
  """
  modes: ‘RGB’ (3x8-bit pixels, true color)
         'YCbCr’ (3x8-bit pixels, color video format)
         ‘L’ (8-bit pixels, black and white)

  Read image using its path.
  [*] Default value is gray-scale, 
      else image is read by YCbCr format as the paper said.
  [*] Also this reads images saved as (0-255) values
      for float format images use imageio_imread instead
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='L').astype(type)
  else:
    return scipy.misc.imread(path, mode=mode).astype(type)
  
def imageio_imread(path):
  """
   imageio based imread reads image in its orginal form even if its in
   - ve floats
  """
  return(imageio.imread(path))

def pydicom_imread(path):
  """ reads dicom image with filename path 
  and dtype be its original form
  """
  input_image = pydicom.dcmread(path)
  return(input_image.pixel_array.astype('float32'))

def raw_imread(path, shape=(256, 256), dtype='int16'):
  input_image = np.fromfile(path, dtype=dtype).astype('float32')
  input_image = input_image.reshape(shape)
  return(input_image)

def imsave(image, path, svtype=None):
  
  """
    imageio will save values in its orginal form even if its float
    if svtype='original' is specified
    else scipy save will save the image in (0 - 255 values)
    scipy new update has removed imsave from scipy.misc due
    to reported errors ... so just use imwrite from imageio 
    by declaring orginal and changing the data types accordingly
  """
  if svtype == "original":
    return(imageio.imwrite(path, image))
  else:
    return scipy.misc.imsave(path, image)

def imsave_raw(image, path):
  fileID = open(path, 'wb')
  image.tofile(fileID)
  fileID.close()

def getimages4rmdir(foldername, randN=None):
  ''' Returns image list (path) for an input
  directory. Sorted is true by default to remain consistant.
  randN is an array of len(images) whose [0, len(images)] index
  is randomly permuted.
  '''
  data_dir = os.path.join(os.getcwd(), foldername)
  images   = sorted(glob.glob(os.path.join(data_dir, "*.*")))
  if (len(images)==0): sys.exit("ERROR ! No images or incorrect image path.\n")

  if (randN !=None):
    images = np.array(images)
    images = list(images[randN]) 
  return images
